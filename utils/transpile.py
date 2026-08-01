import math
import operator
import os
import torch
import torch.export
from torch.export import export
from transformers import AutoModelForCausalLM, AutoConfig
from typing import Dict, Any, List

# ==============================================================================
# 1. Torch DType to C++ DType Mapping
# ==============================================================================
TORCH_DTYPE_TO_TG = {
    torch.float32: "DType::FLOAT32",
    torch.float: "DType::FLOAT32",
    torch.float16: "DType::BF16",
    torch.bfloat16: "DType::BF16",
    torch.int32: "DType::INT32",
    torch.int: "DType::INT32",
    torch.int64: "DType::INT64",
    torch.long: "DType::INT64",
    torch.bool: "DType::BOOL",
}

# ==============================================================================
# 2. Translation Rule Registry
# ==============================================================================
class TranslationRuleRegistry:
    def __init__(self):
        self.rules = []

    def register(self, matcher):
        def decorator(fn):
            self.rules.append((matcher, fn))
            return fn
        return decorator

    def apply(self, node, op_target_str, cpp_var, emitter):
        for matcher, fn in self.rules:
            if matcher(op_target_str, node):
                fn(node, op_target_str, cpp_var, emitter)
                return True
        return False

registry = TranslationRuleRegistry()


@registry.register(lambda op, node: any(skip_op in op for skip_op in (
    "_assert_tensor_metadata", "_assert", "detach", 
    "wrap_with_set_grad_enabled", "sym_size", "sym_stride", "sym_numel"
)))
def translate_pass_through(node, op_target_str, cpp_var, emitter):
    target_arg = None
    for arg in node.args:
        if isinstance(arg, torch.fx.Node):
            target_arg = arg
            break
    if target_arg is None:
        for arg in node.kwargs.values():
            if isinstance(arg, torch.fx.Node):
                target_arg = arg
                break
    if target_arg is not None:
        emitter.node_vars[node.name] = emitter.ensure_logical_id(target_arg)


@registry.register(lambda op, node: "getitem" in op or "get_item" in op)
def translate_getitem(node, op_target_str, cpp_var, emitter):
    arg0 = node.args[0]
    idx = node.args[1] if len(node.args) > 1 else 0
    if isinstance(arg0, torch.fx.Node) and arg0.name in emitter.node_vars:
        val = emitter.node_vars[arg0.name]
        if isinstance(val, (list, tuple)) and idx < len(val):
            emitter.node_vars[node.name] = val[idx]
        elif isinstance(val, str):
            emitter.node_vars[node.name] = val
    elif isinstance(arg0, (list, tuple)) and idx < len(arg0):
        emitter.node_vars[node.name] = emitter.ensure_logical_id(arg0[idx])


@registry.register(lambda op, node: "dropout" in op or "alias" in op)
def translate_dropout_alias(node, op_target_str, cpp_var, emitter):
    id0 = emitter.ensure_logical_id(node.args[0])
    emitter.node_vars[node.name] = id0


@registry.register(lambda op, node: "contiguous" in op)
def translate_contiguous(node, op_target_str, cpp_var, emitter):
    id0 = emitter.ensure_logical_id(node.args[0])
    emitter.code_lines.append(f"        LogicalId {cpp_var} = g.contiguous({id0});")


@registry.register(lambda op, node: "embedding" in op or "gather" in op)
def translate_embedding_gather(node, op_target_str, cpp_var, emitter):
    id0 = emitter.ensure_logical_id(node.args[0])
    id1 = emitter.ensure_logical_id(node.args[1])
    emitter.code_lines.append(f"        LogicalId {cpp_var} = g.gather({id0}, {id1});")


@registry.register(lambda op, node: any(c_op in op for c_op in ("to.dtype", "type_as", "cast", "_to_copy")))
def translate_cast(node, op_target_str, cpp_var, emitter):
    id0 = emitter.ensure_logical_id(node.args[0])
    target_dtype = "DType::FLOAT32"
    if len(node.args) > 1 and isinstance(node.args[1], torch.dtype):
        target_dtype = TORCH_DTYPE_TO_TG.get(node.args[1], "DType::FLOAT32")
    elif "dtype" in node.kwargs and isinstance(node.kwargs["dtype"], torch.dtype):
        target_dtype = TORCH_DTYPE_TO_TG.get(node.kwargs["dtype"], "DType::FLOAT32")
    else:
        target_dtype = emitter.get_target_dtype(node)
        
    # Extra check if casting to another node's dtype
    if len(node.args) > 1 and isinstance(node.args[1], torch.fx.Node):
        other_node = node.args[1]
        target_dtype = emitter.get_target_dtype(other_node)
        
    emitter.code_lines.append(f"        LogicalId {cpp_var} = g.cast({id0}, {target_dtype});")


@registry.register(lambda op, node: "mean" in op)
def translate_mean(node, op_target_str, cpp_var, emitter):
    id0 = emitter.ensure_logical_id(node.args[0])
    dim = -1
    if len(node.args) > 1 and node.args[1] is not None:
        if isinstance(node.args[1], (list, tuple)) and len(node.args[1]) > 0:
            dim = int(node.args[1][0])
        elif isinstance(node.args[1], int):
            dim = int(node.args[1])
    elif "dim" in node.kwargs:
        d = node.kwargs["dim"]
        if isinstance(d, (list, tuple)) and len(d) > 0:
            dim = int(d[0])
        elif isinstance(d, int):
            dim = int(d)

    dim_size = 1.0
    in_shape = emitter.get_target_shape(node.args[0])
    if in_shape:
        if dim < 0:
            dim += len(in_shape)
        if 0 <= dim < len(in_shape):
            dim_size = float(in_shape[dim])

    dim_id = emitter.ensure_logical_id(dim)
    sum_var = emitter.get_unique_var("sum")
    size_var = emitter.ensure_logical_id(dim_size)

    emitter.code_lines.append(f"        LogicalId {sum_var} = g.sum({id0}, {dim_id});")
    emitter.code_lines.append(f"        LogicalId {cpp_var} = g.div({sum_var}, {size_var});")


@registry.register(lambda op, node: "gelu" in op)
def translate_gelu(node, op_target_str, cpp_var, emitter):
    # TODO: This gelu is a bit unreadable, could use a helper for math ops
    x_id = emitter.ensure_logical_id(node.args[0])
    c1 = emitter.ensure_logical_id(0.044715)
    c2 = emitter.ensure_logical_id(0.7978845608028654)
    half = emitter.ensure_logical_id(0.5)
    one = emitter.ensure_logical_id(1.0)
    two = emitter.ensure_logical_id(2.0)
    neg_one = emitter.ensure_logical_id(-1.0)
    e_const = emitter.ensure_logical_id(2.718281828459045)

    x_sq = emitter.get_unique_var("x_sq")
    x_cube = emitter.get_unique_var("x_cube")
    t1 = emitter.get_unique_var("t1")
    t2 = emitter.get_unique_var("t2")
    t3 = emitter.get_unique_var("t3")
    two_u = emitter.get_unique_var("two_u")
    exp_2u = emitter.get_unique_var("exp_2u")
    num = emitter.get_unique_var("num")
    den = emitter.get_unique_var("den")
    tanh_v = emitter.get_unique_var("tanh_v")
    one_p_tanh = emitter.get_unique_var("one_p_tanh")
    half_x = emitter.get_unique_var("half_x")

    emitter.code_lines.append(f"        LogicalId {x_sq} = g.mul({x_id}, {x_id});")
    emitter.code_lines.append(f"        LogicalId {x_cube} = g.mul({x_sq}, {x_id});")
    emitter.code_lines.append(f"        LogicalId {t1} = g.mul({x_cube}, {c1});")
    emitter.code_lines.append(f"        LogicalId {t2} = g.add({x_id}, {t1});")
    emitter.code_lines.append(f"        LogicalId {t3} = g.mul({t2}, {c2});")
    emitter.code_lines.append(f"        LogicalId {two_u} = g.mul({t3}, {two});")
    emitter.code_lines.append(f"        LogicalId {exp_2u} = g.pow({e_const}, {two_u});")
    emitter.code_lines.append(f"        LogicalId {num} = g.add({exp_2u}, {neg_one});")
    emitter.code_lines.append(f"        LogicalId {den} = g.add({exp_2u}, {one});")
    emitter.code_lines.append(f"        LogicalId {tanh_v} = g.div({num}, {den});")
    emitter.code_lines.append(f"        LogicalId {one_p_tanh} = g.add({one}, {tanh_v});")
    emitter.code_lines.append(f"        LogicalId {half_x} = g.mul({x_id}, {half});")
    emitter.code_lines.append(f"        LogicalId {cpp_var} = g.mul({half_x}, {one_p_tanh});")


@registry.register(lambda op, node: "expand" in op)
def translate_expand(node, op_target_str, cpp_var, emitter):
    id0 = emitter.ensure_logical_id(node.args[0])
    target_shape = []
    if len(node.args) > 1 and isinstance(node.args[1], (list, tuple)):
        target_shape = [int(s) for s in node.args[1]]
    else:
        target_shape = emitter.get_target_shape(node, [])

    in_shape = emitter.get_target_shape(node.args[0], [])

    curr_var = id0
    if target_shape and in_shape:
        if len(in_shape) < len(target_shape):
            padded_in_shape = [1] * (len(target_shape) - len(in_shape)) + in_shape
            sh_var = f"shape_{cpp_var}_pad"
            emitter.code_lines.append(f"        int32_t {sh_var}[] = {{ {', '.join(map(str, padded_in_shape))} }};")
            curr_var = emitter.get_unique_var("pad")
            emitter.code_lines.append(f"        LogicalId {curr_var} = g.reshape({id0}, g.constant({{ {len(padded_in_shape)} }}, {sh_var}, DType::INT32));")
            curr_shape = padded_in_shape
        else:
            curr_shape = list(in_shape)

        for d in range(len(target_shape)):
            if d < len(curr_shape) and curr_shape[d] == 1 and target_shape[d] > 1:
                next_var = emitter.get_unique_var("rep")
                emitter.code_lines.append(f"        LogicalId {next_var} = g.repeat({curr_var}, {target_shape[d]}, {d});")
                curr_var = next_var
                curr_shape[d] = target_shape[d]

    emitter.node_vars[node.name] = curr_var


@registry.register(lambda op, node: any(f_op in op for f_op in ("new_ones", "ones", "new_zeros", "zeros", "full", "new_full")))
def translate_full(node, op_target_str, cpp_var, emitter):
    target_dtype = emitter.get_target_dtype(node)
    fill_val = 1.0
    if "zero" in op_target_str:
        fill_val = 0.0
    elif "full" in op_target_str:
        if len(node.args) > 1 and isinstance(node.args[1], (int, float)):
            fill_val = float(node.args[1])
        elif len(node.args) > 2 and isinstance(node.args[2], (int, float)):
            fill_val = float(node.args[2])

    val_id = emitter.ensure_logical_id(fill_val)
    target_shape = emitter.get_target_shape(node, [1])
    if len(node.args) > 1 and isinstance(node.args[1], (list, tuple)):
        target_shape = [int(s) for s in node.args[1]]

    shape_arr_var = f"shape_{cpp_var}"
    emitter.code_lines.append(f"        int32_t {shape_arr_var}[] = {{ {', '.join(map(str, target_shape))} }};")
    shape_id = emitter.get_unique_var("sh")
    emitter.code_lines.append(f"        LogicalId {shape_id} = g.constant({{ {len(target_shape)} }}, {shape_arr_var}, DType::INT32);")
    emitter.code_lines.append(f"        LogicalId {cpp_var} = g.fill({val_id}, {shape_id});")


@registry.register(lambda op, node: "diff" in op)
def translate_diff(node, op_target_str, cpp_var, emitter):
    id0 = emitter.ensure_logical_id(node.args[0])
    
    in_shape = emitter.get_target_shape(node.args[0], [])
    rank = len(in_shape) if in_shape else 2

    dim = -1
    if len(node.args) > 2 and isinstance(node.args[2], int):
        dim = node.args[2]
    elif "dim" in node.kwargs and isinstance(node.kwargs["dim"], int):
        dim = node.kwargs["dim"]

    if dim < 0:
        dim += rank

    starts1 = [0] * rank
    ends1 = [2147483647] * rank
    steps1 = [1] * rank
    starts1[dim] = 1

    starts0 = [0] * rank
    ends0 = [2147483647] * rank
    steps0 = [1] * rank
    ends0[dim] = -1

    s1_var = emitter.get_unique_var("s1")
    s0_var = emitter.get_unique_var("s0")
    neg_s0 = emitter.get_unique_var("neg_s0")

    emitter.code_lines.append(f"        int32_t starts1_{cpp_var}[] = {{ {', '.join(map(str, starts1))} }};")
    emitter.code_lines.append(f"        int32_t ends1_{cpp_var}[] = {{ {', '.join(map(str, ends1))} }};")
    emitter.code_lines.append(f"        int32_t steps1_{cpp_var}[] = {{ {', '.join(map(str, steps1))} }};")
    emitter.code_lines.append(f"        LogicalId {s1_var} = g.slice({id0}, g.constant({{ {rank} }}, starts1_{cpp_var}, DType::INT32), g.constant({{ {rank} }}, ends1_{cpp_var}, DType::INT32), g.constant({{ {rank} }}, steps1_{cpp_var}, DType::INT32));")

    emitter.code_lines.append(f"        int32_t starts0_{cpp_var}[] = {{ {', '.join(map(str, starts0))} }};")
    emitter.code_lines.append(f"        int32_t ends0_{cpp_var}[] = {{ {', '.join(map(str, ends0))} }};")
    emitter.code_lines.append(f"        int32_t steps0_{cpp_var}[] = {{ {', '.join(map(str, steps0))} }};")
    emitter.code_lines.append(f"        LogicalId {s0_var} = g.slice({id0}, g.constant({{ {rank} }}, starts0_{cpp_var}, DType::INT32), g.constant({{ {rank} }}, ends0_{cpp_var}, DType::INT32), g.constant({{ {rank} }}, steps0_{cpp_var}, DType::INT32));")

    emitter.code_lines.append(f"        LogicalId {neg_s0} = g.neg({s0_var});")
    emitter.code_lines.append(f"        LogicalId {cpp_var} = g.add({s1_var}, {neg_s0});")


@registry.register(lambda op, node: any(f"aten.{cmp}" in op for cmp in ("eq", "ne", "lt", "le", "gt", "ge")))
def translate_cmp(node, op_target_str, cpp_var, emitter):
    id0 = emitter.ensure_logical_id(node.args[0])
    id1 = emitter.ensure_logical_id(node.args[1]) if len(node.args) > 1 else emitter.ensure_logical_id(0)

    if "eq" in op_target_str and "ne" not in op_target_str and "seq" not in op_target_str:
        emitter.code_lines.append(f"        LogicalId {cpp_var} = g.eq({id0}, {id1});")
    elif "ne" in op_target_str:
        eq_var = emitter.get_unique_var("eq")
        emitter.code_lines.append(f"        LogicalId {eq_var} = g.eq({id0}, {id1});")
        emitter.code_lines.append(f"        LogicalId {cpp_var} = g.logical_not({eq_var});")
    elif "lt" in op_target_str:
        emitter.code_lines.append(f"        LogicalId {cpp_var} = g.lt({id0}, {id1});")
    elif "le" in op_target_str:
        gt_var = emitter.get_unique_var("gt")
        emitter.code_lines.append(f"        LogicalId {gt_var} = g.lt({id1}, {id0});")
        emitter.code_lines.append(f"        LogicalId {cpp_var} = g.logical_not({gt_var});")
    elif "gt" in op_target_str:
        emitter.code_lines.append(f"        LogicalId {cpp_var} = g.lt({id1}, {id0});")
    elif "ge" in op_target_str:
        lt_var = emitter.get_unique_var("lt")
        emitter.code_lines.append(f"        LogicalId {lt_var} = g.lt({id0}, {id1});")
        emitter.code_lines.append(f"        LogicalId {cpp_var} = g.logical_not({lt_var});")


@registry.register(lambda op, node: any(f"aten.{l_op}" in op or f"__{l_op}__" in op for l_op in ("and", "or", "not", "bitwise")))
def translate_logical(node, op_target_str, cpp_var, emitter):
    if "not" in op_target_str:
        id0 = emitter.ensure_logical_id(node.args[0])
        emitter.code_lines.append(f"        LogicalId {cpp_var} = g.logical_not({id0});")
    elif "and" in op_target_str:
        id0 = emitter.ensure_logical_id(node.args[0])
        id1 = emitter.ensure_logical_id(node.args[1])
        emitter.code_lines.append(f"        LogicalId {cpp_var} = g.logical_and({id0}, {id1});")
    elif "or" in op_target_str:
        id0 = emitter.ensure_logical_id(node.args[0])
        id1 = emitter.ensure_logical_id(node.args[1])
        emitter.code_lines.append(f"        LogicalId {cpp_var} = g.logical_or({id0}, {id1});")


@registry.register(lambda op, node: "where" in op)
def translate_where(node, op_target_str, cpp_var, emitter):
    cond_id = emitter.ensure_logical_id(node.args[0])
    x_id = emitter.ensure_logical_id(node.args[1])
    y_id = emitter.ensure_logical_id(node.args[2])

    target_dtype = emitter.get_target_dtype(node)

    cond_bool = emitter.get_unique_var("c_bool")
    cond_float = emitter.get_unique_var("c_flt")
    one_id = emitter.ensure_logical_id(1.0)
    not_cond = emitter.get_unique_var("not_c")
    t1 = emitter.get_unique_var("t1")
    t2 = emitter.get_unique_var("t2")

    emitter.code_lines.append(f"        LogicalId {cond_bool} = g.cast({cond_id}, DType::BOOL);")
    emitter.code_lines.append(f"        LogicalId {cond_float} = g.cast({cond_bool}, {target_dtype});")
    emitter.code_lines.append(f"        LogicalId {not_cond} = g.add({one_id}, g.neg({cond_float}));")
    emitter.code_lines.append(f"        LogicalId {t1} = g.mul({cond_float}, {x_id});")
    emitter.code_lines.append(f"        LogicalId {t2} = g.mul({not_cond}, {y_id});")
    emitter.code_lines.append(f"        LogicalId {cpp_var} = g.add({t1}, {t2});")


@registry.register(lambda op, node: "index" in op and "index_put" not in op)
def translate_index(node, op_target_str, cpp_var, emitter):
    data_id = emitter.ensure_logical_id(node.args[0])
    idx_args = node.args[1] if len(node.args) > 1 else []
    idx_tensor = None
    if isinstance(idx_args, (list, tuple)):
        for item in idx_args:
            if isinstance(item, torch.fx.Node):
                idx_tensor = item
                break
    elif isinstance(idx_args, torch.fx.Node):
        idx_tensor = idx_args

    if idx_tensor is not None:
        idx_id = emitter.ensure_logical_id(idx_tensor)
        emitter.code_lines.append(f"        LogicalId {cpp_var} = g.gather({data_id}, {idx_id});")


@registry.register(lambda op, node: "add" in op)
def translate_add(node, op_target_str, cpp_var, emitter):
    id0 = emitter.ensure_logical_id(node.args[0])
    id1 = emitter.ensure_logical_id(node.args[1])
    emitter.code_lines.append(f"        LogicalId {cpp_var} = g.add({id0}, {id1});")


@registry.register(lambda op, node: "sub" in op)
def translate_sub(node, op_target_str, cpp_var, emitter):
    id0 = emitter.ensure_logical_id(node.args[0])
    id1 = emitter.ensure_logical_id(node.args[1])
    neg_var = emitter.get_unique_var("neg")
    emitter.code_lines.append(f"        LogicalId {neg_var} = g.neg({id1});")
    emitter.code_lines.append(f"        LogicalId {cpp_var} = g.add({id0}, {neg_var});")


@registry.register(lambda op, node: "mul" in op)
def translate_mul(node, op_target_str, cpp_var, emitter):
    id0 = emitter.ensure_logical_id(node.args[0])
    id1 = emitter.ensure_logical_id(node.args[1])
    emitter.code_lines.append(f"        LogicalId {cpp_var} = g.mul({id0}, {id1});")


@registry.register(lambda op, node: "div" in op)
def translate_div(node, op_target_str, cpp_var, emitter):
    id0 = emitter.ensure_logical_id(node.args[0])
    id1 = emitter.ensure_logical_id(node.args[1])
    emitter.code_lines.append(f"        LogicalId {cpp_var} = g.div({id0}, {id1});")


@registry.register(lambda op, node: "pow" in op)
def translate_pow(node, op_target_str, cpp_var, emitter):
    id0 = emitter.ensure_logical_id(node.args[0])
    id1 = emitter.ensure_logical_id(node.args[1])
    emitter.code_lines.append(f"        LogicalId {cpp_var} = g.pow({id0}, {id1});")


@registry.register(lambda op, node: any(mm_op in op for mm_op in ("mm", "bmm", "matmul", "linear", "addmm")))
def translate_dot(node, op_target_str, cpp_var, emitter):
    id0 = emitter.ensure_logical_id(node.args[0])
    id1 = emitter.ensure_logical_id(node.args[1])
    emitter.code_lines.append(f"        LogicalId {cpp_var} = g.dot({id0}, {id1});")


@registry.register(lambda op, node: "neg" in op)
def translate_neg(node, op_target_str, cpp_var, emitter):
    id0 = emitter.ensure_logical_id(node.args[0])
    emitter.code_lines.append(f"        LogicalId {cpp_var} = g.neg({id0});")


@registry.register(lambda op, node: "sin" in op)
def translate_sin(node, op_target_str, cpp_var, emitter):
    id0 = emitter.ensure_logical_id(node.args[0])
    emitter.code_lines.append(f"        LogicalId {cpp_var} = g.sin({id0});")


@registry.register(lambda op, node: "cos" in op)
def translate_cos(node, op_target_str, cpp_var, emitter):
    id0 = emitter.ensure_logical_id(node.args[0])
    emitter.code_lines.append(f"        LogicalId {cpp_var} = g.cos({id0});")


@registry.register(lambda op, node: "log" in op)
def translate_log(node, op_target_str, cpp_var, emitter):
    id0 = emitter.ensure_logical_id(node.args[0])
    emitter.code_lines.append(f"        LogicalId {cpp_var} = g.log({id0});")


@registry.register(lambda op, node: any(r_op in op for r_op in ("view", "reshape", "unsqueeze", "squeeze")))
def translate_reshape(node, op_target_str, cpp_var, emitter):
    id0 = emitter.ensure_logical_id(node.args[0])
    shape_list = []
    if len(node.args) > 1 and isinstance(node.args[1], (list, tuple)):
        shape_list = [int(s) if isinstance(s, (int, float)) else -1 for s in node.args[1]]
    else:
        shape_list = emitter.get_target_shape(node, [1])

    # TODO: This clamp logic to int32 is slightly odd and might mask shape bugs for massive dimensions
    def clamp_to_int32(v):
        return max(-2147483648, min(2147483647, v))

    shape_list = [clamp_to_int32(int(s)) if isinstance(s, (int, float)) else -1 for s in shape_list]

    shape_arr_var = f"shape_{cpp_var}"
    emitter.code_lines.append(f"        int32_t {shape_arr_var}[] = {{ {', '.join(map(str, shape_list))} }};")
    emitter.code_lines.append(f"        LogicalId {cpp_var} = g.reshape({id0}, g.constant({{ {len(shape_list)} }}, {shape_arr_var}, DType::INT32));")


@registry.register(lambda op, node: "permute" in op or "transpose" in op or op.endswith(".t"))
def translate_permute(node, op_target_str, cpp_var, emitter):
    id0 = emitter.ensure_logical_id(node.args[0])
    if "transpose" in op_target_str or op_target_str.endswith(".t"):
        # TODO: Also assumes rank 4 if shape isn't available
        in_shape = emitter.get_target_shape(node.args[0], [])
        rank = len(in_shape) if in_shape else 4
        d0 = node.args[1] if len(node.args) > 1 else 0
        d1 = node.args[2] if len(node.args) > 2 else 1
        if isinstance(d0, int) and d0 < 0:
            d0 += rank
        if isinstance(d1, int) and d1 < 0:
            d1 += rank
        perm_dims = list(range(rank))
        if isinstance(d0, int) and isinstance(d1, int) and d0 < rank and d1 < rank:
            perm_dims[d0], perm_dims[d1] = perm_dims[d1], perm_dims[d0]
    else:
        perm_dims = [int(d) for d in node.args[1]] if len(node.args) > 1 and isinstance(node.args[1], (list, tuple)) else [0, 1]

    # TODO: See previous shape clamping logic
    def clamp_to_int32(v):
        return max(-2147483648, min(2147483647, v))

    perm_dims = [clamp_to_int32(int(d)) for d in perm_dims]

    perm_arr_var = f"perm_{cpp_var}"
    emitter.code_lines.append(f"        int32_t {perm_arr_var}[] = {{ {', '.join(map(str, perm_dims))} }};")
    emitter.code_lines.append(f"        LogicalId {cpp_var} = g.permute({id0}, g.constant({{ {len(perm_dims)} }}, {perm_arr_var}, DType::INT32));")


@registry.register(lambda op, node: "slice" in op or "select" in op)
def translate_slice(node, op_target_str, cpp_var, emitter):
    id0 = emitter.ensure_logical_id(node.args[0])
    # TODO: Also assumes rank 4 if shape isn't available
    in_shape = emitter.get_target_shape(node.args[0], [])
    rank = len(in_shape) if in_shape else 4

    dim = int(node.args[1]) if len(node.args) > 1 and isinstance(node.args[1], int) else 0
    if dim < 0:
        dim += rank

    start = int(node.args[2]) if len(node.args) > 2 and node.args[2] is not None and isinstance(node.args[2], int) else 0
    end = int(node.args[3]) if len(node.args) > 3 and node.args[3] is not None and isinstance(node.args[3], int) else 2147483647
    step = int(node.args[4]) if len(node.args) > 4 and node.args[4] is not None and isinstance(node.args[4], int) else 1

    def clamp_to_int32(v):
        return max(-2147483648, min(2147483647, v))

    start = clamp_to_int32(start)
    end = clamp_to_int32(end)
    step = clamp_to_int32(step)

    starts = [0] * rank
    ends = [2147483647] * rank
    steps = [1] * rank

    if dim < rank:
        starts[dim] = start
        ends[dim] = end
        steps[dim] = step

    emitter.code_lines.append(f"        int32_t starts_{cpp_var}[] = {{ {', '.join(map(str, starts))} }};")
    emitter.code_lines.append(f"        int32_t ends_{cpp_var}[] = {{ {', '.join(map(str, ends))} }};")
    emitter.code_lines.append(f"        int32_t steps_{cpp_var}[] = {{ {', '.join(map(str, steps))} }};")
    emitter.code_lines.append(
        f"        LogicalId {cpp_var} = g.slice({id0}, "
        f"g.constant({{ {len(starts)} }}, starts_{cpp_var}, DType::INT32), "
        f"g.constant({{ {len(ends)} }}, ends_{cpp_var}, DType::INT32), "
        f"g.constant({{ {len(steps)} }}, steps_{cpp_var}, DType::INT32));"
    )


@registry.register(lambda op, node: "cat" in op or "concat" in op)
def translate_concat(node, op_target_str, cpp_var, emitter):
    tensor_nodes = node.args[0] if len(node.args) > 0 and isinstance(node.args[0], (list, tuple)) else []
    cpp_tensors = [emitter.ensure_logical_id(t) for t in tensor_nodes]
    dim = int(node.args[1]) if len(node.args) > 1 and isinstance(node.args[1], int) else 0

    axis_var = f"axis_{cpp_var}"
    emitter.code_lines.append(f"        int32_t {axis_var} = {dim};")
    emitter.code_lines.append(f"        LogicalId {cpp_var} = g.concat({{ {', '.join(cpp_tensors)} }}, g.constant({{ 1 }}, &{axis_var}, DType::INT32));")


@registry.register(lambda op, node: "arange" in op)
def translate_arange(node, op_target_str, cpp_var, emitter):
    start, stop, step = 0, 10, 1
    if len(node.args) == 1:
        stop = int(node.args[0]) if isinstance(node.args[0], int) else 10
    elif len(node.args) >= 2:
        start = int(node.args[0]) if isinstance(node.args[0], int) else 0
        stop = int(node.args[1]) if isinstance(node.args[1], int) else 10
    if len(node.args) >= 3:
        step = int(node.args[2]) if isinstance(node.args[2], int) else 1

    start_id = emitter.ensure_logical_id(start)
    stop_id = emitter.ensure_logical_id(stop)
    step_id = emitter.ensure_logical_id(step)

    emitter.code_lines.append(f"        LogicalId {cpp_var} = g.arange({start_id}, {stop_id}, {step_id});")


@registry.register(lambda op, node: "sum" in op or "max" in op)
def translate_reduction(node, op_target_str, cpp_var, emitter):
    id0 = emitter.ensure_logical_id(node.args[0])
    dim = 0
    if len(node.args) > 1:
        if isinstance(node.args[1], (list, tuple)) and len(node.args[1]) > 0:
            dim = int(node.args[1][0])
        elif isinstance(node.args[1], int):
            dim = int(node.args[1])
    dim_id = emitter.ensure_logical_id(dim)
    method = "sum" if "sum" in op_target_str else "max"
    emitter.code_lines.append(f"        LogicalId {cpp_var} = g.{method}({id0}, {dim_id});")


@registry.register(lambda op, node: "rsqrt" in op)
def translate_rsqrt(node, op_target_str, cpp_var, emitter):
    id0 = emitter.ensure_logical_id(node.args[0])
    neg_half_id = emitter.ensure_logical_id(-0.5)
    emitter.code_lines.append(f"        LogicalId {cpp_var} = g.pow({id0}, {neg_half_id});")



# ==============================================================================
# 2. C++ Code Generator Emitter
# ==============================================================================
class TensorGraphCodeEmitter:
    def __init__(self, model_name: str, weight_path: str):
        self.model_name = model_name
        self.weight_path = weight_path
        self.code_lines: List[str] = []
        self.node_vars: Dict[str, Any] = {}  # FX Node Name -> C++ LogicalId variable or tuple/list
        self.var_counter = 0

    def get_unique_var(self, prefix: str = "tmp") -> str:
        self.var_counter += 1
        return f"{prefix}_{self.var_counter}"

    def emit_header(self):
        self.code_lines.extend(
            [
                "#pragma once",
                '#include "core/graph.hpp"',
                '#include "core/types.hpp"',
                "",
                "// Auto-generated by transpile.py using torch.export",
                f"class {self.model_name}Generated {{",
                "public:",
                "    Graph &g;",
                "    const std::string w_path;",
                "",
                f"    {self.model_name}Generated(Graph &graph, const std::string &weight_path)",
                "        : g(graph), w_path(weight_path) {}",
                "",
                "    LogicalId build_graph(LogicalId input_ids) {",
            ]
        )

    def emit_footer(self, final_var_name: str) -> str:
        final_cpp_var = self.node_vars.get(final_var_name, "LogicalId()")
        if isinstance(final_cpp_var, (list, tuple)):
            final_cpp_var = final_cpp_var[0]
        self.code_lines.extend([f"        return {final_cpp_var};", "    }", "};"])
        return "\n".join(self.code_lines)

    
    def get_target_dtype(self, node, default="DType::FLOAT32"):
        if hasattr(node, "meta") and "val" in node.meta and hasattr(node.meta["val"], "dtype"):
            return TORCH_DTYPE_TO_TG.get(node.meta["val"].dtype, default)
        return default

    def get_target_shape(self, node, default=None):
        if hasattr(node, "meta") and "val" in node.meta and hasattr(node.meta["val"], "shape"):
            return [int(s) for s in node.meta["val"].shape]
        return default
    

    def register_node(self, fx_node_name: str, cpp_var: str = "") -> str:
        clean_name = cpp_var if cpp_var else f"var_{fx_node_name.replace('.', '_')}"
        self.node_vars[fx_node_name] = clean_name
        return clean_name

    def ensure_logical_id(self, arg: Any) -> str:
        """Ensures an argument is converted to a C++ LogicalId variable name."""
        if isinstance(arg, torch.fx.Node):
            if arg.name in self.node_vars:
                res = self.node_vars[arg.name]
                if isinstance(res, (list, tuple)):
                    return res[0]
                return str(res)
            return f"var_{arg.name.replace('.', '_')}"
        elif isinstance(arg, (list, tuple)):
            items = [self.ensure_logical_id(item) for item in arg]
            return f"{{{', '.join(items)}}}"
        elif isinstance(arg, bool):
            v_name = self.get_unique_var("bool")
            val_str = "true" if arg else "false"
            self.code_lines.append(f"        bool val_{v_name} = {val_str};")
            self.code_lines.append(
                f"        LogicalId {v_name} = g.constant({{1}}, &val_{v_name}, DType::BOOL);"
            )
            return v_name
        elif isinstance(arg, int):
            v_name = self.get_unique_var("int")
            self.code_lines.append(f"        int32_t val_{v_name} = {arg};")
            self.code_lines.append(
                f"        LogicalId {v_name} = g.constant({{1}}, &val_{v_name}, DType::INT32);"
            )
            return v_name
        elif isinstance(arg, float):
            v_name = self.get_unique_var("float")
            if math.isnan(arg):
                val_str = "0.0f"
            elif math.isinf(arg):
                val_str = "-3.3895313892515355e+38f" if arg < 0 else "3.3895313892515355e+38f"
            else:
                val_str = f"{arg:.9g}"
                if "." not in val_str and "e" not in val_str and "E" not in val_str:
                    val_str += ".0"
                val_str += "f"
            self.code_lines.append(f"        float val_{v_name} = {val_str};")
            self.code_lines.append(
                f"        LogicalId {v_name} = g.constant({{1}}, &val_{v_name}, DType::FLOAT32);"
            )
            return v_name
        elif isinstance(arg, str) and (arg in self.node_vars.values() or arg.startswith("var_") or arg == "input_ids"):
            return arg
        else:
            v_name = self.get_unique_var("val")
            self.code_lines.append(
                f"        float val_{v_name} = static_cast<float>({arg});"
            )
            self.code_lines.append(
                f"        LogicalId {v_name} = g.constant({{1}}, &val_{v_name}, DType::FLOAT32);"
            )
            return v_name

    def transpile_tensor_to_constant(self, node_name: str, tensor: torch.Tensor) -> str:
        """Formats and emits a PyTorch tensor as a C++ g.constant(...) call with actual data."""
        shape = list(tensor.shape)
        dtype = tensor.dtype

        if dtype in (torch.float32, torch.float16, torch.bfloat16):
            cpp_dtype = "DType::FLOAT32"
            cpp_type = "float"
            suffix = "f"
            zero_val = "0.0f"
        elif dtype in (torch.int32, torch.int16, torch.uint8, torch.int8):
            cpp_dtype = "DType::INT32"
            cpp_type = "int32_t"
            suffix = ""
            zero_val = "0"
        elif dtype in (torch.int64,):
            cpp_dtype = "DType::INT64"
            cpp_type = "int64_t"
            suffix = "LL"
            zero_val = "0LL"
        elif dtype in (torch.bool,):
            cpp_dtype = "DType::BOOL"
            cpp_type = "bool"
            suffix = ""
            zero_val = "false"
        else:
            cpp_dtype = "DType::FLOAT32"
            cpp_type = "float"
            suffix = "f"
            zero_val = "0.0f"

        cpp_var = self.register_node(node_name)
        shape_str = f"{{{', '.join(map(str, shape))}}}"
        num_elements = tensor.numel()

        # If on meta device, fall back to g.fill zero constant
        if getattr(tensor, "is_meta", False) or tensor.device.type == "meta":
            zero_var = self.get_unique_var("zero")
            shape_var = self.get_unique_var("sh")
            shape_arr = f"shape_arr_{cpp_var}"

            self.code_lines.append(f"        {cpp_type} val_{zero_var} = {zero_val};")
            self.code_lines.append(f"        LogicalId {zero_var} = g.constant({{1}}, &val_{zero_var}, {cpp_dtype});")

            if num_elements == 0:
                self.code_lines.append(f"        LogicalId {cpp_var} = g.constant({shape_str}, nullptr, {cpp_dtype});")
            else:
                self.code_lines.append(f"        int32_t {shape_arr}[] = {shape_str};")
                self.code_lines.append(f"        LogicalId {shape_var} = g.constant({{ {len(shape)} }}, {shape_arr}, DType::INT32);")
                self.code_lines.append(f"        LogicalId {cpp_var} = g.fill({zero_var}, {shape_var});")
            return cpp_var

        # Extract actual data for real tensors
        # TODO: Fix bf16 support when converting to float32 explicitly as bfloat16 arrays vs casts
        tensor_cpu = tensor.detach().cpu().resolve_conj().resolve_neg()

        # Optimization for large all-zero tensors (keeps compile sizes reasonable)
        # TODO: This optimization is a bit hacky to keep compile sizes reasonable for large all-zero tensors
        if num_elements > 1024 and torch.all(tensor_cpu == 0):
            zero_var = self.get_unique_var("zero")
            shape_var = self.get_unique_var("sh")
            shape_arr = f"shape_arr_{cpp_var}"

            self.code_lines.append(f"        {cpp_type} val_{zero_var} = {zero_val};")
            self.code_lines.append(f"        LogicalId {zero_var} = g.constant({{1}}, &val_{zero_var}, {cpp_dtype});")
            self.code_lines.append(f"        int32_t {shape_arr}[] = {shape_str};")
            self.code_lines.append(f"        LogicalId {shape_var} = g.constant({{ {len(shape)} }}, {shape_arr}, DType::INT32);")
            self.code_lines.append(f"        LogicalId {cpp_var} = g.fill({zero_var}, {shape_var});")
            return cpp_var

        if cpp_type == "float":
            flat_data = tensor_cpu.to(torch.float32).numpy().flatten()
        elif cpp_type == "int32_t":
            flat_data = tensor_cpu.to(torch.int32).numpy().flatten()
        elif cpp_type == "int64_t":
            flat_data = tensor_cpu.to(torch.int64).numpy().flatten()
        elif cpp_type == "bool":
            flat_data = tensor_cpu.to(torch.bool).numpy().flatten()
        else:
            flat_data = tensor_cpu.to(torch.float32).numpy().flatten()

        if len(flat_data) == 0:
            self.code_lines.append(f"        LogicalId {cpp_var} = g.constant({shape_str}, nullptr, {cpp_dtype});")
            return cpp_var

        arr_name = f"arr_{cpp_var}"

        if cpp_type == "bool":
            # TODO: Improve boolean string formatting to avoid massive long sequences of "false"
            elements_str = ", ".join("true" if val else "false" for val in flat_data)
        elif cpp_type == "float":
            def format_float(v):
                if math.isnan(v):
                    return "0.0f"
                if math.isinf(v):
                    return "-3.3895313892515355e+38f" if v < 0 else "3.3895313892515355e+38f"
                s = f"{v:.9g}"
                if "." not in s and "e" not in s and "E" not in s:
                    s += ".0"
                return s + "f"
            elements_str = ", ".join(format_float(val) for val in flat_data)
        else:
            elements_str = ", ".join(f"{val}{suffix}" for val in flat_data)

        self.code_lines.append(f"        {cpp_type} {arr_name}[] = {{ {elements_str} }};")
        self.code_lines.append(f"        LogicalId {cpp_var} = g.constant({shape_str}, {arr_name}, {cpp_dtype});")
        return cpp_var


# 1. Define a clean wrapper to strip HF custom classes & disable KV Cache
class GemmaExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        # use_cache=False: prevents creation of DynamicCache
        # return_dict=False: returns raw tuple (logits, ...) instead of CausalLMOutput
        outputs = self.model(input_ids, use_cache=False, return_dict=False)

        # Return only the logits Tensor so torch.export sees pure Tensor outputs
        logits = outputs[0]
        return logits


def transpile_gemma_3():
    model_path = "./models/google/gemma-3-270m"
    if not os.path.exists(model_path):
        model_path = "google/gemma-3-270m"

    print(f"[1/4] Reading Config from: {model_path}...")

    # 1. Load configuration
    config = AutoConfig.from_pretrained(model_path)

    # 2. Instantiate model on CPU so buffers and constants have actual real data
    model = AutoModelForCausalLM.from_config(
        config, attn_implementation="eager"  # Use standard eager attention
    )
    model.eval()

    wrapped_model = GemmaExportWrapper(model)

    # 3. Create dummy inputs on CPU
    dummy_input_ids = torch.ones((1, 8), dtype=torch.int32)

    print("[2/4] Exporting model with torch.export.export()...")

    # PyTorch natively traces the model symbolically
    exported_program: torch.export.ExportedProgram = export(
        wrapped_model, (dummy_input_ids,)
    )

    print("Model export successful!")

    # 4. Extract signature and graph module
    graph_module = exported_program.graph_module
    signature = exported_program.graph_signature

    # All parameter names and graph operations are fully intact!
    # TODO: This mapping and prefix removal could be cleaner and more robust
    # Note: `lm_head.weight` comes directly from the parameter registered in PyTorch module (even if it's tied inside model.embed_tokens)
    # Build an ID -> Canonical Name mapping from the loaded model
    canonical_param_names = {}
    for name, param in model.named_parameters():
        if id(param) not in canonical_param_names:
            # The first time we see this tensor memory, record its name
            # (e.g., 'model.embed_tokens.weight' is seen before 'lm_head.weight')
            canonical_param_names[id(param)] = name

    param_map = {}
    for node_name, param_name in signature.inputs_to_parameters.items():
        # Get the actual parameter from the wrapper
        actual_param = wrapped_model.get_parameter(param_name)
        # Look up the canonical name using its memory ID
        canon_name = canonical_param_names[id(actual_param)]
        param_map[node_name] = canon_name
        
    for node_name, buffer_name in signature.inputs_to_buffers.items():
        actual_buf = wrapped_model.get_buffer(buffer_name)
        canon_name = canonical_param_names.get(
            id(actual_buf), buffer_name.removeprefix("model.")
        )
        param_map[node_name] = canon_name

    print("[3/4] Lowering FX ATen graph to tensor_graphs_cpp...")
    emitter = TensorGraphCodeEmitter(
        "Gemma3_270M", "models/gemma-3-270m/model.safetensors"
    )
    emitter.emit_header()

    # Traverse FX graph nodes in topological execution order
    for node in graph_module.graph.nodes:
        # A. Placeholder Nodes (Inputs vs Weights vs Constants)
        if node.op == "placeholder":
            if node.name in signature.user_inputs:
                # Runtime input (e.g. input_ids)
                cpp_var = emitter.register_node(node.name, "input_ids")
                emitter.code_lines.append(f"        // Input: {node.name}")
                emitter.node_vars[node.name] = "input_ids"
            elif node.name in param_map:
                # Model weight -> g.weight(w_path, "model.layers.0...")
                hf_weight_key = param_map[node.name]
                cpp_var = emitter.register_node(node.name)
                emitter.code_lines.append(
                    f'        LogicalId {cpp_var} = g.cast(g.weight(w_path, "{hf_weight_key}"), DType::FLOAT32);'
                )
            elif node.name in getattr(signature, "inputs_to_lifted_tensor_constants", {}):
                # Lifted tensor constant!
                const_name = signature.inputs_to_lifted_tensor_constants[node.name]
                if hasattr(exported_program, "constants") and exported_program.constants is not None and const_name in exported_program.constants:
                    tensor = exported_program.constants[const_name]
                    emitter.transpile_tensor_to_constant(node.name, tensor)
                else:
                    cpp_var = emitter.register_node(node.name)
                    emitter.code_lines.append(f"        float val_{cpp_var} = 0.0f;")
                    emitter.code_lines.append(f"        LogicalId {cpp_var} = g.constant({{1}}, &val_{cpp_var}, DType::FLOAT32);")
            else:
                print(f"WARNING: Unknown placeholder '{node.name}' - defaulting to 0.0f")
                cpp_var = emitter.register_node(node.name)
                emitter.code_lines.append(f"        float val_{cpp_var} = 0.0f;")
                emitter.code_lines.append(f"        LogicalId {cpp_var} = g.constant({{1}}, &val_{cpp_var}, DType::FLOAT32);")

        elif node.op == "get_attr":
            attr = getattr(graph_module, node.target, None)
            if isinstance(attr, torch.Tensor):
                emitter.transpile_tensor_to_constant(node.name, attr)
            else:
                cpp_var = emitter.register_node(node.name)
                emitter.code_lines.append(f"        float val_{cpp_var} = 0.0f;")
                emitter.code_lines.append(f"        LogicalId {cpp_var} = g.constant({{1}}, &val_{cpp_var}, DType::FLOAT32);")

        # B. Call Function Nodes (ATen Ops)
        elif node.op == "call_function":
            op_target_str = str(node.target)
            cpp_var = emitter.register_node(node.name)

            if registry.apply(node, op_target_str, cpp_var, emitter):
                 continue

            print(f"WARNING: Unknown function '{op_target_str}' in node '{node.name}'")
            emitter.code_lines.append(
                f"        // UNHANDLED ATen Op: {op_target_str} for node {node.name}"
            )
            fallback_arg = None
            for arg in node.args:
                if isinstance(arg, torch.fx.Node):
                    fallback_arg = emitter.ensure_logical_id(arg)
                    break
            if fallback_arg is None:
                fallback_arg = "LogicalId()"
            emitter.code_lines.append(f"        LogicalId {cpp_var} = {fallback_arg};")

        # C. Output Node
        elif node.op == "output":
            final_node = (
                node.args[0][0]
                if isinstance(node.args[0], (list, tuple))
                else node.args[0]
            )
            emitter.emit_footer(
                final_node.name
                if isinstance(final_node, torch.fx.Node)
                else str(final_node)
            )

    print("[4/4] Writing generated C++ header file...")
    cpp_source = emitter.code_lines
    output_filename = "tensor_graphs_cpp/models/gemma-3-270m-generated.hpp"
    with open(output_filename, "w") as f:
        f.write("\n".join(cpp_source))
    print(f"Successfully generated: {output_filename}")


if __name__ == "__main__":
    transpile_gemma_3()