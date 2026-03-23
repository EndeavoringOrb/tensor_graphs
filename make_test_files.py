import os
import json
import torch
import shutil
from safetensors.torch import save_file

TEST_DIR = "tensor_graphs_cpp/tests"


def dtype_to_str(dt):
    if dt == torch.float32:
        return "FLOAT32"
    if dt == torch.int32:
        return "INT32"
    if dt == torch.bfloat16:
        return "BF16"
    if dt == torch.bool:
        return "BOOL"
    raise ValueError(f"Unknown dtype: {dt}")


tests = []


def add_test(op, inputs, output):
    tests.append((op, inputs, output))


# --- Element-wise ---
a = torch.rand((4, 8), dtype=torch.float32)
b = torch.rand((4, 8), dtype=torch.float32)
add_test("ADD", [a, b], a + b)
add_test("MUL", [a, b], a * b)
add_test("DIVIDE", [a, b], a / b)
add_test("POWER", [a, b], a**b)
add_test("SIN", [a], torch.sin(a))
add_test("COS", [a], torch.cos(a))
add_test("NEGATE", [a], -a)

# --- DOT (Batched 3D) ---
a_dot = torch.rand((2, 4, 8), dtype=torch.float32)
b_dot = torch.rand((2, 8, 4), dtype=torch.float32)
add_test("DOT", [a_dot, b_dot], torch.matmul(a_dot, b_dot))

# --- Reduce ---
a_sum = torch.rand((4, 8, 4), dtype=torch.float32)
axis_sum = torch.tensor([-1], dtype=torch.int32)
add_test("SUM", [a_sum, axis_sum], torch.sum(a_sum, dim=-1, keepdim=True))
add_test("MAX", [a_sum, axis_sum], torch.max(a_sum, dim=-1, keepdim=True).values)

# --- Manipulation ---
a_res = torch.rand((4, 8), dtype=torch.float32)
target_dims = torch.tensor([2, 16], dtype=torch.int32)
add_test("RESHAPE", [a_res, target_dims], a_res.reshape((2, 16)))

a_perm = torch.rand((2, 4, 8), dtype=torch.float32)
perm = torch.tensor([0, 2, 1], dtype=torch.int32)
add_test("PERMUTE", [a_perm, perm], a_perm.permute(0, 2, 1))

a_cat = torch.rand((2, 4), dtype=torch.float32)
b_cat = torch.rand((2, 4), dtype=torch.float32)
axis_cat = torch.tensor([1], dtype=torch.int32)
add_test("CONCAT", [a_cat, b_cat, axis_cat], torch.cat([a_cat, b_cat], dim=1))

a_cast = torch.randint(1, 10, (4, 8), dtype=torch.int32)
add_test("CAST", [a_cast], a_cast.to(torch.float32))

a_triu = torch.rand((4, 4), dtype=torch.float32)
k_triu = torch.tensor([1], dtype=torch.int32)
add_test("TRIU", [a_triu, k_triu], torch.triu(a_triu, diagonal=1))

a_slice = torch.rand((4, 8), dtype=torch.float32)
starts = torch.tensor([1, 2], dtype=torch.int32)
ends = torch.tensor([3, 6], dtype=torch.int32)
steps = torch.tensor([1, 2], dtype=torch.int32)
add_test("SLICE", [a_slice, starts, ends, steps], a_slice[1:3:1, 2:6:2])

data_gather = torch.rand((10, 8), dtype=torch.float32)
idx_gather = torch.tensor([2, 5, 0], dtype=torch.int32)
add_test("GATHER", [data_gather, idx_gather], data_gather[idx_gather.long()])

a_rep = torch.rand((2, 1, 4), dtype=torch.float32)
repeats = torch.tensor([3], dtype=torch.int32)
axis_rep = torch.tensor([1], dtype=torch.int32)
add_test("REPEAT", [a_rep, repeats, axis_rep], a_rep.repeat(1, 3, 1))

start_ar = torch.tensor([2], dtype=torch.int32)
stop_ar = torch.tensor([8], dtype=torch.int32)
step_ar = torch.tensor([2], dtype=torch.int32)
add_test(
    "ARANGE", [start_ar, stop_ar, step_ar], torch.arange(2, 8, 2, dtype=torch.int32)
)

val_fill = torch.tensor([3.14], dtype=torch.float32)
shape_fill = torch.tensor([2, 4], dtype=torch.int32)
add_test("FILL", [val_fill, shape_fill], torch.full((2, 4), 3.14, dtype=torch.float32))

if os.path.exists(TEST_DIR):
    shutil.rmtree(TEST_DIR)
os.makedirs(TEST_DIR, exist_ok=True)

for i, (op, inputs, output) in enumerate(tests):
    test_dir = f"{TEST_DIR}/{i}"
    os.makedirs(test_dir, exist_ok=True)

    info = {"optype": op, "inputs": [], "output": None}

    tensors = {}
    for j, inp in enumerate(inputs):
        inp_contig = inp.contiguous().clone()
        info["inputs"].append(
            {
                "shape": list(inp_contig.shape),
                "strides": list(inp_contig.stride()),
                "dtype": dtype_to_str(inp_contig.dtype),
            }
        )
        tensors[f"input.{j}"] = inp_contig

    out_contig = output.contiguous().clone()
    info["output"] = {
        "shape": list(out_contig.shape),
        "strides": list(out_contig.stride()),
        "dtype": dtype_to_str(out_contig.dtype),
    }
    tensors["output"] = out_contig

    with open(f"{test_dir}/info.json", "w") as f:
        json.dump(info, f, indent=2)

    save_file(tensors, f"{test_dir}/data.safetensors")

print(f"Generated {len(tests)} tests in {TEST_DIR}/ directory.")
