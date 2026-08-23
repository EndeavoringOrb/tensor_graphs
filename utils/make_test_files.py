import os
import shutil

import torch
import torch.nn.functional as F
from binary import DTYPE_MAP, OP_TYPES, BinaryWriter
from safetensors.torch import save_file

TEST_DIR = "tensor_graphs_cpp/tests/data"


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

a_log = torch.rand((4, 8), dtype=torch.float32) + 0.1
add_test("LOG", [a_log], torch.log(a_log))

# --- Logical & Comparison ---
a_lt = torch.rand((4, 8), dtype=torch.float32)
b_lt = torch.rand((4, 8), dtype=torch.float32)
add_test("LT", [a_lt, b_lt], a_lt < b_lt)

a_eq = torch.randint(0, 5, (4, 8), dtype=torch.int32)
b_eq = torch.randint(0, 5, (4, 8), dtype=torch.int32)
add_test("EQ", [a_eq, b_eq], a_eq == b_eq)

a_and = torch.randint(0, 2, (4, 8), dtype=torch.bool)
b_and = torch.randint(0, 2, (4, 8), dtype=torch.bool)
add_test("AND", [a_and, b_and], a_and & b_and)

a_or = torch.randint(0, 2, (4, 8), dtype=torch.bool)
b_or = torch.randint(0, 2, (4, 8), dtype=torch.bool)
add_test("OR", [a_or, b_or], a_or | b_or)

a_not = torch.randint(0, 2, (4, 8), dtype=torch.bool)
add_test("NOT", [a_not], ~a_not)


# --- DOT (Batched 3D) ---
a_dot = torch.rand((2, 4, 8), dtype=torch.float32)
b_dot = torch.rand((2, 8, 4), dtype=torch.float32)
add_test("DOT", [a_dot, b_dot], torch.matmul(a_dot, b_dot))

# --- Reduce ---
a_sum = torch.rand((4, 8, 4), dtype=torch.float32)
axis_sum = torch.tensor([-1], dtype=torch.int32)
add_test("SUM", [a_sum, axis_sum], torch.sum(a_sum, dim=-1, keepdim=True))
add_test("MAX", [a_sum, axis_sum], torch.max(a_sum, dim=-1, keepdim=True).values)

a_argmax = torch.rand((4, 8, 4), dtype=torch.float32)
axis_argmax = torch.tensor([-1], dtype=torch.int32)
k_argmax = torch.tensor([2], dtype=torch.int32)
add_test(
    "ARGMAX",
    [a_argmax, axis_argmax, k_argmax],
    torch.topk(a_argmax, k=2, dim=-1).indices.to(torch.int32),
)

# --- Manipulation ---
a_res = torch.rand((4, 8), dtype=torch.float32)
target_dims = torch.tensor([2, 16], dtype=torch.int32)
add_test("RESHAPE", [a_res, target_dims], a_res.reshape((2, 16)))

# --- Permute ---
a_perm = torch.rand((2, 4, 8), dtype=torch.float32)
perm = torch.tensor([0, 2, 1], dtype=torch.int32)
add_test("PERMUTE", [a_perm, perm], a_perm.permute(0, 2, 1))

# --- Concat ---
a_cat = torch.rand((2, 4), dtype=torch.float32)
b_cat = torch.rand((2, 4), dtype=torch.float32)
axis_cat = torch.tensor([1], dtype=torch.int32)
add_test("CONCAT", [axis_cat, a_cat, b_cat], torch.cat([a_cat, b_cat], dim=1))

# --- Cast ---
a_cast = torch.randint(1, 10, (4, 8), dtype=torch.int32)
add_test("CAST", [a_cast], a_cast.to(torch.float32))

# --- Triu ---
a_triu = torch.rand((4, 4), dtype=torch.float32)
k_triu = torch.tensor([1], dtype=torch.int32)
add_test("TRIU", [a_triu, k_triu], torch.triu(a_triu, diagonal=1))

# --- Slice ---
a_slice = torch.rand((4, 8), dtype=torch.float32)
starts = torch.tensor([1, 2], dtype=torch.int32)
ends = torch.tensor([3, 6], dtype=torch.int32)
steps = torch.tensor([1, 2], dtype=torch.int32)
add_test("SLICE", [a_slice, starts, ends, steps], a_slice[1:3:1, 2:6:2])

# --- Gather ---
data_gather = torch.rand((10, 8), dtype=torch.float32)
idx_gather = torch.tensor([2, 5, 0], dtype=torch.int32)
add_test("GATHER", [data_gather, idx_gather], data_gather[idx_gather.long()])

# --- Repeat ---
a_rep = torch.rand((2, 1, 4), dtype=torch.float32)
repeats = torch.tensor([3], dtype=torch.int32)
axis_rep = torch.tensor([1], dtype=torch.int32)
add_test("REPEAT", [a_rep, repeats, axis_rep], a_rep.repeat(1, 3, 1))

# --- Arange ---
start_ar = torch.tensor([2], dtype=torch.int32)
stop_ar = torch.tensor([8], dtype=torch.int32)
step_ar = torch.tensor([2], dtype=torch.int32)
add_test(
    "ARANGE", [start_ar, stop_ar, step_ar], torch.arange(2, 8, 2, dtype=torch.int32)
)

# --- Fill ---
val_fill = torch.tensor([3.14], dtype=torch.float32)
shape_fill = torch.tensor([2, 4], dtype=torch.int32)
add_test("FILL", [val_fill, shape_fill], torch.full((2, 4), 3.14, dtype=torch.float32))

# --- Memory / Structural ---
target_scat = torch.zeros((4, 8), dtype=torch.float32)
updates_scat = torch.rand((2, 4), dtype=torch.float32)
starts_scat = torch.tensor([1, 2], dtype=torch.int32)
ends_scat = torch.tensor([3, 6], dtype=torch.int32)
steps_scat = torch.tensor([1, 1], dtype=torch.int32)
out_scatter = target_scat.clone()
out_scatter[1:3:1, 2:6:1] = updates_scat
add_test(
    "SCATTER",
    [target_scat, updates_scat, starts_scat, ends_scat, steps_scat],
    out_scatter,
)

img = torch.rand((1, 3, 8, 8), dtype=torch.float32)
k_size = torch.tensor([3], dtype=torch.int32)
stride = torch.tensor([1], dtype=torch.int32)
padding = torch.tensor([1], dtype=torch.int32)
out_im2col = F.unfold(img, kernel_size=3, padding=1, stride=1)
add_test("IM2COL", [img, k_size, stride, padding], out_im2col)


if os.path.exists(TEST_DIR):
    shutil.rmtree(TEST_DIR)
os.makedirs(TEST_DIR, exist_ok=True)

for i, (op, inputs, output) in enumerate(tests):
    test_dir = f"{TEST_DIR}/{i}"
    os.makedirs(test_dir, exist_ok=True)

    record = {
        "kernelId": OP_TYPES.index(op),
        "buildContextId": 0,
        "hwTag": "",
        "inputShapes": [list(inp.shape) for inp in inputs],
        "outputShapes": [list(output.shape)],
        "inputStrides": [list(inp.stride()) for inp in inputs],
        "outputStrides": [list(output.stride())],
        "inputDTypes": [DTYPE_MAP[inp.dtype] for inp in inputs],
        "outputDTypes": [DTYPE_MAP[output.dtype]],
        "inputConstants": [b"" for _ in inputs],
        "backends": [0],  # CPU
        "inputBackends": [[0] for _ in inputs],
        "runTime": 0.0,
    }

    tensors = {}
    for j, inp in enumerate(inputs):
        tensors[f"input.{j}"] = inp.contiguous().clone()
    tensors["output"] = output.contiguous().clone()

    with open(f"{test_dir}/info.bin", "wb") as f:
        bw = BinaryWriter(f)
        bw.write_record(record)

    save_file(tensors, f"{test_dir}/data.safetensors")

print(f"Generated {len(tests)} tests in {TEST_DIR}/ directory.")
