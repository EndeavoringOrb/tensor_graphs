import numpy as np
from ....backend.registry import KernelRegistry
from ....ir.dtypes import DType, Backend, TensorSignature
from ....ops.atomic_types import OpType
from ....ops.atomic.divide import divide_ref


# --- 1. Generic Vector ---
@KernelRegistry.register(
    OpType.DIVIDE,
    [
        TensorSignature(DType.FP32, (None,), Backend.CPU_NUMPY),
        TensorSignature(DType.FP32, (None,), Backend.CPU_NUMPY),
    ],
    reference_factory=divide_ref,
)
def div_generic_vector(inputs, outputs, attrs):
    result = inputs[0] / inputs[1]
    outputs[0][:] = result


@KernelRegistry.register(
    OpType.DIVIDE,
    [
        TensorSignature(DType.FP32, (None, None)),
        TensorSignature(DType.FP32, (None, None)),
    ],
    reference_factory=divide_ref,
)
def div_generic_matrix(inputs, outputs, attrs):
    result = inputs[0] / inputs[1]
    outputs[0][:] = result


@KernelRegistry.register(
    OpType.DIVIDE,
    [TensorSignature(DType.FP32, (1,)), TensorSignature(DType.FP32, (None, None))],
    reference_factory=divide_ref,
)
def div_scalar_broadcast(inputs, outputs, attrs):
    result = inputs[0] / inputs[1]
    outputs[0][:] = result


@KernelRegistry.register(
    OpType.DIVIDE,
    [TensorSignature(DType.FP32, (None, None)), TensorSignature(DType.FP32, (1,))],
    reference_factory=divide_ref,
)
def div_matrix_scalar(inputs, outputs, attrs):
    result = inputs[0] / inputs[1]
    outputs[0][:] = result


@KernelRegistry.register(
    OpType.DIVIDE,
    [TensorSignature(DType.FP32, shape=None), TensorSignature(DType.FP32, shape=None)],
    reference_factory=divide_ref,
)
def div_generic_tensor(inputs, outputs, attrs):
    result = inputs[0] / inputs[1]
    outputs[0][:] = result
