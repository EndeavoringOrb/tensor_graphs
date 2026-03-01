#pragma once
#include "core/types.hpp"

struct ViewOps
{
    static TensorView repeat(uint32_t nodeId, const TensorView &input,
                             uint32_t dim, uint32_t repeats)
    {
        if (dim >= input.shape.size())
        {
            std::stringstream ss;
            ss << "[ViewOps.repeat] Dimension " << dim << " out of bounds for shape ";
            ss << toString(input.shape) << " (node " << nodeId << ")";
            throw ViewOpValidationError(ss.str(), nodeId, OpType::REPEAT,
                                        input.shape, dim);
        }

        if (input.shape[dim] != 1)
        {
            std::stringstream ss;
            ss << "[ViewOps.repeat] Can only broadcast dimension of size 1. "
               << "Dimension " << dim << " has size " << input.shape[dim]
               << ", requested repeats=" << repeats << " (node " << nodeId << ")";
            throw ViewOpValidationError(ss.str(), nodeId, OpType::REPEAT,
                                        input.shape, dim);
        }

        TensorView output = input;
        output.shape[dim] = repeats;
        output.strides[dim] = 0; // Broadcasting stride
        return output;
    }

    static TensorView reshape(uint32_t nodeId, const TensorView &input,
                              const std::vector<uint32_t> &newShape)
    {
        if (!input.isContiguous())
        {
            std::stringstream ss;
            ss << "[ViewOps.reshape] Cannot reshape non-contiguous tensor. "
               << "Current strides: [";
            for (size_t i = 0; i < input.strides.size(); ++i)
            {
                if (i > 0)
                    ss << ", ";
                ss << input.strides[i];
            }
            ss << "] (node " << nodeId << ")";
            throw ViewOpValidationError(ss.str(), nodeId, OpType::RESHAPE,
                                        input.shape);
        }

        uint64_t inputElements = countElements(input.shape);
        uint64_t outputElements = countElements(newShape);

        if (inputElements != outputElements)
        {
            std::stringstream ss;
            ss << "[ViewOps.reshape] Shape mismatch: expected "
               << outputElements << " elements, got " << inputElements
               << ". Reshaping " << toString(input.shape)
               << " to " << toString(newShape) << " (node " << nodeId << ")";
            throw ShapeMismatchError(ss.str(), nodeId, input.shape, newShape);
        }

        TensorView output = input;
        output.shape = newShape;
        output.strides = TensorView::calcContiguousStrides(newShape);
        return output;
    }

    static TensorView permute(const TensorView &input, const std::vector<uint32_t> &dims)
    {
        if (dims.size() != input.shape.size())
        {
            throw ViewOpValidationError(
                "[ViewOps.permute] Dimension count mismatch",
                0, OpType::PERMUTE, input.shape);
        }

        for (uint32_t d : dims)
        {
            if (d >= input.shape.size())
            {
                std::stringstream ss;
                ss << "[ViewOps.permute] Dimension " << d << " out of bounds for rank "
                   << input.shape.size();
                throw ViewOpValidationError(ss.str(), 0, OpType::PERMUTE,
                                            input.shape);
            }
        }

        TensorView output;
        output.baseOffset = input.baseOffset;
        output.shape.resize(dims.size());
        output.strides.resize(dims.size());

        for (size_t i = 0; i < dims.size(); ++i)
        {
            output.shape[i] = input.shape[dims[i]];
            output.strides[i] = input.strides[dims[i]];
        }
        return output;
    }

    static TensorView slice(const TensorView &input, uint32_t dim,
                            uint32_t start, uint32_t stop, uint32_t step)
    {
        if (dim >= input.shape.size())
        {
            throw ViewOpValidationError(
                "[ViewOps.slice] Dimension out of bounds",
                0, OpType::SLICE, input.shape, dim);
        }

        if (start > stop || stop > input.shape[dim])
        {
            std::stringstream ss;
            ss << "[ViewOps.slice] Invalid slice range: [" << start << ", "
               << stop << ") for dimension " << dim << " with size "
               << input.shape[dim];
            throw ViewOpValidationError(ss.str(), 0, OpType::SLICE,
                                        input.shape, dim);
        }

        if (step == 0)
        {
            throw ViewOpValidationError(
                "[ViewOps.slice] Step cannot be zero",
                0, OpType::SLICE, input.shape, dim);
        }

        TensorView output = input;
        output.baseOffset += start * input.strides[dim];
        output.shape[dim] = (stop - start + step - 1) / step;
        output.strides[dim] *= step;

        return output;
    }
};