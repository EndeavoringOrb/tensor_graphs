#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "core/types.hpp"
#include "core/graph.hpp"
#include "core/memory.hpp"
#include "core/session.hpp"

#include "generated/kernels_all.gen.hpp"
#include "generated/build_context.gen.hpp"

namespace py = pybind11;

// Helper to convert tg_cpp DType to NumPy format strings
std::string dtype_to_numpy_format(DType dtype)
{
    switch (dtype)
    {
    case DType::FLOAT32:
        return py::format_descriptor<float>::format();
    case DType::INT32:
        return py::format_descriptor<int32_t>::format();
    case DType::BF16:
        return "u2"; // Represent BF16 as uint16 raw bits for NumPy
    case DType::BOOL:
        return py::format_descriptor<bool>::format();
    default:
        return "b";
    }
}

PYBIND11_MODULE(tg_cpp, m)
{
    m.doc() = "Tensor Graphs C++ Accelerated Backend";

    // --- Enums ---
    py::enum_<DType>(m, "DType")
        .value("FLOAT32", DType::FLOAT32)
        .value("INT32", DType::INT32)
        .value("BF16", DType::BF16)
        .value("BOOL", DType::BOOL)
        .export_values();

    py::enum_<Backend>(m, "Backend")
        .value("CPU", Backend::CPU)
        .value("CUDA", Backend::CUDA)
        .export_values();

    py::enum_<StorageType>(m, "StorageType")
        .value("TRANSIENT", StorageType::TRANSIENT)
        .value("PERSISTENT", StorageType::PERSISTENT)
        .export_values();

    // --- Structs ---
    py::class_<TensorView>(m, "TensorView")
        .def(py::init<>())
        .def_readwrite("baseOffset", &TensorView::baseOffset)
        .def_readwrite("shape", &TensorView::shape)
        .def_readwrite("strides", &TensorView::strides)
        .def_readwrite("dtype", &TensorView::dtype)
        .def_static("calc_contiguous_strides", &TensorView::calcContiguousStrides);

    // --- Memory Management ---
    py::class_<MemoryManager>(m, "MemoryManager")
        .def(py::init<std::unordered_map<Backend, uint64_t>>())
        .def(py::init([]()
                      { 
            // Default constructor if no sizes provided
            return new MemoryManager({{Backend::CPU, 1024 * 1024 * 512}}); }))
        .def("add_buffer", [](MemoryManager &self, Backend b, uint64_t size)
             { self.buffers.emplace(b, DeviceBuffer(b, size)); })
        .def("allocate", &MemoryManager::allocate,
             py::arg("backend"), py::arg("nodeId"), py::arg("sizeBytes"), py::arg("storageType"),
             py::arg("refCount") = 0, py::arg("cost") = 0.0f)
        .def("init_all", &MemoryManager::init) // Aliased to match gemma example
        .def("init", &MemoryManager::init);

    // --- Graph Building ---
    py::class_<Graph>(m, "Graph")
        .def(py::init<>())
        .def_property_readonly("count", [](const Graph &g) { return (uint32_t)g.nodes.size(); })
        .def("allocate_node", [](Graph &g) { return g.allocateNode().id; })
        .def("constant", [](Graph &g, const std::vector<uint32_t> &shape, py::buffer b, DType dtype)
             {
            py::buffer_info info = b.request();
            return g.constant(shape, info.ptr, dtype); })
        .def("weight", &Graph::weight)
        .def("input", &Graph::input)
        .def("inputWithId", &Graph::inputWithId,
             py::arg("id"), py::arg("shape"), py::arg("dtype"), py::arg("view"),
             py::arg("storageType") = StorageType::PERSISTENT)

        // Math operations
        .def("add", &Graph::add)
        .def("mul", &Graph::mul)
        .def("div", &Graph::div)
        .def("dot", &Graph::dot)
        .def("sin", &Graph::sin)
        .def("cos", &Graph::cos)
        .def("neg", &Graph::neg)
        .def("pow", &Graph::pow)

        // Reduction
        .def("sum", &Graph::sum)
        .def("max", &Graph::max)

        // Manipulation
        .def("reshape", &Graph::reshape)
        .def("permute", &Graph::permute)
        .def("slice", &Graph::slice)
        .def("concat", &Graph::concat)
        .def("cast", &Graph::cast)
        .def("repeat", &Graph::repeat)
        .def("arange", &Graph::arange)
        .def("triu", &Graph::triu)
        .def("gather", &Graph::gather)
        .def("fill", &Graph::fill)
        .def("copyto", &Graph::copyto)
        .def("im2col", &Graph::im2col)
        .def("contiguous", &Graph::contiguous);

    // --- Execution Session ---
    py::class_<Session>(m, "Session")
        .def(py::init<Graph &, MemoryManager &, uint32_t, std::string>(),
             py::arg("graph"), py::arg("memory"), py::arg("rootId"), py::arg("cachePath") = "")
        .def("run", [](Session &self, py::dict inputs)
             {
            std::unordered_map<uint32_t, const void*> c_inputs;
            std::vector<py::buffer_info> infos; // Keep buffers alive during the call
            
            for (auto item : inputs) {
                uint32_t nodeId = item.first.cast<uint32_t>();
                py::buffer b = item.second.cast<py::buffer>();
                infos.push_back(b.request());
                c_inputs[nodeId] = infos.back().ptr;
            }
            self.run(c_inputs); })
        .def("get_output", [](Session &self, uint32_t nodeId, Graph &g) -> py::array
             {
            const void* ptr = self.getOutput(nodeId);
            if (!ptr) throw std::runtime_error("Output pointer is null. Did the session run?");

            // Retrieve metadata to build correct NumPy view
            const auto& node = g.nodes[nodeId];
            std::vector<ssize_t> shape;
            for(auto d : node.shape) shape.push_back(d);
            
            std::vector<ssize_t> strides;
            uint64_t elementSize = getDTypeSize(node.dtype);
            // tg_cpp uses element strides, NumPy uses byte strides
            auto elementStrides = TensorView::calcContiguousStrides(node.shape);
            for(auto s : elementStrides) strides.push_back(s * elementSize);

            return py::array(py::dtype(dtype_to_numpy_format(node.dtype)), shape, strides, ptr); }, py::arg("nodeId"), py::arg("graph"));
}