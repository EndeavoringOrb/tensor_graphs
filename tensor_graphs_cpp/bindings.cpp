#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "core/types.hpp"
#include "core/graph.hpp"
#include "core/memory.hpp"
#include "core/session.hpp"

namespace py = pybind11;

PYBIND11_MODULE(tg_cpp, m)
{
    m.doc() = "Tensor Graphs C++ Accelerated Backend";

    // Enums
    py::enum_<DType>(m, "DType")
        .value("FLOAT32", DType::FLOAT32)
        .value("INT32", DType::INT32)
        .value("BF16", DType::BF16)
        .value("BOOL", DType::BOOL)
        .export_values();

    py::enum_<Backend>(m, "Backend")
        .value("CPU", Backend::CPU)
        .export_values();

    py::enum_<StorageType>(m, "StorageType")
        .value("TRANSIENT", StorageType::TRANSIENT)
        .value("PERSISTENT", StorageType::PERSISTENT)
        .export_values();

    // Structs
    py::class_<TensorView>(m, "TensorView")
        .def(py::init<>())
        .def_readwrite("baseOffset", &TensorView::baseOffset)
        .def_readwrite("shape", &TensorView::shape)
        .def_readwrite("strides", &TensorView::strides)
        .def_readwrite("dtype", &TensorView::dtype);

    // Memory Management
    py::class_<MemoryManager>(m, "MemoryManager")
        .def(py::init<>())
        .def("add_buffer",[](MemoryManager &self, Backend b, uint64_t size)
             { self.buffers.emplace(b, DeviceBuffer(size)); })
        .def("allocate", &MemoryManager::allocate,
             py::arg("backend"), py::arg("nodeId"), py::arg("sizeBytes"), py::arg("storageType"),
             py::arg("refCount") = 0, py::arg("cost") = 0.0f)
        .def("init", &MemoryManager::init);

    // Graph Building
    py::class_<Graph>(m, "Graph")
        .def(py::init<>())
        .def("allocateId", &Graph::allocateId)
        .def("constant",[](Graph &g, std::vector<uint32_t> shape,
                            py::array_t<float> data, DType dtype)
             { return g.constant(shape, data.data(), dtype); })
        .def("weight",[](Graph &g, const std::string &path, const std::string &name)
             { return g.weight(path, name); })
        .def("input", &Graph::input)
        .def("inputWithId", &Graph::inputWithId, py::arg("id"), py::arg("shape"), py::arg("dtype"), py::arg("view"), py::arg("storageType") = StorageType::PERSISTENT)
        // Math operations
        .def("add", &Graph::add)
        .def("mul", &Graph::mul)
        .def("div", &Graph::div)
        .def("dot", &Graph::dot)
        .def("sin", &Graph::sin)
        .def("cos", &Graph::cos)
        .def("neg", &Graph::neg)
        .def("pow", &Graph::pow)
        // Reduction operations
        .def("sum", &Graph::sum)
        .def("max", &Graph::max)
        // Manipulation operations
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
        .def("im2col", &Graph::im2col);

    // Execution Session
    py::class_<Session>(m, "Session")
        .def(py::init<Graph &, MemoryManager &, uint32_t, std::string>())
        .def("run", [](Session &self, py::dict inputs)
             {
            std::unordered_map<uint32_t, const void*> c_inputs;
            
            for (auto item : inputs) {
                uint32_t nodeId = item.first.cast<uint32_t>();
                py::buffer b = item.second.cast<py::buffer>();
                py::buffer_info info = b.request();
                c_inputs[nodeId] = info.ptr;
            }
            
            self.run(c_inputs); })
        .def("get_output", [](Session &self, uint32_t nodeId) -> py::array
             {
            // This logic is a bit simplified; needs to match DType to numpy format
            const void* ptr = self.getOutput(nodeId);
            // Assuming float for now for demonstration
            return py::array_t<float>({100}, {sizeof(float)}, (float*)ptr); });
}
