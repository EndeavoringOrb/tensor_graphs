from enum import Enum

from pydantic import GetCoreSchemaHandler
from pydantic.dataclasses import dataclass
from pydantic_core import core_schema


def prefixed_enum[E: Enum](cls: type[E]):
    prefix = f"{cls.__name__}."

    @classmethod
    def __get_pydantic_core_schema__(
        source, source_type, handler: GetCoreSchemaHandler
    ):
        def preprocess(v):
            return cls[v.removeprefix(prefix)] if isinstance(v, str) else v

        return core_schema.no_info_before_validator_function(
            preprocess,
            handler(source_type),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda v: f"{prefix}{v.name}"
            ),
        )

    cls.__get_pydantic_core_schema__ = __get_pydantic_core_schema__
    return cls


@prefixed_enum
class Op(Enum):
    INPUT = 0
    ADD = 1
    MUL = 2
    COPYTO = 3
    SQRT = 4


@prefixed_enum
class Handle(
    Enum
):  # for determining if a kernel can be used. the thing that distinguishes a handle is the read/write api.
    STORAGE = 0  # file descriptors
    CPP = 1  # pointers
    OPENCL = 2  # cl_mem??


@prefixed_enum
class EngineType(
    Enum
):  # for determining if a kernel can be used. the thing that distinguishes an engine type is if the kernel can be run on that hardware.
    CPU = 0
    QUALCOMM_IGPU = 1


@dataclass(frozen=True)
class Engine:  # for cost estimation
    idx: int  # global engine idx. i.e. cpu=0, gpu0=1, gpu1=2
    engine_type: EngineType


@dataclass(frozen=True)
class MemSpace:  # an allocated buffer of memory
    idx: int  # where is this physically.
    handle_type: Handle  # how is this read/written.


# system with disk, cpu, discrete gpu
cpu = Engine(idx=0, engine_type=EngineType.CPU)
gpu = Engine(idx=1, engine_type=EngineType.QUALCOMM_IGPU)
storage = MemSpace(idx=0, handle_type=Handle.STORAGE)
ram_cpu = MemSpace(idx=1, handle_type=Handle.CPP)
ram_cpu_opencl = MemSpace(idx=1, handle_type=Handle.OPENCL)
ram_gpu = MemSpace(idx=2, handle_type=Handle.OPENCL)

engine_capabilities = {
    cpu: {storage, ram_cpu, ram_cpu_opencl},
    gpu: {ram_gpu, ram_cpu_opencl},
}

op_costs = {
    Op.INPUT: 0,
    Op.ADD: 1,
    Op.MUL: 2,
    Op.COPYTO: 10,
    Op.SQRT: 10,
    (Op.SQRT, EngineType.QUALCOMM_IGPU): 1,
}


def does_engine_support_mem_space(engine: Engine, mem_space: MemSpace):
    return mem_space in engine_capabilities[engine]


@dataclass
class Node:
    name: str
    op: Op
    children: list[str]
    mem_space: MemSpace
    engine: Engine
    size: int  # size in bytes
    birth: int = -1  # start time. inclusive
    death: int = -1  # end time. exclusive
    cost: int = -1

    def __post_init__(self):
        self.cost = op_costs.get((self.op, self.engine.engine_type), op_costs[self.op])

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.name == other.name
        if isinstance(other, str):
            return self.name == other
        return NotImplemented

    def __hash__(self):
        return hash(self.name)


@dataclass
class Buffer:
    idx: int
    mem_space: MemSpace
    size: int
    start: int
    end: int
    offset: int = -1
