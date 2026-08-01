from .bufferize import Buffer, overlaps
from .iter_dispatch import Handle, MemSpace


def test_overlaps():
    mem_space = MemSpace(0, Handle.CPP)

    a = Buffer(0, mem_space, 1, 0, 1)
    b = Buffer(1, mem_space, 1, 1, 2)
    assert not overlaps(a, b)

    a = Buffer(0, mem_space, 1, 0, 2)
    b = Buffer(1, mem_space, 1, 1, 3)
    assert overlaps(a, b)

    a = Buffer(0, mem_space, 1, 1, 3)
    b = Buffer(1, mem_space, 1, 0, 2)
    assert overlaps(a, b)


if __name__ == "__main__":
    test_overlaps()
