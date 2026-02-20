from ..config import DEBUG_EXECUTION
import time


class Timer:
    def __init__(self, name="Elapsed"):
        if not DEBUG_EXECUTION:
            return
        self.name = name
        self.start = time.perf_counter()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not DEBUG_EXECUTION:
            return
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
        print(f"{self.name}: {self.elapsed:.2f} seconds")
