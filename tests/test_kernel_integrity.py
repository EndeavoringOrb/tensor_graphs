import unittest
from tensor_graphs.backend.verifier import KernelVerifier


class TestKernelIntegrity(unittest.TestCase):
    def test_all_kernels_match_decomposition(self):
        """
        Automatically verifies that ALL registered composite kernels
        match their TensorNode decomposition logic.
        """
        results = KernelVerifier.verify_all_composite_kernels()

        print("\n=== Kernel Verification Report ===")
        print(f"Passed : {len(results['passed'])}")
        print(f"Skipped: {len(results['skipped'])}")
        print(f"Failed : {len(results['failed'])}")

        for p in results["passed"]:
            print(f"  ✓ {p}")

        for f in results["failed"]:
            print(f"  ✗ {f[0]}: {f[1]}")

        # Fail the test if there are any failures
        self.assertEqual(
            len(results["failed"]),
            0,
            f"Found {len(results['failed'])} kernel inconsistencies.",
        )


if __name__ == "__main__":
    unittest.main()
