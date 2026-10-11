"""Native interface and baseline eligibility shared by discovery and workers."""
# Bump when the runner requires a changed archived native API or eligibility policy.
NATIVE_INTERFACE_VERSION = 2


def baselineEligible(kernel):
    return (not kernel["is_view"] and not kernel["is_reference"]
            and kernel["has_reference_factory"])


def configurationErrors(configuration):
    errors = []
    dtype_sizes = {0: 4, 1: 4, 2: 8, 3: 2, 4: 1, 5: 0, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1}
    import math
    constants_by_input = configuration["input_constants"] or [[] for _ in configuration["input_shapes"]]
    for index, (shape, dtype, constants) in enumerate(zip(configuration["input_shapes"],
            configuration["input_dtypes"], constants_by_input, strict=True)):
        if constants:
            expected = math.prod(shape) * dtype_sizes[dtype]
            if len(constants) != expected:
                errors.append({"input_index": index, "expected_constant_bytes": expected,
                               "actual_constant_bytes": len(constants)})
    return errors
