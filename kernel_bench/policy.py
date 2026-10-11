"""Native interface and baseline eligibility shared by discovery and workers."""
# Bump when the runner requires a changed archived native API or eligibility policy.
NATIVE_INTERFACE_VERSION = 1


def baselineEligible(kernel):
    return (not kernel["is_view"] and not kernel["is_reference"]
            and kernel["has_reference_factory"])
