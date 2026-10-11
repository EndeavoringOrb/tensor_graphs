"""Shared rules consumed by the build linter and API."""
REDUNDANCY_PATTERNS = {
    r"inputs\.size\(\)": (
        "Input Count Check",
        "The engine already validates input count based on the macro arguments.",
    ),
    r"inputs\s*\[\d+\]\.backend": (
        "Input Backend Check",
        "Input backends are validated via the backend list in the registration macro.",
    ),
    r"output\.backend": (
        "Output Backend Check",
        "The output backend is validated by the registry before calling match().",
    ),
    r"isContiguous\s*\(\s*(inputs|inViews)\s*\[": (
        "Input Contiguity Check",
        "The Planner handles 'Contiguity Repair'. Use the boolean list in the macro instead.",
    ),
    r"inputs\s*\[\d+\]\.dtype != DType::": (
        "Input DType Check",
        "DTypes are already validated against the DType list in the registration macro.",
    ),
    r"inputs\s*\[0\]\.storageType\s*==\s*StorageType::PERSISTENT": (
        "Persistent Storage Check",
        "The engine's inplace safety logic now handles this automatically.",
    ),
}


STRUCTURAL_RULES = {
    "registration_arity": "Literal registration input metadata lists must match min_num_inputs.",
    "kernel_registration_detection_error": "Kernel implementation files must contain a registration macro.",
    "nested_reference_factory_header": "Keep ref.hpp at the fused folder root.",
    "reference_factory_header_missing": "Each fused kernel folder must contain ref.hpp.",
    "invalid_reference_factory_count": "Define exactly one reference factory in each fused ref.hpp.",
    "opencl_kernel_must_be_nested": "Place OpenCL kernels in a subfolder of the fused kernel folder.",
    "opencl_source_and_wrapper_must_be_together": "Keep each OpenCL source alongside its wrapper.",
    "reference_factory_in_kernel_file": "Define reference factories in ref.hpp, not implementation files.",
    "dangling_reference_hazard_detected": "Do not retain EClass/ENode references while mutating the egraph.",
}


def ruleLabel(rule_id):
    return rule_id + ": " + STRUCTURAL_RULES[rule_id]


def getRules():
    return ([{"id": name.lower().replace(" ", "_"), "description": reason, "pattern": pattern, "scope": "match"}
             for pattern, (name, reason) in REDUNDANCY_PATTERNS.items()] +
            [{"id": rule_id, "description": description, "scope": "layout" if rule_id != "dangling_reference_hazard_detected" else "rewrite"}
             for rule_id, description in STRUCTURAL_RULES.items()])
