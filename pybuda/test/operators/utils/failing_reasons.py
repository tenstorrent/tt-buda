# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Failing reasons for pytests marks


# Compilation failed
# Inference failed
# Validation failed
# Special case not supported
# ...


class FailingReasons:
    NOT_IMPLEMENTED = "Not implemented operator"

    BUGGY_SHAPE = "Buggy shape"

    MICROBATCHING_UNSUPPORTED = "Higher microbatch size is not supported"

    UNSUPPORTED_DIMENSION = "Unsupported dimension"

    UNSUPORTED_AXIS = "Unsupported axis parameter"

    UNSUPPORTED_PARAMETER_VALUE = "Unsupported parameter value"

    UNSUPPORTED_SPECIAL_CASE = "Unsupported special case"

    # Error message: E           RuntimeError: TT_ASSERT @ pybuda/csrc/passes/lowering_context.cpp:28: old_node->node_type() != graphlib::NodeType::kPyOp
    # Error for input shape (1, 1, 10000, 1). Error message: RuntimeError: TT_ASSERT @ pybuda/csrc/placer/lower_to_placer.cpp:245:
    COMPILATION_FAILED = "Model compilation failed"

    # Error message: E           AssertionError: Error during inference
    INFERENCE_FAILED = "Inference failed"

    # "Error message: E          AssertionError: Data mismatch detected"
    # Validation error caused by pcc threshold
    DATA_MISMATCH = "Verification failed due to data mismatch"

    UNSUPPORTED_TYPE_FOR_VALIDATION  = "Verification failed due to unsupported type in verify_module"

    # "Fatal python error - xfail does not work; UserWarning: resource_tracker: There appear to be 26 leaked semaphore objects to clean up at shutdown"
    # "Fatal python error - xfail does not work. Error message: Fatal Python error: Segmentation fault; UserWarning: resource_tracker: There appear to be 26 leaked semaphore objects to clean up at shutdown"
    SEMAPHORE_LEAK = "Semaphore leak"

    # RuntimeError: Fatal Python error: Segmentation fault
    SEG_FAULT = "Inference failed due to seg fault"

    UNSUPPORTED_INPUT_SOURCE = "Unsupported input source"

