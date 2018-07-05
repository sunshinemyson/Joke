#==============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright 2018 Qualcomm Technologies, Inc. All rights reserved.
#  Confidential & Proprietary - Qualcomm Technologies, Inc. ("QTI")
#
#  The party receiving this software directly from QTI (the "Recipient")
#  may use this software as reasonably necessary solely for the purposes
#  set forth in the agreement between the Recipient and QTI (the
#  "Agreement"). The software may be used in source code form solely by
#  the Recipient's employees (if any) authorized by the Agreement. Unless
#  expressly authorized in the Agreement, the Recipient may not sublicense,
#  assign, transfer or otherwise provide the source code to any third
#  party. Qualcomm Technologies, Inc. retains all ownership rights in and
#  to the software
#
#  This notice supersedes any other QTI notices contained within the software
#  except copyright notices indicating different years of publication for
#  different portions of the software. This notice does not supersede the
#  application of any third party copyright notice to that third party's
#  code.
#
#  @@-COPYRIGHT-END-@@
#==============================================================================

#------------------------------------------------------------------------------
#   Errors
#------------------------------------------------------------------------------
ERROR_ASYMMETRIC_PADS_VALUES = "SNPE does not support asymmetric pads values"
ERROR_ACTIVATION_FUNCTION_UNSUPPORTED="SNPE does not support activation function {}"
ERROR_ADD_BIAS_PREV_NO_BIAS="Cannot squash bias-add op onto predecessor {} with type {}"
ERROR_ATTRIBUTE_MISSING="Node {} is missing required attribute {}"
ERROR_ATTRIBUTE_WRONG_TYPE="Node {}: requested to extract parameter {} with type {}, but stored as type {}"
ERROR_BATCHNORM_TEST_ONLY = "SNPE supports only test mode for BatchNormalization Ops"
ERROR_BROADCAST_NOT_SUPPORTED = "Cannot convert op {}: SNPE does not support broadcast operations"
ERROR_DECONV_OUTPUT_PADDING_UNSUPPORTED="SNPE does not support output padding for ConvTranspose ops"
ERROR_DECONV_RECTANGULAR_STRIDE_UNSUPPORTED="SNPE does not support rectangular strides for ConvTranspose ops"
ERROR_FC_WRONG_INPUT_SIZE = "FC Node {}: input size expected by weights ({}) does not match input size of buffer ({})"
ERROR_FC_AXIS_UNSUPPORTED='SNPE only supports an axis value of 1 for FC layers'
ERROR_FC_AXIS_W_UNSUPPORTED='SNPE only supports an axis_w value of 1 for FC layers'
ERROR_GEMM_TRANSPOSE_NOT_SUPPORTED="SNPE does not support input transpositions for GEMM."
ERROR_INPUT_DATA_ORDER_UNEXPECTED="op expected input {} in {} order, got: {}"
ERROR_INPUT_UNEXPECTED_RANK="non-opaque input {} has unexpected rank of {}"
ERROR_KERNEL_SHAPE_DIFFERS_FROM_WEIGHTS = "kernel_shape parameter differs from weights shape"
ERROR_MAX_ROI_POOL_BATCH_UNSUPPORTED="SNPE does not currently support a batch dimension greater than 1 for MaxRoiPool ops"
ERROR_MUL_SCALE_PREV_NOT_BATCHNORM="Cannot squash scale op onto predecessor {} with type {}"
ERROR_PADDING_TYPE_UNSUPPORTED="Node {}: SNPE does not support padding type {}"
ERROR_PERMUTE_TOO_MANY_DIMENSIOSN="SNPE does not support permute with >3 dimensions"
ERROR_PERMUTE_UNEXPECTED_INPUT_ORDER="Permute op got unexpected input data order {}"
ERROR_WEIGHTS_MISSING_KEY="Expected a static initializer for value {}"
ERROR_ONNX_NOT_FOUND="No onnx installation found on PYTHONPATH: {}"
ERROR_RESHAPE_BATCH_UNSUPPORTED="SNPE does not support a batch dimension greater than 1 for reshape ops"
ERROR_RESHAPE_UNEXPECTED_INPUT_ORDER="Reshape op got unexpected input data order {}"

#------------------------------------------------------------------------------
#   Warnings
#------------------------------------------------------------------------------
WARNING_BROADCAST_ADD="SNPE does not support broadcast Add operations, will attempt to interpret as bias-add operation"
WARNING_BROADCAST_MUL="SNPE does not support broadcast Mul operations, will attempt to interpret as scale operation"
WARNING_GEMM="SNPE does not support GEMM in the general case, attempting to interpret as FC"

#------------------------------------------------------------------------------
#   Info
#------------------------------------------------------------------------------
INFO_DLC_SAVE_LOCATION = "Saving model at {}"
INFO_STATIC_RESHAPE = "Applying static reshape to to {}: new name {} new shape {}"

#------------------------------------------------------------------------------
#   Debug
#------------------------------------------------------------------------------
DEBUG_AXES_TO_SNPE_ORDER_ENTRY="Node {}: axes_to_snpe_order"
DEBUG_AXES_TO_SNPE_ORDER_INPUT_SIZE = "Input buffer {}: shape {}"
DEBUG_INFERRED_SHAPE = "Node {}: inferred output shape {}"
DEBUG_CONVERTING_NODE="Attempting to convert node {} with type {}"
