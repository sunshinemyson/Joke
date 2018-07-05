#!/usr/bin/env python
# //=============================================================================
# //  @@-COPYRIGHT-START-@@
# //
# //  Copyright 2015-2016 Qualcomm Technologies, Inc. All rights reserved.
# //  Confidential & Proprietary - Qualcomm Technologies, Inc. ("QTI")
# //
# //  The party receiving this software directly from QTI (the "Recipient")
# //  may use this software as reasonably necessary solely for the purposes
# //  set forth in the agreement between the Recipient and QTI (the
# //  "Agreement"). The software may be used in source code form solely by
# //  the Recipient's employees (if any) authorized by the Agreement. Unless
# //  expressly authorized in the Agreement, the Recipient may not sublicense,
# //  assign, transfer or otherwise provide the source code to any third
# //  party. Qualcomm Technologies, Inc. retains all ownership rights in and
# //  to the software
# //
# //  This notice supersedes any other QTI notices contained within the software
# //  except copyright notices indicating different years of publication for
# //  different portions of the software. This notice does not supersede the
# //  application of any third party copyright notice to that third party's
# //  code.
# //
# //  @@-COPYRIGHT-END-@@
# //=============================================================================


error_codes_to_messages = {
    # //=============================================================================
    # //                 TENSORFLOW CONVERTER ERROR CODES
    # //=============================================================================

    # start of the batchnorm errors
    'ERROR_TF_GENERAL_ABSTRACT_CLASS_MUST_BE_INHERITED': "Not implemented. This abstract class must be inherited.",
    'ERROR_TF_BATCHNORM_RESOLVE_VARIANCE': "Cannot resolve BatchNorm layer due to missing variance value.",
    'ERROR_TF_BATCHNORM_RESOLVE_EPSILON': "Cannot resolve BatchNorm layer due to missing epsilon value.",
    'ERROR_TF_BATCHNORM_RESOLVE_SCALE': "Cannot resolve BatchNorm layer due to missing scale value.",
    'ERROR_TF_BATCHNORM_RESOLVE_MEAN': "Cannot resolve BatchNorm layer due to missing mean value.",
    'ERROR_TF_BATCHNORM_RESOLVE_BETA': "Cannot resolve BatchNorm layer due to missing beta value.",
    'ERROR_TF_BATCHNORM_GLOBALNORMALIZATION_INPUT': "Cannot resolve BatchNorm layer due to BatchNormWithGlobalNormalization node not having at least 4 const inputs (mean, variance, beta, scale).",

    # start of the concat errors
    'ERROR_TF_CONCAT_INPUT': "Concatenation layer requires at least two inputs.",

    # start of the conv errors
    'ERROR_TF_CONV_RESOLVE_BIAS': "Cannot resolve convolution layer due to missing bias after operation: {}",
    'ERROR_TF_CONV_RESOLVE_WEIGHTS': "Cannot resolve convolution layer due to missing weights for operation: {}",
    'ERROR_TF_CONV_RESOLVE_DILATION': "Cannot resolve convolution layer due to missing dilation for operation: {}",

    # start of the Deconv errors
    'ERROR_TF_DECONV_CANT_FIND_WEIGHTS_NODE': "Cannot resolve deconvolution layer due to missing weights for operation:",
    'ERROR_TF_DECONV_CANT_FIND_BIAS_NODE': "Cannot resolve deconvolution layer due to missing bias for operation:",
    'ERROR_TF_DECONV_NO_SUPPORT_RECT_PADDING': "Deconvolution does not support rectangular padding!",
    # start of the fullyconnected errors
    'ERROR_TF_MATMUL_RESOLVE_WEIGHTS': "Cannot resolve fully connected layer due to missing weights for MatMul operation: {}",
    'ERROR_TF_MATMUL_RESOLVE_BIAS': "Cannot resolve fully connected layer due to missing biases for BiasAdd operation: {}",
    # start of the prelu errors
    'ERROR_TF_RESOLVE_PRELU_COEFF': "Cannot resolve PReLu layer due to missing coefficient values.",

    # start of the converter errors
    'ERROR_TF_ADD_N_NUM_OF_INPUTS': "Expected two or more inputs for AddN operation: {}, converter cannot resolve at least two inputs",
    'ERROR_TF_EXPECTED_SINGLE_OUTPUT_FROM_PREVIOUS_LAYER': "Expected single output operation from previous layer.",
    'ERROR_TF_EXPECTED_SINGLE_OUTPUT_FOR_OP': "Expected single output for operation: {}",
    'ERROR_TF_EXPECTED_SINGLE_INPUT_FOR_OP': "Expected single input for operation: {}",
    'ERROR_TF_EXPECTED_SINGLE_OUTPUT_TENSOR_FOR_OP': "Expected single output tensor for operation: {}",
    'ERROR_TF_OPERATION_CONSUMED_BY_TWO_BUILDERS': "Operation {} resolved to multiple layers {}.",
    'ERROR_TF_UNABLE_TO_DETERMINE_PARAMS_TENSOR_FOR_NODE': "Unable to determine parameter tensor for {} node. Candidates = [{}]",
    'ERROR_TF_UNABLE_TO_RESOLVE_GRAPH_INPUT_DIMS': "Unable to resolve output dimensions for input node {}.",
    'ERROR_TF_UNEXPECTED_INPUT_SHAPE': "Unexpected input shape! expected: {}, actual: {}",
    'ERROR_TF_NO_INPUT_TO_CREATE_LAYER': "INTERNAL: No builder found to create layer from descriptor: {}",
    'ERROR_TF_INPUT_NODE_NOT_WITHIN_GRAPH': "Input node {} not within graph.",
    'ERROR_TF_OUTPUT_NODE_NOT_WITHIN_GRAPH': "Output node {} not within graph.",
    'ERROR_TF_OPERATION_ALREADY_MAPPED_TO_LAYER': "Operation already mapped to a layer.",
    'ERROR_TF_OPERATION_NOT_MAPPED_TO_LAYER': "Some operations in the Tensorflow graph were not resolved to a layer. "
                                              "You can use --allow_unconsumed_nodes for partial graph resolution!",

    # start of the loader errors
    'ERROR_TF_INPUT_NODE_SHAPE_DIMS_MISMATCH': "One input shape must be specified for every input node.",
    'ERROR_TF_INPUT_TYPES_AND_NAMES_NOT_IN_PAIRS': "Input types and names must be specified in pairs.",
    'ERROR_TF_INVALID_INPUT_DIMS': "Invalid input dimensions: {}",
    'ERROR_TF_GRAPH_FILE_DOES_NOT_EXIST': "Graph file does not exist {}",
    'ERROR_TF_NODES_NOT_FOUND_IN_GRAPH': "No nodes found in graph.",
    'ERROR_TF_CANNOT_IMPORT_GRAPH_FROM_META': "Failed to import graph from meta: {}",
    'ERROR_TF_GRAPH_META_EMPTY': "Graph meta is empty.",
    'ERROR_TF_NODE_NOT_FOUND_IN_GRAPH': "Node not found in graph. Node name: {}",

    # start of the util errors
    'ERROR_TF_OPERATION_NOT_FOUND': "Operation with type {} not found within {}",
    'ERROR_TF_INPUT_OPERATION_NOT_FOUND': "Input operation not found for {}",
    'ERROR_TF_MULTIPLE_NODES_FOUND': "Expected single node with type {}",
    'ERROR_TF_UNABLE_TO_FIND_OUTPUT_OPERATION': "Unable to find output operations for op {} in scope {}",
    'ERROR_TF_INPUT_DOES_NOT_MATCH_COUNT': "Operation ({}) inputs do not match expected count: {} vs {}",
    'ERROR_TF_INPUT_DOES_NOT_MATCH_TYPES': "Operation ({}) inputs do not match expected types: {} vs {}",
    'ERROR_TF_LAYER_INPUT_COUNT_ERROR': "Layer {} expects {} input(s), actual {}",
    'ERROR_TF_LAYER_NO_INPUT_FOUND': "{} layer {} requires at least one input layer.",
    'ERROR_TF_FALLBACK_TO_ONDEMAND_EVALUATION': "Unable to resolve operation output shapes in single pass. "
                                                "Using on-demand evaluation!",
    'ERROR_TF_SSD_ANCHOR_INPUT_MISSING': 'Unable to resolve box encoding anchor input later.',
    'ERROR_TF_SSD_NMS_REQUIRES_2_INPUTS': 'Multi Class Non Max Suppression expects 2 input layers.',
    'ERROR_TF_SSD_NMS_REQUIRES_SINGLE_INPUT_TENSOR': 'Multi Class Non Max Suppression expects single input tensor.',
    'ERROR_TF_SSD_NMS_CAN_NOT_RESOLVE_SCORE_THRESHOLD': 'Unable to resolve score threshold for Multi Class Non Max Suppression layer.',
    'ERROR_TF_SSD_NMS_CAN_NOT_RESOLVE_IOU': 'Unable to resolve IOU threshold for Multi Class Non Max Suppression layer.',


    # //=============================================================================
    # //                 CAFFE CONVERTER ERROR CODES
    # //=============================================================================

    'ERROR_CAFFE_NUM_BOTTOM_NOT_EQ_TO_NUM_TOP': "Cannot resolve DropOut layer due to number of bottom (inputs) != number top (outputs).",
    'ERROR_CAFFE_CONV_PARAMS_MISSING_KERNEL_FIELDS': "Cannot resolve Convolution layer {}. Missing kernel filter dimensions.",
    'ERROR_CAFFE_UDL_SET_IS_NOT_DICT': "UDL improperly specified. Must be dictionary.",
    'ERROR_CAFFE_INPUT_TYPES_LAYER_NAMES_NOT_IN_PAIRS': "Input types {} and layer names {} must be specified in pairs",
    'ERROR_CAFFE_CAFFE_PARSING_ERROR': "Caffe could not parse {}: {}",

    'ERROR_CAFFE_LAYER_OF_TYPE_SCALE_NOT_PRECEEDED_BY_BATCHNORM': "Cannot resolve Scale layer {} as it is not preceded by a BatchNorm layer into which it can be folded.",
    'ERROR_CAFFE_LAYER_TYPE_NOT_SUPPORTED': "Cannot resolve {} layer of type {} which is not yet supported by this conversion script.",

    'ERROR_CAFFE_UDL_FACTORY_FUNCS_NOT_SUPPLIED': "UDL factory functions improperly specified. Must be a dictionary instead of {}",
    'ERROR_CAFFE_UDL_BLOB_SIZE_IS_ZERO': "Cannot resolve UDL layer {}. Blob size of 0 not supported.",
    'ERROR_CAFFE_CONCAT_BATCH_DIM_ERR': "Cannot resolve Concat layer {}. Concatenation along batch dimension not supported.",
    'ERROR_CAFFE_CONCAT_AXIS_NOT_ALIGNED': "Cannot resolve Concat layer {}. The axis order of all input buffers are not aligned.",
    'ERROR_CAFFE_CROP_LAYER_OUTPUT_DIM_ERR': "Cannot resolve Crop layer . Crop supports only 1 or 3 output dimensions.",
    'ERROR_CAFFE_DATA_LAYER_ERR_NO_INPUT_DIM': "Cannot resolve Data layer {}. No input dimension specified.",
    'ERROR_CAFFE_NO_INPUT_PARAM_SPECIFIED': "Cannot resolve Input layer {}. No input_param field is specified.",
    'ERROR_CAFFE_DROPOUT_LAYER_WITH_MUL_OUTPUTS_ERR': "Cannot resolve DropOut layer {}. Multiple outputs not supported.",
    'ERROR_CAFFE_UNRECOGNIZED_ELEMENTWISE_OP': "Cannot resolve ElementWise layer {}. Unsupported operation {}.",
    'ERROR_CAFFE_INDEX_BASED_UPSAMPLING_DOES_NOT_SUPPORT_RECT_POOL': "Cannot resolve Pooling layer {}. Indexed based upsampling does not support rectangular pool regions.",
    'ERROR_CAFFE_PERMUTE_LAYER_MISSING_ORDER_FIELD': "Cannot resolve Permute layer {}. Missing order field.",
    'ERROR_CAFFE_PRELU_NON_CHANNEL_SHARED_SUPPORT_ONLY': "Cannot resolve Prelu layer {}. Only non-channel-shared supported.",
    'ERROR_CAFFE_NO_SUPPORT_DENSE_UPSAMPLING': "Cannot resolve Upsampling layer {}. Dense upsampling not supported.",
    'ERROR_CAFFE_NO_SUPPORT_BATCH_WISE_SLICING': "Cannot resolve Slice layer {}. Batch-wise slicing is not supported.",
    'ERROR_CAFFE_CONV_PARAMS_MISSING_KERNEL_SIZE': "Cannot resolve Convolution layer {}. Missing kernel filter dimensions.", #No layer name attribute
    'ERROR_CAFFE_PREPROCESSING_SET_TWICE_ON_MULTIPLE_INPUTS': "Input preprocessing is not supported on multiple input networks.",
    'ERROR_CAFFE_CANNOT_SET_INPUT_TYPE_NV21_ENCODING': "NV21 encoding requires input type to be set to image.",
    'ERROR_CAFFE_CROP_SIZE_LARGER_THAN_INPUT_DIMS': "The crop size ({}) cannot be larger than the input dimensions {}",
    'ERROR_CAFFE_MEAN_DATA_NOT_LARGE_ENOUGH': "Mean data (shape={}) must be large enough to cover the image (shape={})",
    'ERROR_CAFFE_INVALID_MEAN_VAL_SPECIFICATION': "Invalid number of mean values specified.",
    'ERROR_CAFFE_MEAN_DATA_WRONG_DIMS': "Mean data for data layer {} has the wrong dimensions: expected {}, got {}",
    'ERROR_CAFFE_UNSUPPORTED_INPUT_DIMS': "Cannot support input layer other than 2 or 4 dimensions. Unsupported layer {}.",
    'ERROR_CAFFE_UNSUPPORTED_PYTHON_MODULE': "Cannot resolve Python layer {}. Unsupported module layer {}.",
    'ERROR_CAFFE_PROPOSAL_LAYER_MISSING_PARAM_STR_FIELD': "Cannot find param_str in PYTHON: ProposalLayer layer {}.",
    'ERROR_CAFFE_TILE_AXIS_NOT_SUPPORTED': "Cannot resolve Tile layer {}. No equivalent axis for Caffe axis {}.",
    'ERROR_CAFFE_TILE_BATCH_DIM_ERR': "Cannot resolve Tile layer {}. Concatenation along batch dimension not supported.",
    'ERROR_CAFFE_INVALID_SSD_PARAM': "SSD layer {} param {} contains invalid value {}.",
    'ERROR_CAFFE_MISSING_SSD_PARAM': "SSD layer {} missing param {}.",

    # //=============================================================================
    # //                 CAFFE2 CONVERTER ERROR CODES
    # //=============================================================================

    'ERROR_CAFFE2_NUM_BOTTOM_NOT_EQ_TO_NUM_TOP': "Cannot resolve DropOut layer due to number of bottom (inputs) != number top (outputs).",
    'ERROR_CAFFE2_PRETRAINED_DATA_NAME_ERR': "Unexpected pretrained data name {} or inputs {}",
    'ERROR_CAFFE2_DATA_SINGLE_INSTANCE_EXPECTED': "Expected a single instance of data in pretrained data op {}.",
    'ERROR_CAFFE2_DUPLICATE_DATA_DETECTED': "Duplicate data name {} detected.",
    'ERROR_CAFFE2_WEIGHT_NAME_NOT_IN_MAP': "Cannot find weight buffer {} in model.",
    'ERROR_CAFFE2_WEIGHT_SHAPE_NOT_IN_MAP': "Cannot find weight shape for {} in model.",
    'ERROR_CAFFE2_WEIGHT_VALUES_NOT_IN_MAP': "Cannot find weight values for {} in model",
    'ERROR_CAFFE2_SPATIAL_BATCH_NORM_PARAMS_ORDER_ERR': "Cannot resolve BatchNorm layer {}. Expected 4 parameters in this order: scale, bias, mean, and variance.",
    'ERROR_CAFFE2_CONV_LAYER_INPUT_ERR': "Cannot resolve Convolution layer. Expected at least one set of weights for op {}.",
    'ERROR_CAFFE2_DECONV_LAYER_INPUT_ERR': "Cannot resolve Deconvolution layer. Expected at least one set of weights for op {}.",
    'ERROR_CAFFE2_OP_INPUT_WEIGHT_BIASES_ERR': "Cannot resolve FullyConnected layer. Expected one set of weights and biases for op {}.",
    'ERROR_CAFFE2_NORMALIZATION_PARAMS_ORDER_ERR': "Cannot resolve InstanceNormalization layer. Expected two params in order: scale, bias for instance normalization op {}.",
    'ERROR_CAFFE2_PRELU_EXPECTED_SLOPE_PARAM_ERR': "Cannot resolve PReLu layer. Expected slope parameter for PReLu op {}.",
    'ERROR_CAFFE2_SNPE_OP_SUPPORT_ERR': "Cannot resolve op type {} which is not yet supported by this conversion script.",
    'ERROR_CAFFE2_PROCESSING_OP_ERR': "Error processing op: {}. Error: {}.",
    'ERROR_CAFFE2_PARSING_NETWORK_DEF': "Error parsing network definition {}. Error: {}.",
    'ERROR_CAFFE2_NOT_INPUT_OR_OUTPUT_FOR_REORDER': "{} is not an external input or output, remove from reorder_list",
    'ERROR_CAFFE2_INPUT_DIMS_NOT_VALID': "No input dimensions specified.",
    'ERROR_CAFFE2_INPUT_DIMS_FORMAT_NOT_VALID': "Invalid input dimensions format. Should follow -i 'input_name' 3,224,224 -i 'input_name2' 3,100,50 etc.",
    'ERROR_CAFFE2_INPUT_DIMS_CHANNEL_FORMAT_NOT_VALID': "Invalid input dimension format. Expect 3 channels separated by commas: 3,224,224.",
    'ERROR_CAFFE2_DATA_NOT_AN_EXTERNAL_DATA_INPUT': "{} is not an external data input.",
    'ERROR_CAFFE2_NO_OPS_PRESENT_IN_CAFFE2_CONVERTER': "No operators present in the Caffe2 network.",
    'ERROR_CAFFE2_ONLY_NCHW_ORDER_SUPPORTED': "Unsupported order {} specified. Only NCHW order supported.",
    'ERROR_CAFFE2_CANT_PROCESS_ARGS': "Invalid arguments. Couldn't process argument: {}.",
    'ERROR_CAFFE2_DUPLICATE_ARG_FOUND': "Duplicate argument found: {}.",
    'ERROR_CAFFE2_CONCAT_LAYER_AXIS_NOT_SUPPORTED': "Cannot resolve Concat layer {}. No equivalent axis for Caffe2 axis {}.",
    'ERROR_CAFFE2_UNSUPPORTED_PADDING_OP_FOR_CONV_LAYER': "Cannot resolve Convolution layer {}. Unsupported padding op {} specified.",
    'ERROR_CAFFE2_DROPOUT_LAYER_WITH_MUL_OUTPUTS_ERR': "Cannot resolve DropOut layer {}. Multiple outputs not supported.",
    'ERROR_CAFFE2_CAFFE_POOLING_ONLY_LEGACY_SUPPORTED': "Cannot resolve Pooling layer. Only legacy Caffe pooling is supported.", #NO NAME
    'ERROR_CAFFE2_LRN_ARG_MISSING': "Cannot resolve LRN layer. Missing one of size, alpha, beta, or bias.", #NO NAME
    'ERROR_CAFFE2_RESHAPE_OP_NO_INPUT_OR_ARG_SHAPE': "Cannot resolve Reshape layer {}. No input or argument 'shape' specified.",
    'ERROR_CAFFE2_INVALID_OP_TYPE': "Cannot resolve Reshape layer. Invalid type {}.", #NO NAME
    'ERROR_CAFFE2_SLICE_OP_INDICIES_MISSING': "Cannot resolve Slice layer. Missing start or end indices.", #NO NAME
    'ERROR_CAFFE2_SLICE_OP_INPUT_DIM_MISMATCH': "Cannot resolve Slice layer. Start or end dimensions do not match input dimensions.", #NO NAME
    'ERROR_CAFFE2_SLICE_ONLY_SUPPORTED_FOR_3_DIMS_DATA': "Cannot resolve Slice layer {}. Slice is only supported for 3 dimensional data.", #NO NAME
    'ERROR_CAFFE2_NO_SUPPORT_BATCH_WISE_SLICING': "Cannot resolve Slice layer. Batch-wise slicing is not supported.", #No NAME
    'ERROR_CAFFE2_AXIS_ORDER_SPLIT_ARGS_ONLY_SUPPORTED': "Cannot resolve Split layer {}. Only axis, order, and split arg supported.", #NO NAME
    'ERROR_CAFFE2_GET_INPUT_ID_INVALID_INPUT': "INTERNAL: get_input_id needs a valid input.",
    'ERROR_CAFFE2_GET_INPUT_ID_INVALID_OUTPUT': "INTERNAL: get_output_name needs a valid output.",
    'ERROR_CAFFE2_REFLECT_PAD_MODE_ONLY_SUPPORTED': "Only 'reflect' image padding mode supported. Mode {} not supported.",
    'ERROR_CAFFE2_ARG_KERNEL_LEN_NOT_EXPECTED': "Invalid length of kernels specified. Expected 2 parameters but {} specified.",
    'ERROR_CAFFE2_CROP_SIZE_LARGER_THAN_INPUT_DIMS': "The crop size ({}) cannot be larger than the input dimensions {}",
    'ERROR_CAFFE2_MEAN_DATA_WRONG_DIMS': "Mean data for data layer {} has the wrong dimensions: expected {}, got {}",
    'ERROR_CAFFE2_UNEXPECTED_ZERO_BLOB_SIZE': "Unexpected blob size is 0",
    'ERROR_CAFFE2_ADD_ONLY_SAME_INPUT_SHAPES_SUPPORTED_ERR': "Only same shaped/sized inputs are currently supported for Add op {}",
    'ERROR_CAFFE2_POOLING_LEGACY_AND_DEFAULT_PADDING_SUPPORTED': "Cannot resolve Pooling layer {}. Only legacy Caffe and default Caffe2 padding styles are supported.",
    'ERROR_CAFFE2_CONV_LEGACY_AND_DEFAULT_PADDING_SUPPORTED': "Cannot resolve Convolution layer {}. Only legacy Caffe and default Caffe2 padding styles are supported.",
    'ERROR_CAFFE2_TILE_AXIS_NOT_SUPPORTED': "Cannot resolve Tile layer {}. No equivalent axis for Caffe2 axis {}.",
    'ERROR_CAFFE2_TILE_INPUTS_NOT_SUPPORTED': "Tile layer only supports 'tiles' and 'axis' as arguments, not inputs.",
    'ERROR_CAFFE2_WRONG_NUMBER_OF_INPUTS':"Could not resolve {} layer {}. Expected {} inputs, got {}",
    'ERROR_CAFFE2_WRONG_NUMBER_OF_OUTPUTS':"Could not resolve {} layer {}. Expected {} outputs, got {}",
    'ERROR_CAFFE2_IMPLODE_BATCH_INPUT':"Cannot resolve ImplodeBatch op {}: Predecessor must have type RoIAlign, but instead has {}",
    "ERROR_CAFFE2_UNSUPPORTED_ARGS_ERR":"Unsupported arguments: {}"
}

warning_codes_to_messages = {
    # //=============================================================================
    # //                 TENSORFLOW CONVERTER WARNING CODES
    # //=============================================================================

    # start of the converter warning messages
    'WARNING_TF_SCOPE_OP_NOT_CONSUMED': "Operation ({}) not consumed by converter: {}.",

    # //=============================================================================
    # //                 CAFFE CONVERTER WARNING CODES
    # //=============================================================================
    'WARNING_CAFFE_OMIT_DATA_LAYER_INCL_TRAIN': "Omitting data layer with include phase TRAIN.",
    'WARNING_CAFFE_FEWER_COEFFS_THAN_INPUT_NUM': "Fewer coeffs given than number of inputs. Padding with 1.0.",
    'WARNING_CAFFE_MORE_COEFFS_THAN_INPUT_NUM': "More coeffs than number of inputs. Ignoring extras.",

    # //=============================================================================
    # //                 CAFFE2 CONVERTER WARNING CODES
    # //=============================================================================
    'WARNING_CAFFE2_IGNORE_LOCAL_PADDING': "Using reflection padding in conv layer. Ignoring local conv padding params.",
    'WARNING_CAFFE2_CROP_INPUT_BUFFER_MORE_THAN_ONE_EL': "crop_input_buffer seems to have more than one elements, expecting one, taking the fisrt one {}.",
    'WARNING_CAFFE2_SUB_MEAN_BUFFER_MORE_THAN_ONE_EL': "subtract_mean_input_buffer seems to have more than one elements, expecting one, taking the fisrt one {}.",

}

debug_codes_to_messages = {
    # //=============================================================================
    # //                 TENSORFLOW CONVERTER DEBUGGUING CODES
    # //=============================================================================

    # start of the util debugging messages
    'DEBUG_TF_SCOPE_PRINT': "Scope({})",
    'DEBUG_TF_OP_NAME_TYPE_PRINT': "\tOperation({}) [{}])",

    # //=============================================================================
    # //                 CAFFE CONVERTER DEBUGGING CODES
    # //=============================================================================

    'DEBUG_CAFFE_OUTPUT_BUFFER_DUMP': "BufferProxy dump _output_buffers {}",
    'DEBUG_CAFFE_OUTPUT_BUFFER_PRINT': "\tBuffer: {}",
    'DEBUG_CAFFE_INPUT_BUFFER_DUMP': "BufferProxy dump pending input proxy size {}",
    'DEBUG_CAFFE_KEY_VALUE_PRINT': "\t + {} : {}",
    'DEBUG_CAFFE_OUTPUT_DUMP': "BufferProxy dump output proxy size {}.",
    'DEBUG_CAFFE_BUFFER_ALIAS_GEN': "BufferProxy generating alias buf: {} alias is {}.",
    'DEBUG_CAFFE_BUFFER_DROPOUT_HANDLE': "BufferProxy handling dropout special case.",
    'DEBUG_CAFFE_BUFFER_ADD_IMPLICIT_SCALE_LAYER': "BufferProxy add_implicit_scale_layer.",
    'DEBUG_CAFFE_BUUFERPROXY_INSTALLATION': "BufferProxy installing buffer proxy {} for original buffer {}.",
    'DEBUG_CAFFE_BUFFERPROXY_ADDING_LAYER': "BufferProxy adding layer name {}.",
    'DEBUG_CAFFE_BUFFERPROXY_ADD_LAYER_BUFFER': "BufferProxy add_layer {} buffer {}.",
    'DEBUG_CAFFE_BUFFERPROXY_ADD_LAYER_BUFFER_ALIAS_TO': "BufferProxy add_layer {} buffer {} will alias to {}.",
    'DEBUG_CAFFE_BUFFERPROXY_ADD_LAYER_BUFFER_DESCR': "BufferProxy add_layer buffer {} has dims {} of type {}.",
    'DEBUG_CAFFE_BUFFERPROXY_ADD_LAYER_BUFFER_PROXY_TO': "BufferProxy add_layer {} buffer {} is proxy to {}.",
    'DEBUG_CAFFE_NETWORK_TOPOLOGY_ADD_LAYER': "NetworkTopology adding layer name {} layer type {}.",
    'DEBUG_CAFFE_NETWORK_TOPOLOGY_DONE_ADDING': "NetworkTopology done adding.",
    'DEBUG_CAFFE_NETWORK_TOPOLOGY_GET_INPUT_BUFFER_NAME': "NetworkTopology get_input_buffer_name {}.",
    'DEBUG_CAFFE_NETWORK_TOPOLOGY_GET_INPUT_BUFFER_NAME_RET': "NetworkTopology get_input_buffer_name ret {}.",
    'DEBUG_CAFFE_NETWORK_TOPOLOGY_GET_OUTPUT_BUFFER_NAME': "NetworkTopology get_output_buffer_name {}.",
    'DEBUG_CAFFE_NETWORK_TOPOLOGY_GET_BUFFER_NAME': "NetworkTopology get_buffer_name {}.",
    'DEBUG_CAFFE_NETWORK_TOPOLOGY_GET_BUFFER_NAME_RET': "NetworkTopology get_buffer_name returned {}.",
    'DEBUG_CAFFE_PRINT_UDL_FACTORY_FUNCS': "UDL factory funcs: {}.",
    'DEBUG_CAFFE_PRINT_LAYER_TYPE': "layer type {}.",
    'DEBUG_CAFFE_PREWORK_OF_LAYER': "Doing per-work of layer {}.",
    'DEBUG_CAFFE_NO_PERMUTE_ORDER': "no_permute_order = {} permute_order = {}",
    'DEBUG_CAFFE_IMPLICIT_PERMUTE_LAYER': "Implicit permute layer with permute order {} is required for layer {}.",
    'DEBUG_CAFFE_POSTWORK_OF_LAYER': "Doing post-work of layer {}.",
    'DEBUG_CAFFE_ADDING_LAYER': "Adding layer {}.",
    'DEBUG_CAFFE_ADDING_LAYER_TOP_NAME_IS_TREATED_THE_SAME': "Adding layer with layer name and top name treated the same. The layer name {} is registered with buffer {}.",
    'DEBUG_CAFFE_ADDING_LAYER_TOP_NAME_IS_TREATED_DIFFERENTLY': "Adding layer with layer name and top name treated differently. The top name {} is registered.",
    'DEBUG_CAFFE_CONVERT_UDL': "Converting User defined layer {} layer {}.",
    'DEBUG_CAFFE_UDL_INPUT_DIMS': "UDL input dims: {}",
    'DEBUG_CAFFE_OUTPUT_DIMS_IDX': "output dim idx: {}",
    'DEBUG_CAFFE_UDL_OUTPUT_DIMS': "UDL output dims: {}",
    'DEBUG_CAFFE_CONVERTING_LAYER': "Converting {} layer {}.",
    'DEBUG_CAFFE_CONVERTING_BATCH_NORMALIZATION_LAYER': "Converting batch normalization (bn) layer {}.",
    'DEBUG_CAFFE_MERGING_SCALE_LAYER': "Merging scale layer {}.",

    'DEBUG_CAFFE_WEIGHT_DIMS': "Weight dims ({}, {}, {}, {})",
    'DEBUG_CAFFE_INPUT_DIMS': "input dim ({}, {}, {})",
    'DEBUG_CAFFE_OUTPUT_DIMS': "output dim {}",
    'DEBUG_CAFFE_CONVERTING_INPUT_LAYER': "Converting input layer {}.",
    'DEBUG_CAFFE_OMITTING_DROPOUT_LAYER': "Omitting dropout layer {}.",
    'DEBUG_CAFFE_DROPOUT_LAYER_WITHOUT_INPUT_BUFFER': "Dropout layer without in-place buffer.",
    'DEBUG_CAFFE_WEIGHTS_SHAPE': "weights shape {}.",
    'DEBUG_CAFFE_SNPE_PERMUTE_ORDER': "Caffe permute order {} SNPE permute order {}.",
    'DEBUG_CAFFE_NO_PERMUTE_REQUIRED_FOR_SNPE': "No permute is required for SNPE.",
    'DEBUG_CAFFE_ADDING_IMPLICIT_LAYER': "Adding implicit permute layer {} for the layer name {}.",
    'DEBUG_CAFFE_PRINT_PERMUTE_ORDER': "Permute order: {}",
    'DEBUG_CAFFE_SLICE_DIM': "slice_dim",
    'DEBUG_CAFFE_AXIS': "axis",
    'DEBUG_CAFFE_DEFINE_SLICE_DIM_AXIS_FIELD': "Neither slice_dim or axis field specified in slice param.",
    'DEBUG_CAFFE_AXIS_DEFAULT_FOR_LAYER': "Defaulting to axis=1 for layer type {} layer {}.",
    'DEBUG_CAFFE_GET_INPUT_ID_BUFFER': "get_input_id {} buffer name {}.",
    'DEBUG_CAFFE_GET_INPUT_NAME_BUFFER': "get_input_name layer name {} buffer name {} got input name {}.",
    'DEBUG_CAFFE_GET_INPUT_ID_LIST_BUFFER': "get_input_id_list list {} buffer name {}.",
    'DEBUG_CAFFE_GET_INPUT_NAMES_BUFFER': "get_input_names layer name {} buffer name {}.",
    'DEBUG_CAFFE_GET_OUTPUT_NAMES_BUFFER': "get_output_names layer name {} buffer name {}.",
    'DEBUG_CAFFE_GET_OUTPUT_NAME_BUFFER': "get_output_name layer name {} output buffer name {}.",
    'DEBUG_CAFFE_GET_INPUT_DIMS': "get_input_dims of input name {}.",
    'DEBUG_CAFFE_GET_OUTPUT_DIMS': "get_output_dims of output name {}.",
    'DEBUG_CAFFE_SETTING_UP_PREPROCESSING': "Setting up preprocessing",
    'DEBUG_CAFFE_ADDING_DATA_LAYER_W_DIMS': "Adding data layer {} with input dim {}.",
    'DEBUG_CAFFE_ADDING_IMPLICIT_SCALE_LAYER': "Adding implicit scale layer: {}, last layer name: {} network dim {}.",
    'DEBUG_CAFFE_ADDED_IMPLICIT_SCALE_LAYER_W_DIMS': "Added implicit scale layer with dimensions {}.",
    'DEBUG_CAFFE_SANITY_CHECK_DIM_OF_LAYER': "sanity check dim of {} is {}.",
    'DEBUG_CAFFE_PREPROCESSING_MEAN_DATA_FILE': "Processing mean_data_file {}.",

    # //=============================================================================
    # //                 CAFFE2 CONVERTER DEBUGGING CODES
    # //=============================================================================
    'DEBUG_CAFFE2_OUTPUT_BUFFER_DUMP': "BufferProxy dump _output_buffers {}.",
    'DEBUG_CAFFE2_OUTPUT_BUFFER_PRINT': "\tBuffer: {}",
    'DEBUG_CAFFE2_INPUT_BUFFER_DUMP': "BufferProxy dump pending input proxy size {}.",
    'DEBUG_CAFFE2_KEY_VALUE_PRINT': "\t{} : {}",
    'DEBUG_CAFFE2_OUTPUT_DUMP': "BufferProxy dump output proxy size {}.",
    'DEBUG_CAFFE2_BUFFER_ALIAS_GEN': "NetworkTopology generating alias buf: {} alias is {}",
    'DEBUG_CAFFE2_LAYER_ALIAS_GEN': "NetworkTopology generating alias layer name: {} alias is {}",
    'DEBUG_CAFFE2_LAYER_NAME_GEN': "NetworkTopology generating layer name: {}",
    'DEBUG_CAFFE2_BUFFER_DROPOUT_HANDLE': "BufferProxy handling dropout special case.",
    'DEBUG_CAFFE2_INSTALL_BUFFER_PROXY': "BufferProxy install_buffer_proxy.",
    'DEBUG_CAFFE2_BUFFERPROXY_ADDING_LAYER': "BufferProxy adding layer name {}.",
    'DEBUG_CAFFE2_BUFFERPROXY_ADD_LAYER_BUFFER': "BufferProxy add_layer {} buffer {}.",
    'DEBUG_CAFFE2_BUFFERPROXY_ADD_LAYER_BUFFER_ALIAS_TO': "BufferProxy add_layer {} buffer {} will alias to {}.",
    'DEBUG_CAFFE2_BUFFERPROXY_ADD_LAYER_BUFFER_DESCR': "BufferProxy add_layer buffer {} has dims {} of type {}.",
    'DEBUG_CAFFE2_BUFFERPROXY_ADD_LAYER_BUFFER_PROXY_TO': "BufferProxy add_layer {} buffer {} is proxy to {}.",
    'DEBUG_CAFFE2_NETWORK_TOPOLOGY_ADD_LAYER': "NetworkTopology adding layer name {} layer type {}.",
    'DEBUG_CAFFE2_NETWORK_TOPOLOGY_DONE_ADDING': "NetworkTopology done adding",
    'DEBUG_CAFFE2_NETWORK_TOPOLOGY_GET_INPUT_BUFFER_NAME': "NetworkTopology get_input_buffer_name {}.",
    'DEBUG_CAFFE2_NETWORK_TOPOLOGY_GET_INPUT_BUFFER_NAME_RET': "NetworkTopology get_input_buffer_name ret {}.",
    'DEBUG_CAFFE2_NETWORK_TOPOLOGY_GET_OUTPUT_BUFFER_NAME': "NetworkTopology get_output_buffer_name {}.",
    'DEBUG_CAFFE2_NETWORK_TOPOLOGY_GET_BUFFER_NAME': "NetworkTopology get_buffer_name {} as {}.",
    'DEBUG_CAFFE2_PREDATA_PARSE': "Error parsing initialization network data {}: {}",
    'DEBUG_CAFFE2_PREDATA_WEIGHTMAP': "Initializing weight map entry for {}.",
    'DEBUG_CAFFE2_PREDATA_WEIGHT_EXTERNAL': "{} not an external input, skipping.",
    'DEBUG_CAFFE2_PREDATA_WEIGHT_DATA': "{} first input to layer, must be data, skipping.",
    'DEBUG_CAFFE2_PREDATA_DECONV_WEIGHT_SHAPE': "deconv weight shape: {}",
    'DEBUG_CAFFE2_PREDATA_FC_WEIGHT_SHAPE': "fc weight shape: {}",
    'DEBUG_CAFFE2_PREDATA_STRIPPED_FC_SHAPE': "stripped fc weight shape dimension, now: {}",
    'DEBUG_CAFFE2_PREDATA_PRELU_WEIGHT': "Getting PRelu weights: {}",
    'DEBUG_CAFFE2_PREDATA_RESHAPE_DIM': "Getting reshape dimensions: 'shape'",

    'DEBUG_CAFFE2_CONVERT_FOUND_DATA': "Found data input '{}' w/dims: {}",
    'DEBUG_CAFFE2_CONVERT_PROCESSNET_NO_PREPROC': "No pre-processing enabled, adding data layer(s)",
    'DEBUG_CAFFE2_CONVERT_PROCESSNET_ADD_LAYER': "Adding data layer {} with input dims: {}",
    'DEBUG_CAFFE2_CONVERT_PROCESSING_ARG': "Processing network argument: {}",
    'DEBUG_CAFFE2_CONVERT_PROCESSING_OP': "Processing op: {}",
    'DEBUG_CAFFE2_CONVERT_PROCESS_INP': "Op: {} inputs: {}",
    'DEBUG_CAFFE2_CONVERT_PROCESS_INP_DIMS': "Op: {}, op type: {}, input dims: {}",
    'DEBUG_CAFFE2_CONVERT_PROCESS_OUT': "Op: {} outputs: {}",
    'DEBUG_CAFFE2_CONVERT_PROCESS_OUT_DIMS': "Op: {}, op type: {}, output dims: {}",
    'DEBUG_CAFFE2_CONVERT_PROCESS_NUM_ARGS': "# args: {}",
    'DEBUG_CAFFE2_CONVERT_GET_ARGS': "Getting arg: {}",
    'DEBUG_CAFFE2_CONVERT_ADDING_OP': "Adding op: {}",
    'DEBUG_CAFFE2_CONVERT_ADD_OP_SAME_BUFF': "Adding op with op name and top name treated the same. The op name {} is registered with buffer {}.",
    'DEBUG_CAFFE2_CONVERT_ADD_OP_DIFF_BUFF': "Adding op with op name and output name treated differently. The output name {} is registered.",
    'DEBUG_CAFFE2_CONVERT_PRE_WORK_OP': "Doing pre-work of op {}.",
    'DEBUG_CAFFE2_CONVERT_NO_PERM_ORDER': "no_permute_order = {}, permute_order = {}",
    'DEBUG_CAFFE2_CONVERT_IMPLIC_PERM_LAYER': "Implicit permute layer with permute order {} is required for layer {}.",
    'DEBUG_CAFFE2_CONVERT_POST_WORK_LAYER': "Doing post-word of layer {}.",
    'DEBUG_CAFFE2_CONVERT_OP_TO_BATCH_LAYER': "Converting {} op to batchnorm layer {}.",
    'DEBUG_CAFFE2_CONVERT_CONCAT_DIM': "Concat layer {} input dim is {}.",
    'DEBUG_CAFFE2_CONVERT_CONV_OP': "Converting {} operation {}.",
    'DEBUG_CAFFE2_CONVERT_CONV_WEIGHT_DIMS': "Weight dims ({}, {}, {}, {})",
    'DEBUG_CAFFE2_CONVERT_CONV_BIAS_DIMS': "Bias dim ({})",
    'DEBUG_CAFFE2_CONVERT_CONV_INP_DIMS': "Input dims ({}, {}, {})",
    'DEBUG_CAFFE2_CONVERT_DECONV_WEIGHT_DIM': "Weight dims ({}, {}, {}, {})",
    'DEBUG_CAFFE2_SKIP_DROPOUT_LAYER': "Omitting dropout layer {}.",
    'DEBUG_CAFFE2_CONVERT_FC_INP': "FC op inputs: {}",
    'DEBUG_CAFFE2_CONVERT_FC_WEIGHTS': "Weights shape {}",
    'DEBUG_CAFFE2_CONVERT_POOL_OUTPUT': "Output dim {}",
    'DEBUG_CAFFE2_CONVERT_ADD_IMPL_LAYER': "Adding implicit permute layer: {} for the layer name: {}",
    'DEBUG_CAFFE2_CONVERT_PERM_ORDER': "Permute order: {}",
    'DEBUG_CAFFE2_CONVERT_RESIZE_BIAS': "Resizing 1 bias to len of: {}",
    'DEBUG_CAFFE2_CONVERT_RESHAPE_DIMS': "Getting reshape dims from arguments.",
    'DEBUG_CAFFE2_CONVERT_SLICEOP_INP': "SliceOp input dims: {}",
    'DEBUG_CAFFE2_CONVERT_SLICEOP_START': "SliceOp start dims: {}",
    'DEBUG_CAFFE2_CONVERT_SLICEOP_END': "SliceOp end dims: {}",
    'DEBUG_CAFFE2_CONVERT_GET_INPUT_ID': "get_input_id {} buffer name {}",
    'DEBUG_CAFFE2_CONVERT_GET_INP_NAME': "get_input_name op name {} input name {} and 'real' input name {}",
    'DEBUG_CAFFE2_CONVERT_GET_INP_NAMES': "get_input_names found inputs: {}",
    'DEBUG_CAFFE2_CONVERT_GET_INP_ID_LIST': "get_input_id_list list {} buffer name {}",
    'DEBUG_CAFFE2_CONVERT_GET_OUT_NAMES': "get_output_names operator name {} buffer name {}",
    'DEBUG_CAFFE2_CONVERT_GET_LAYER_NAME': "get_layer_name name {}",
    'DEBUG_CAFFE2_CONVERT_GET_OUT_NAME': "get_output_name operator name {} output buffer name {}",
    'DEBUG_CAFFE2_CONVERT_GET_INP_DIMS': "get_input_dims of input name {}",
    'DEBUG_CAFFE2_CONVERT_GET_OUT_DIMS': "get_output_dims of input name {}",
    'DEBUG_CAFFE2_CONVERT_PADDING_OP': "Handling image padding op.",
    'DEBUG_CAFFE2_CONVERT_GOT_PADX': "Got padx: {} and pady: {}",
    'DEBUG_CAFFE2_CONVERT_GOT_STRIDEX': "Got strideX: {} and strideY: {}",
    'DEBUG_CAFFE2_CONVERT_GOT_KX': "Got kx: {} and ky: {}",
    'DEBUG_CAFFE2_CONVERT_NO_PREPROCESS': "No preprocessing enabled, skipping ImageInputOp.",
    'DEBUG_CAFFE2_CONVERT_REMAP_INPUT': "Op: {}, remapping input {} to {}",
    'DEBUG_CAFFE2_CONVERT_REMAP_OUTPUT': "Op: {}, remapping output {} to {}",
    'DEBUG_CAFFE2_CONVERT_SETUP_PREPROCESS_OP': "Setting up preprocessing for op: {} w/data input: {}",
    'DEBUG_CAFFE2_CONVERT_SETUP_ADD_LAYER': "Adding data layer {} with input dim: {}",
    'DEBUG_CAFFE2_CONVERT_GET_OUTPUT_BUFFER_NAME': "get_output_buffer_name of {} : {}",
    'DEBUG_CAFFE2_CONVERT_OUTPUT_BUFFER': "output buffers of {} : {}",
    'DEBUG_CAFFE2_CONVERT_LAYER': "Converting {} operation {}",
    'DEBUG_CAFFE2_CONVERT_UDL': "Converting User defined layer {} layer {}",
    'DEBUG_CAFFE2_UDL_INPUT_DIMS': "UDL input dims: {}",
    'DEBUG_CAFFE2_UDL_OUTPUT_DIM_IDX': "output dim idx {}",
    'DEBUG_CAFFE2_UDL_OUTPUT_DIMS': "UDL output dims: {}"

}

progress_codes_to_messages = {

    # //=============================================================================
    # //                 TENSORFLOW CONVERTER INFO CODES
    # //=============================================================================

    # start of the converter info messages
    'INFO_ALL_BUILDING_NETWORK':
    """
==============================================================
Building Network
==============================================================""",
    'INFO_TF_BUILDING_INPUT_LAYER': "Building layer (INPUT) with node: {}, shape {}",
    'INFO_TF_CONVERTING_SCOPES': "Converting scope ({}): {}",
    'INFO_ALL_BUILDING_LAYER_W_NODES': "Building layer ({}) with nodes: {}",

    # //=============================================================================
    # //                 CAFFE CONVERTER INFO CODES
    # //=============================================================================
    'INFO_CAFFE_CAFFE_INSTALLATION_ERROR': "Caffe installation in use: {}",
    'INFO_CAFFE_LAYER_TYPE_DEF_ERROR': "For definition of layer types, look for 'enum LayerType {' in https://raw.githubusercontent.com/BVLC/caffe/master/src/caffe/proto/caffe.proto",
    'INFO_CAFFE_BLOB_NAME_IS_A_DEPENDEE_OF': "Blob {} is a dependee of {}.",
    'INFO_CAFFE_BLOB_NAME_IS_OPERATED_ON_BY_AND_USED_BY': "Blob {} is operated on by: {} and is used by: {}",
    'INFO_CAFFE_OMIT_SILENCE_LAYER': "Omitting Silence layer {}",

    'INFO_CAFFE2_SETUP_EXTERNAL_INPUT_REORDERING': "Setting up external input {} to convert from CHW to HWC",
    'INFO_CAFFE2_SETUP_EXTERNAL_OUTPUT_REORDERING': "Setting up external output {} to convert from HWC to CHW"
}


def _wrapper_(error_code, message_table):
    try:
        message = message_table[error_code]
    except KeyError:
        message = ""

    def _formatter_(*args):
        if message.count('{}') == len(args):
            return "{}: {}".format(error_code, message.format(*[str(arg) for arg in args]))
        else:
            return "{}: N/A".format(error_code)
    if message.count('{}') > 0:
        return _formatter_
    else:
        return '{}: {}'.format(error_code, message)


def get_message(error_code):
    return _wrapper_(error_code, error_codes_to_messages)


def get_warning_message(error_code):
    return _wrapper_(error_code, warning_codes_to_messages)


def get_debugging_message(error_code):
    return _wrapper_(error_code, debug_codes_to_messages)


def get_progress_message(error_code):
    return _wrapper_(error_code, progress_codes_to_messages)
