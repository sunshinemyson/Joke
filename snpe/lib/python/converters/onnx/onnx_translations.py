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
import numpy
from .. import translation, op_adapter
from ..op_graph import AxisFormat
from snpe import modeltools
from util import *

OnnxTranslations = translation.TranslationBank()
ADD_OP = "ADD_OP"
INFER_SHAPE = "INFER_SHAPE"
REMOVE_NOOP = "REMOVE_NOOP"
SQUASH_BATCHNORM = "SQUASH_BATCHNORM"
SQUASH_SCALE = "SQUASH_SCALE"
AXES_TO_SNPE_ORDER = "AXES_TO_SNPE_ORDER"


def inject_implicit_permute(graph, input_name, target_format, permute_order, consumers=[]):
    permute_name = input_name +'.'+target_format.lower()
    implicit_permute = op_adapter.PermuteOp(permute_name, permute_order)
    graph.inject(implicit_permute, input_name, permute_name, consumers)
    # since the implicit permute won't be visited in this pass, go
    # ahead and set the correct order for its buffer here.
    permute_buf = graph.get_buffer(permute_name)
    permute_buf.axis_format = target_format

def enforce_input_format(node, graph, input_name, target_format, permute_order):
    input_buf = graph.get_buffer(input_name)
    if input_buf.axis_format == AxisFormat.NONTRIVIAL:
        inject_implicit_permute(graph, input_name, target_format, permute_order)
    elif input_buf.axis_format != target_format:
        raise ValueError(ERROR_INPUT_DATA_ORDER_UNEXPECTED.format(name,
                                                                  target_format,
                                                                  input_buf.axis_format))

def permute_shape(shape, order):
    return [ shape[i] for i in order ]

# well-known permute orders
NCS_TO_NSC = [1,2,0]
TBF_TO_BTF = [1,0,2]

def image_to_snpe_order(node, graph):
    """Axis transformation for layers which take in and emit only image-valued data"""

    # (1) if any of our inputs are NONTRIVIAL, put a permute
    # of NCS -> NSC in front of them. This will be shared
    # with everyone who consumes that buffer, so don't specify consumers
    for name in node.input_names:
        # fetch input buffers one by one to avoid degenerate case where
        # an op uses the same input more than once and needs to permute it.
        enforce_input_format(node, graph, name, AxisFormat.NSC, NCS_TO_NSC)

    # (2) Update all of our output buffers to be in NSC order.
    # For this version of the converter, drop the batch dimension.
    for buf in graph.get_output_buffers(node):
        buf.shape = permute_shape(buf.shape[1:], NCS_TO_NSC)
        buf.axis_format = AxisFormat.NSC

def feature_to_snpe_order(node, graph):
    # Not much to do here, just mark the outputs
    # and for this version drop batch
    for buf in graph.get_output_buffers(node):
        buf.shape = buf.shape[1:]
        buf.axis_format = AxisFormat.FEATURE

def time_series_to_snpe_order(node, graph):
    for name in node.input_names:
        enforce_input_format(node, graph, name, AxisFormat.BTF, TBF_TO_BTF)

    for buf in graph.get_output_buffers(node):
        buf.shape = permute_shape(buf.shape, TBF_TO_BTF)
        buf.axis_format = AxisFormat.BTF

def eltwise_to_snpe_order(node, graph):
    input_buffers = graph.get_input_buffers(node)
    input_orders = [buf.axis_format for buf in input_buffers]
    if AxisFormat.NSC in input_orders:
        image_to_snpe_order(node, graph)
    elif AxisFormat.FEATURE in input_orders:
        feature_to_snpe_order(node, graph)
    elif AxisFormat.BTF in input_orders:
        time_series_to_snpe_order(node, graph)
    else:
        # well hopefully someone knows
        for buf in graph.get_output_buffers(node):
            buf.axis_format = AxisFormat.NONTRIVIAL

def log_axes_to_snpe_order(node, graph):
    LOG_DEBUG(DEBUG_AXES_TO_SNPE_ORDER_ENTRY, node.op.name)
    for input_name in node.input_names:
        LOG_DEBUG(DEBUG_AXES_TO_SNPE_ORDER_INPUT_SIZE,
                  input_name,
                  str(graph.get_buffer(input_name).shape))

class OnnxTranslationBase(translation.Translation):
    def __init__(self):
        translation.Translation.__init__(self)
        self.index_method(ADD_OP, self.add_op)
        self.index_method(INFER_SHAPE, self.infer_output_shapes)
        self.index_method(AXES_TO_SNPE_ORDER, self.axes_to_snpe_order)

    def add_op(self, src_op, graph):
        op = self.extract_parameters(src_op, graph)
        input_names = self.extract_input_names(src_op)
        input_names = filter(lambda name: not graph.weights.has(name), input_names)
        output_names = self.extract_output_names(src_op)
        output_names = filter(lambda name: not graph.weights.has(name), output_names)

        graph.add(op, input_names, output_names)

    def extract_input_names(self, src_op):
        return map(str, src_op.input)

    def extract_output_names(self, src_op):
        return map(str, src_op.output)

    def infer_output_shapes(self, node, input_shapes):
        return [input_shapes[0]]

#------------------------------------------------------------------------------
#   StaticOp
#------------------------------------------------------------------------------
# 'Static' ops are transformations applied to weights, which do not produce
# an actual runtime output.
class OnnxStaticOp(op_adapter.Op):
    TRANSLATION_KEY = 'static'
    def __init__(self, name):
        op_adapter.Op.__init__(self, name, self.TRANSLATION_KEY)

class OnnxStaticTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.index_method(REMOVE_NOOP, self.remove_noop)

    def infer_output_shapes(self, op, input_shapes):
        return []

    def axes_to_snpe_order(self, node, graph):
        pass

    def remove_noop(self, node, graph):
        graph.prune(node)

OnnxTranslations.register(OnnxStaticTranslation(), OnnxStaticOp.TRANSLATION_KEY)

#------------------------------------------------------------------------------
#   Input
#------------------------------------------------------------------------------
# ONNX doesn't have an input layer, but we need to handle the later stages
# of processing for SNPE.
class OnnxInputTranslation(OnnxTranslationBase):
    def axes_to_snpe_order(self, node, graph):
        buf = graph.get_buffer(node.output_names[0])
        if node.op.image_type == 'opaque':
            buf.axis_format = AxisFormat.NONTRIVIAL
        elif buf.rank() == 4:
            buf.shape = permute_shape(buf.shape[1:], NCS_TO_NSC)
            buf.axis_format = AxisFormat.NSC
            node.op.shape = buf.shape
        elif buf.rank() == 1:
            buf.shape = buf.shape[1:]
            buf.axis_format = AxisFormat.FEATURE
            node.op.shape = buf.shape
        else:
            raise ValueError(ERROR_INPUT_UNEXPECTED_RANK.format(node.op.name, buf.rank()))

OnnxTranslations.register(OnnxInputTranslation(),
                          op_adapter.InputOp.TRANSLATION_KEY)

#------------------------------------------------------------------------------
#   Dropout, and other Noops
#------------------------------------------------------------------------------
class OnnxNoop(op_adapter.Op):
    TRANSLATION_KEY = 'noop'
    def __init__(self, name):
        op_adapter.Op.__init__(self, name, self.TRANSLATION_KEY)


class OnnxNoopTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.index_method(REMOVE_NOOP, self.remove_noop)

    def extract_parameters(self, src_op, graph):
        return OnnxNoop(src_op.name)

    def extract_output_names(self, src_op):
        return [str(src_op.output[0])]

    def remove_noop(self, node, graph):
        graph.squash(node, node.input_names[0])

    def axes_to_snpe_order(self, node, graph):
        output_buf = graph.get_output_buffers(node)[0]
        input_buf = graph.get_input_buffers(node)[0]
        output_buf.shape = input_buf.shape
        output_buf.axis_format = input_buf.axis_format

OnnxTranslations.register(OnnxNoopTranslation(),
                          onnx_type('Dropout'),
                          OnnxNoop.TRANSLATION_KEY)
