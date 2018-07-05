#!/usr/bin/env python
# //=============================================================================
# //  @@
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
# //  @@
# //=============================================================================
import re

import numpy as np
import snpe

from converters import code_to_message
from converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from converters.tensorflow.util import ConverterError
from converters.tensorflow.util import GraphHelper
from converters.tensorflow.util import OperationNotFoundError
from converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence,
    ConverterRepeatableSequenceTreeNode,
    NonConsumableConverterSequenceNode
)

from converters.tensorflow.layers.batchnorm import BatchNormLayerResolver


class ConvolutionLayerResolver(LayerResolver, object):
    TF_ATTRIBUTE_STRIDES = 'strides'
    TF_ATTRIBUTE_PADDING = 'padding'

    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, conv_op, bias_op, strides, padding, weights, biases, output_names=None):
            super(ConvolutionLayerResolver.Descriptor, self).__init__('Convolution', name, nodes,
                                                                      output_names=output_names)
            self.conv_op = conv_op
            self.bias_op = bias_op
            self.strides = strides
            self.padding = padding
            self.weights = weights
            self.biases = biases
            self.dilationX = 1
            self.dilationY = 1
            self.groups = len([op for op in nodes if op.type == 'Conv2D'])
            self.output_op = self.child_ops[-1]
            self.input_ops = [conv_op]

        def is_input_op(self, op):
            return op in self.input_ops

    def __init__(self):
        self.sequence = GraphSequence([ConverterSequenceNode('root', ['Conv2D'])])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        if len(matches) == 0:
            return []
        descriptors = []
        for match in matches:
            conv_op = match['root']
            bias_op = None
            biases = None
            strides = conv_op.get_attr(self.TF_ATTRIBUTE_STRIDES)
            padding = conv_op.get_attr(self.TF_ATTRIBUTE_PADDING)
            weights = self.get_weights(graph_helper, conv_op)
            consumed_nodes = list(match.consumed_nodes)
            output_op_nodes_names = [str(match[node.identifier].outputs[0].name) for node in self.sequence.output_nodes]
            try:
                conv_output_ops = graph_helper.get_op_outputs(conv_op)
                bias_op = GraphHelper.filter_single_op_by_type(conv_output_ops, 'BiasAdd')
                biases = self.get_biases(graph_helper, conv_op, bias_op)

            except OperationNotFoundError:
                pass

            if bias_op is None:
                try:
                    conv_output_ops = graph_helper.get_op_outputs(conv_op)
                    bias_op = GraphHelper.filter_single_op_by_type(conv_output_ops, 'Add')
                    biases = self.get_biases(graph_helper, conv_op, bias_op)
                except OperationNotFoundError:
                    pass

            if bias_op is not None and biases is not None:
                output_op_nodes_names = [str(bias_op.outputs[0].name)]
                consumed_nodes.append(bias_op)
            else:
                biases = np.zeros(weights.shape[-1], dtype=np.float32)

            descriptor = ConvolutionLayerResolver.Descriptor(str(conv_op.name), consumed_nodes, conv_op, bias_op,
                                                             strides, padding, weights, biases,
                                                             output_names=output_op_nodes_names)
            descriptors.append(descriptor)
        return descriptors

    def get_biases(self, graph_helper, conv_op, bias_op):
        _, biases_tensor = GraphHelper.get_op_input_tensors(bias_op, ('?', '?'))
        if biases_tensor.op.type not in ['Identity', 'Const']:
            raise ConverterError(code_to_message.get_message('ERROR_TF_CONV_RESOLVE_BIAS')(conv_op.name))
        biases = graph_helper.evaluate_tensor_output(biases_tensor)
        return biases

    def get_weights(self, graph_helper, conv_op):
        _, weights_tensor = GraphHelper.get_op_input_tensors(conv_op, ('?', '?'))
        if weights_tensor.op.type not in ['Identity', 'Const', 'Split']:
            raise ConverterError(code_to_message.get_message('ERROR_TF_CONV_RESOLVE_WEIGHTS')(conv_op.name))
        weights = graph_helper.evaluate_tensor_output(weights_tensor)
        return weights


class DilatedConvolutionLayerResolver(ConvolutionLayerResolver, object):
    class Descriptor(ConvolutionLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(DilatedConvolutionLayerResolver, self).__init__()
        self.graph_sequence = GraphSequence([
            ConverterSequenceNode('space_to_batch', ['SpaceToBatchND']),
            NonConsumableConverterSequenceNode('inputs', ['?']),
            ConverterSequenceNode('dilation_sizes', ['?']),
            ConverterSequenceNode('paddings', ['?']),
            ConverterSequenceNode('conv_op', ['Conv2D']),
            ConverterSequenceNode('kernel', ['?']),
            ConverterSequenceNode('batch_to_space', ['BatchToSpaceND']),
            ConverterSequenceNode('block_shape_out', ['?']),
            ConverterSequenceNode('crops', ['?'])]
        )
        self.graph_sequence.set_inputs('space_to_batch', ['inputs', 'dilation_sizes', 'paddings'])
        self.graph_sequence.set_inputs('conv_op', ['space_to_batch', 'kernel'])
        self.graph_sequence.set_inputs('batch_to_space', ['conv_op', 'block_shape_out', 'crops'])
        self.graph_sequence.set_outputs(['batch_to_space'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.graph_sequence)
        if len(matches) == 0:
            return []
        descriptors = []
        for match in matches:
            conv_op = match['conv_op']
            strides = conv_op.get_attr(self.TF_ATTRIBUTE_STRIDES)
            padding = conv_op.get_attr(self.TF_ATTRIBUTE_PADDING)
            weights = self.get_weights(graph_helper, conv_op)
            consumed_nodes = match.consumed_nodes
            output_op_nodes_names = [str(match[node.identifier].outputs[0].name) for node in
                                     self.graph_sequence.output_nodes]
            try:
                batch_to_space_op = match['batch_to_space']
                conv_output_ops = graph_helper.get_op_outputs(batch_to_space_op)
                bias_op = GraphHelper.filter_single_op_by_type(conv_output_ops, 'BiasAdd')
                biases = self.get_biases(graph_helper, conv_op, bias_op)
                consumed_nodes.append(bias_op)
                output_op_nodes_names = [str(bias_op.outputs[0].name)]
            except OperationNotFoundError:
                bias_op = None
                biases = np.zeros(weights.shape[-1], dtype=np.float32)
            dilation_sizes = match['dilation_sizes']
            dilation_sizes = graph_helper.evaluate_tensor_output(dilation_sizes.outputs[0])
            if np.shape(dilation_sizes) != (2,):
                raise ConverterError(code_to_message.get_message('ERROR_TF_CONV_RESOLVE_DILATION')(conv_op.name))

            d = ConvolutionLayerResolver.Descriptor(str(conv_op.name), consumed_nodes,
                                                    conv_op, bias_op, strides, padding, weights, biases,
                                                    output_names=output_op_nodes_names)
            d.dilationY = int(dilation_sizes[0])
            d.dilationX = int(dilation_sizes[1])
            d.input_ops = [match['space_to_batch']]
            descriptors.append(d)
        return descriptors


class DepthwiseConvolutionLayerResolver(ConvolutionLayerResolver, object):

    def __init__(self):
        super(DepthwiseConvolutionLayerResolver, self).__init__()
        self.graph_sequence_with_bias = GraphSequence([
            ConverterSequenceNode('conv', ['DepthwiseConv2dNative']),
            ConverterSequenceNode('bias', ['BiasAdd']),
            NonConsumableConverterSequenceNode('other', ['?'])
        ])
        self.graph_sequence_with_bias.set_inputs('bias', ['conv', 'other'])
        self.graph_sequence_with_bias.set_outputs(['bias'])

        self.graph_sequence = GraphSequence([ConverterSequenceNode('conv', ['DepthwiseConv2dNative'])])
        self.graph_sequence.set_outputs(['conv'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.graph_sequence)
        matches += graph_matcher.match_sequence(self.graph_sequence_with_bias)
        descriptors = []
        for match in matches:
            self._resolve_from_match(descriptors, graph_helper, match)
        return descriptors

    def _resolve_from_match(self, descriptors, graph_helper, match):
        conv_op = match['conv']
        strides = conv_op.get_attr(self.TF_ATTRIBUTE_STRIDES)
        padding = conv_op.get_attr(self.TF_ATTRIBUTE_PADDING)
        weights = self.get_weights(graph_helper, conv_op)
        weights = np.transpose(weights, [0, 1, 3, 2])

        if 'bias' in match:
            biases = self.get_biases(graph_helper, conv_op, match['bias'])
        else:
            biases = np.zeros(np.shape(weights)[-1], dtype=np.float32)
        consumed_nodes = match.consumed_nodes
        d = ConvolutionLayerResolver.Descriptor(str(conv_op.name), consumed_nodes,
                                                conv_op, None, strides, padding, weights, biases)
        input_tensor, _ = GraphHelper.get_op_input_tensors(conv_op, ('?', '?'))
        d.groups = graph_helper.get_op_output_shape(input_tensor)[-1]
        descriptors.append(d)


class DilatedDepthwiseConvolutionLayerResolver(ConvolutionLayerResolver, object):
    class Descriptor(ConvolutionLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(DilatedDepthwiseConvolutionLayerResolver, self).__init__()
        self.graph_sequence = GraphSequence([
            ConverterSequenceNode('space_to_batch', ['SpaceToBatchND']),
            NonConsumableConverterSequenceNode('inputs', ['?']),
            ConverterSequenceNode('dilation_sizes', ['?']),
            ConverterSequenceNode('paddings', ['?']),
            ConverterSequenceNode('conv_op', ['DepthwiseConv2dNative']),
            ConverterSequenceNode('kernel', ['?']),
            ConverterSequenceNode('batch_to_space', ['BatchToSpaceND']),  # output
            ConverterSequenceNode('block_shape_out', ['?']),
            ConverterSequenceNode('crops', ['?'])
        ])
        self.graph_sequence.set_inputs('space_to_batch', ['inputs', 'dilation_sizes', 'paddings'])
        self.graph_sequence.set_inputs('conv_op', ['space_to_batch', 'kernel'])
        self.graph_sequence.set_inputs('batch_to_space', ['conv_op', 'block_shape_out', 'crops'])
        self.graph_sequence.set_outputs(['batch_to_space'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.graph_sequence)
        if len(matches) == 0:
            return []
        descriptor = []
        for match in matches:
            conv_op = match['conv_op']
            strides = conv_op.get_attr(self.TF_ATTRIBUTE_STRIDES)
            padding = conv_op.get_attr(self.TF_ATTRIBUTE_PADDING)
            weights = self.get_weights(graph_helper, conv_op)
            weights = np.transpose(weights, [0, 1, 3, 2])
            consumed_nodes = match.consumed_nodes
            output_op_nodes_names = [str(match[node.identifier].outputs[0].name) for node in
                                     self.graph_sequence.output_nodes]
            try:
                batch_to_space_op = match['batch_to_space']
                conv_output_ops = graph_helper.get_op_outputs(batch_to_space_op)
                bias_op = GraphHelper.filter_single_op_by_type(conv_output_ops, 'BiasAdd')
                biases = self.get_biases(graph_helper, conv_op, bias_op)
                consumed_nodes.append(bias_op)
                output_op_nodes_names = [str(bias_op.outputs[0].name)]
            except OperationNotFoundError:
                bias_op = None
                biases = np.zeros(np.shape(weights)[-1], dtype=np.float32)
            dilation_sizes = match['dilation_sizes']
            dilation_sizes = graph_helper.evaluate_tensor_output(dilation_sizes.outputs[0])
            if np.shape(dilation_sizes) != (2,):
                raise ConverterError(code_to_message.get_message('ERROR_TF_CONV_RESOLVE_DILATION')(conv_op.name))

            d = ConvolutionLayerResolver.Descriptor(str(conv_op.name), consumed_nodes,
                                                    conv_op, bias_op, strides, padding, weights, biases,
                                                    output_names=output_op_nodes_names)

            space_to_batch_op = match['space_to_batch']
            d.groups = graph_helper.get_op_output_shape(space_to_batch_op)[-1]
            d.dilationY = int(dilation_sizes[0])
            d.dilationX = int(dilation_sizes[1])
            d.input_ops = [space_to_batch_op]
            descriptor.append(d)
        return descriptor


class ConvolutionLayerBuilder(LayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ConvolutionLayerResolver.Descriptor
        :rtype: int
        """
        input_dims = converter_context.get_input_layer_output_shape_for(descriptor.input_ops[0])

        if descriptor.bias_op:
            output_dims = converter_context.graph_helper.get_op_output_shape(descriptor.bias_op)
        else:
            output_dims = converter_context.graph_helper.get_op_output_shape(descriptor.output_op)

        pad_y, pad_x, padding_strategy = ConvolutionLayerBuilder.calculate_padding_size(input_size=input_dims[-3:-1],
                                                                                        output_size=output_dims[-3:-1],
                                                                                        strides=descriptor.strides[1:3],
                                                                                        padding=descriptor.padding,
                                                                                        filter_dims=descriptor.weights.shape,
                                                                                        dilation=[descriptor.dilationY,
                                                                                                  descriptor.dilationX])

        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)

        # using layer name for output buffer name
        return converter_context.model.add_conv_layer(name=descriptor.layer_name,
                                                      weights=descriptor.weights,
                                                      bias=descriptor.biases,
                                                      padx=pad_x,
                                                      pady=pad_y,
                                                      padding_mode=snpe.modeltools.PADDING_ZERO,
                                                      padding_size_strategy=padding_strategy,
                                                      stridex=int(descriptor.strides[1]),
                                                      stridey=int(descriptor.strides[2]),
                                                      dilationx=descriptor.dilationX,
                                                      dilationy=descriptor.dilationY,
                                                      input_name=input_name,
                                                      output_name=descriptor.output_names[0],
                                                      groups=descriptor.groups)

    @classmethod
    def calculate_padding_size(cls, input_size, output_size, strides, padding, filter_dims, dilation):
        pad_y, pad_x = 0, 0
        padding_size_strategy = snpe.modeltools.PADDING_SIZE_IMPLICIT_VALID
        if padding == "SAME":
            filter_h = filter_dims[0] + (filter_dims[0] - 1) * (dilation[0] - 1)
            filter_w = filter_dims[1] + (filter_dims[1] - 1) * (dilation[1] - 1)
            pad_y = max(((output_size[0] - 1) * strides[0] + filter_h - input_size[0]), 0)
            pad_x = max(((output_size[1] - 1) * strides[1] + filter_w - input_size[1]), 0)
            # We divide by two and truncate if odd padding given the runtime will
            # take care of Asymmetry
            pad_y /= 2
            pad_x /= 2
            padding_size_strategy = snpe.modeltools.PADDING_SIZE_IMPLICIT_SAME
        return int(pad_y), int(pad_x), padding_size_strategy

    def transform_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):

        filtered_descriptors = [d for d in output_descriptors if isinstance(d, BatchNormLayerResolver.Descriptor)]
        if filtered_descriptors == output_descriptors and len(output_descriptors) == 1:
            descriptor.weights = descriptor.weights * filtered_descriptors[0].weights
            descriptor.biases = (descriptor.biases * filtered_descriptors[0].weights) + filtered_descriptors[0].biases
            descriptor.output_names = output_descriptors[0].output_names
            converter_context.merge_descriptors(output_descriptors[0], descriptor)


class GroupedConvolutionLayerResolver(ConvolutionLayerResolver, object):
    class Descriptor(ConvolutionLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(GroupedConvolutionLayerResolver, self).__init__()
        tree_output_node = ConverterSequenceNode('conv_op', ['Conv2D'])
        self.sequence = GraphSequence([
            ConverterSequenceNode('a', ['Split']),
            ConverterSequenceNode('b', ['Split']),
            ConverterRepeatableSequenceTreeNode('repeatable_graph', tree_output_node, tree_output_node),
            ConverterSequenceNode('concat_op', ['Concat']),
            ConverterSequenceNode('weights', ['Identity', 'Const']),
            NonConsumableConverterSequenceNode('inputs', ['?']),
            NonConsumableConverterSequenceNode('concat_dim', ['Const']),
            NonConsumableConverterSequenceNode('split_dim1', ['Const']),
            ConverterSequenceNode('split_dim2', ['Const'])
        ])
        self.sequence.set_inputs('a', ['inputs', 'split_dim1'])
        self.sequence.set_inputs('b', ['weights', 'split_dim2'])
        self.sequence.set_inputs('repeatable_graph', ['a', 'b'])
        self.sequence.set_inputs('concat_op', ['repeatable_graph', 'concat_dim'])
        self.sequence.set_outputs(['concat_op'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        if len(matches) == 0:
            return []
        descriptors = []
        for match in matches:
            conv_op = match['conv_op_1']
            strides = conv_op.get_attr(self.TF_ATTRIBUTE_STRIDES)
            padding = conv_op.get_attr(self.TF_ATTRIBUTE_PADDING)
            weights = match['weights']
            consumed_nodes = match.consumed_nodes
            output_op_nodes_names = [str(match[node.identifier].outputs[0].name) for node in
                                     self.sequence.output_nodes]
            try:
                concat_op = match['concat_op']
                concat_op_output_ops = graph_helper.get_op_outputs(concat_op)
                bias_op = GraphHelper.filter_single_op_by_type(concat_op_output_ops, 'BiasAdd')
                # need to consume input of bias
                biases = self.get_biases(graph_helper, conv_op, bias_op)
                consumed_nodes.append(bias_op)
                output_op_nodes_names = [str(bias_op.outputs[0].name)]
            except OperationNotFoundError:
                bias_op = None
                biases = np.zeros(weights.outputs[0].get_shape()[-1], dtype=np.float32)

            weights = graph_helper.evaluate_tensor_output(weights.outputs[0])
            descriptor = ConvolutionLayerResolver.Descriptor(str(conv_op.name), consumed_nodes, conv_op, bias_op,
                                                             strides, padding, weights, biases,
                                                             output_names=output_op_nodes_names)
            descriptor.input_ops = [match['a'], match['b']]
            descriptors.append(descriptor)
        return descriptors
