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
from converters import code_to_message
from converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from converters.tensorflow.util import ConverterError
from converters.tensorflow.util import GraphHelper
from converters.tensorflow.util import scoped_op_name
from converters.tensorflow.layers.lstm import LstmLayerResolver
from converters.tensorflow.layers.ignored_patterns import IgnoredLayersResolver
from converters.tensorflow.layers.fill import FillLayerResolver

from converters.tensorflow.graph_matcher import(
    ConverterSequenceNode,
    GraphSequence,
    GraphMatcher
)


class ConcatLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, axis):
            super(ConcatLayerResolver.Descriptor, self).__init__('Concatenation', name, nodes)
            self.axis = axis

    def __init__(self):
        self.sequence = GraphSequence([ConverterSequenceNode('root', ['Concat', 'ConcatV2'])])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        if len(matches) == 0:
            return []
        descriptors = []
        for match in matches:
            concat_op = match['root']
            non_const_inputs = [tensor for tensor in concat_op.inputs if tensor.op.type != 'Const']
            if len(non_const_inputs) < 2:
                continue

            max_shape = 0
            for t in non_const_inputs:
                shape = graph_helper.get_op_output_shape(t.op)
                if len(shape) > max_shape:
                    max_shape = len(shape)

            axis_tensor = GraphHelper.filter_single_op_by_type([t.op for t in concat_op.inputs], 'Const')
            axis = int(graph_helper.evaluate_tensor_output(axis_tensor.outputs[0]))
            if max_shape == 4:
                axis = axis - 1
            consumed_nodes = match.consumed_nodes
            descriptors.append(ConcatLayerResolver.Descriptor(str(concat_op.name), consumed_nodes, axis))

        return descriptors


class ConcatLayerBuilder(LayerBuilder):

    def transform_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        if len(input_descriptors) == 1 and isinstance(input_descriptors[0], IgnoredLayersResolver.Descriptor):
            descriptor.set_ignored(True)
            return

        lstm_inputs = [d for d in input_descriptors if
                       isinstance(d, LstmLayerResolver.UnrolledTimeStepDescriptor) or
                       isinstance(d, LstmLayerResolver.StateDescriptor)]
        if lstm_inputs == input_descriptors:
            converter_context.merge_descriptors(descriptor, lstm_inputs[0])

    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ConcatLayerResolver.Descriptor
        :rtype: int
        """
        if len(input_descriptors) < 2:
            raise ConverterError(code_to_message.get_message('ERROR_TF_CONCAT_INPUT'))

        input_names = self.get_input_names(converter_context, descriptor, input_descriptors)
        return converter_context.model.add_concatenation_layer(descriptor.layer_name,
                                                               input_names,
                                                               descriptor.output_names[0],
                                                               descriptor.axis)
