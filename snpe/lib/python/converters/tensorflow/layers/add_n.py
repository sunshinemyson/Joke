#!/usr/bin/env python
# //=============================================================================
# //  @@-COPYRIGHT-START-@@
# //
# //  Copyright 2016-2017 Qualcomm Technologies, Inc. All rights reserved.
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
from converters import code_to_message
from converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from converters.tensorflow.util import ConverterError
from converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence
)


class AddNLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes):
            super(AddNLayerResolver.Descriptor, self).__init__('ElementWiseSumN', name, nodes)

    def __init__(self):
        self.sequence = GraphSequence([ConverterSequenceNode('root', ['AddN'])])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        if len(matches) == 0:
            return []

        descriptors = []
        for match in matches:
            add_op = match['root']
            descriptors.append(AddNLayerResolver.Descriptor(str(add_op.name), match.consumed_nodes))
        return descriptors


class AddNLayerBuilder(LayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ConcatLayerResolver.Descriptor
        :rtype: int
        """
        input_names = self.get_input_names(converter_context, descriptor, input_descriptors)
        if len(input_names) < 2:
            raise ConverterError(code_to_message.get_message('ERROR_TF_ADD_N_NUM_OF_INPUTS')(descriptor.layer_name))
        output_name = descriptor.output_names[0]
        current_input_names = [input_names[0], input_names[1]]
        current_output_name = descriptor.layer_name + '_unroll_1'
        converter_context.model.add_elementwise_sum_layer(descriptor.layer_name + '_unroll_1',
                                                          [1.0 for _ in current_input_names],
                                                          current_input_names,
                                                          current_output_name)

        for input_index in range(2, len(input_names) - 1):
            current_input_names = [current_output_name, input_names[input_index]]
            current_output_name = descriptor.layer_name + '_unroll_' + str(input_index)
            converter_context.model.add_elementwise_sum_layer(descriptor.layer_name + '_unroll_' + str(input_index),
                                                              [1.0 for _ in current_input_names],
                                                              current_input_names,
                                                              current_output_name)
        current_input_names = [current_output_name, input_names[-1]]
        return converter_context.model.add_elementwise_sum_layer(descriptor.layer_name,
                                                                 [1.0 for _ in current_input_names],
                                                                 current_input_names,
                                                                 output_name)
