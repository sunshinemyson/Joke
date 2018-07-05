#!/usr/bin/env python
# //=============================================================================
# //  @@-COPYRIGHT-START-@@
# //
# //  Copyright 2015-2017 Qualcomm Technologies, Inc. All rights reserved.
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
from converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from converters.tensorflow.util import GraphHelper
from converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)


class CropLayerResolver(LayerResolver, object):

    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, offset, size):
            super(CropLayerResolver.Descriptor, self).__init__('Crop', name, nodes)
            self.offset = offset
            self.size = size

    def __init__(self):
        self.sequence = GraphSequence([
            ConverterSequenceNode('root', ['Slice']),
            NonConsumableConverterSequenceNode('input', ['?']),
            NonConsumableConverterSequenceNode('offsets', ['?']),
            NonConsumableConverterSequenceNode('size', ['?']),
        ])
        self.sequence.set_inputs('root', ['input', 'offsets', 'size'])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        descriptors = []
        for match in matches:
            slice_op = match['root']
            input_shape = graph_helper.get_op_output_shape(match['input'])
            offset = graph_helper.evaluate_tensor_output(match['offsets'].outputs[0])
            size = graph_helper.evaluate_tensor_output(match['size'].outputs[0])
            for index in range(0, len(size)):
                if size[index] == -1:
                    size[index] = input_shape[index] - offset[index]

            consumed_nodes = match.consumed_nodes
            descriptors.append(
                CropLayerResolver.Descriptor(str(slice_op.name), consumed_nodes, offset, size))
        return descriptors


class CropLayerBuilder(LayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ReshapeLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        return converter_context.model.add_crop_layer(descriptor.output_names[0],
                                                      descriptor.offset.tolist(),
                                                      descriptor.size.tolist(),
                                                      input_name,
                                                      descriptor.output_names[0])
