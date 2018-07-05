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
    GraphSequence
)


class ResizeBilinearLayerResolver(LayerResolver, object):
    TF_ATTRIBUTE_ALIGN_CORNERS = 'align_corners'

    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, input_tensor_shape, align_corners=False):
            super(ResizeBilinearLayerResolver.Descriptor, self).__init__('Resize', name, nodes)
            self.align_corners = align_corners
            self.input_tensor_shape = input_tensor_shape
            self.resize_mode = 0

    def __init__(self):
        self.sequence = GraphSequence([ConverterSequenceNode('root', ['ResizeBilinear'])])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        if len(matches) == 0:
            return []
        descriptors = []
        for match in matches:
            resize_op = match['root']
            align_corners_bool = resize_op.get_attr(self.TF_ATTRIBUTE_ALIGN_CORNERS)
            input_tensor, _ = GraphHelper.get_op_input_tensors(resize_op, ('?', '?'))
            input_tensor_shape = graph_helper.get_op_output_shape(input_tensor)
            consumed_nodes = match.consumed_nodes
            descriptors.append(
                ResizeBilinearLayerResolver.Descriptor(str(resize_op.name), consumed_nodes,
                                               input_tensor_shape, align_corners_bool))
        return descriptors


class ResizeNearestNeighborLayerResolver(LayerResolver, object):
    TF_ATTRIBUTE_ALIGN_CORNERS = 'align_corners'

    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, input_tensor_shape, align_corners=False):
            super(ResizeNearestNeighborLayerResolver.Descriptor, self).__init__('ResizeNearestNeighbor', name, nodes)
            self.align_corners = align_corners
            self.input_tensor_shape = input_tensor_shape
            self.resize_mode = 1

    def __init__(self):
        self.sequence = GraphSequence([ConverterSequenceNode('root', ['ResizeNearestNeighbor'])])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        if len(matches) == 0:
            return []
        descriptors = []
        for match in matches:
            resize_op = match['root']
            align_corners_bool = resize_op.get_attr(self.TF_ATTRIBUTE_ALIGN_CORNERS)
            input_tensor, _ = GraphHelper.get_op_input_tensors(resize_op, ('?', '?'))
            input_tensor_shape = graph_helper.get_op_output_shape(input_tensor)
            consumed_nodes = match.consumed_nodes
            descriptors.append(
                ResizeNearestNeighborLayerResolver.Descriptor(str(resize_op.name), consumed_nodes,
                                               input_tensor_shape, align_corners_bool))
        return descriptors


class ResizeLayerBuilder(LayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ReshapeLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_shape = converter_context.graph_helper.get_op_output_shape(descriptor.child_ops[0])
        output_shape = output_shape[-3:] if len(output_shape) > 3 else output_shape
        return converter_context.model.add_scaling_layer(descriptor.output_names[0],
                                                         output_shape,
                                                         pad_value=0.0,
                                                         maintain_aspect_ratio=False,
                                                         resize_mode=descriptor.resize_mode,
                                                         scale_height=0.0,
                                                         scale_width=0.0,
                                                         input_name=input_name,
                                                         output_name=descriptor.output_names[0],
                                                         align_corners=descriptor.align_corners)
