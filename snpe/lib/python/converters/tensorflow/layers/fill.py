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
import numpy as np
from converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from converters.tensorflow.util import GraphHelper
from converters.tensorflow.graph_matcher import(
    ConverterSequenceNode,
    GraphSequence
)


class FillLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, shape, scalar):
            super(FillLayerResolver.Descriptor, self).__init__('Fill', name, nodes)
            self.shape = shape
            self.scalar = scalar

    def __init__(self):
        self.sequence = GraphSequence([ConverterSequenceNode('root', ['Fill'])])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        if len(matches) == 0:
            return []
        descriptors = []
        for match in matches:
            fill_op = match['root']
            consumed_nodes = match.consumed_nodes
            shape_tensor, scalar_tensor = GraphHelper.get_op_input_tensors(fill_op, ('?', 'Const'))
            shape = graph_helper.evaluate_tensor_output(shape_tensor).tolist()
            while len(shape) > 3:
                shape = shape[1:]

            while len(shape) < 3:
                shape = [1] + shape
            scalar = graph_helper.evaluate_tensor_output(scalar_tensor)

            d = FillLayerResolver.Descriptor(str(fill_op.name), consumed_nodes, shape, scalar)
            descriptors.append(d)

        return descriptors


class FillLayerBuilder(LayerBuilder):

    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: FillLayerResolver.Descriptor
        :rtype: int
        """
        tensor = np.zeros(descriptor.shape, dtype=np.float32)
        tensor[...] = descriptor.scalar

        return converter_context.model.add_const_layer(descriptor.output_names[0],
                                                       list(np.shape(tensor)),
                                                       tensor)
