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
import snpe

from converters.tensorflow.common import LayerDescriptor
from converters.tensorflow.layers.relu_min_max import ReluMinMaxLayerResolver
from converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence
)


class Relu6LayerResolver(ReluMinMaxLayerResolver, object):

    class Descriptor(ReluMinMaxLayerResolver.Descriptor):
        def __init__(self, name, nodes):
            super(Relu6LayerResolver.Descriptor, self).__init__('Relu6', name, nodes, min_clamp=0, max_clamp=6)

    def __init__(self):
        self.sequence = GraphSequence([ConverterSequenceNode('root', ['Relu6'])])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        if len(matches) == 0:
            return []
        potential_descriptors = []
        for match in matches:
            relu6_op = match['root']
            consumed_nodes = match.consumed_nodes
            potential_descriptors.append(
                Relu6LayerResolver.Descriptor(str(relu6_op.name), consumed_nodes))
        return potential_descriptors
