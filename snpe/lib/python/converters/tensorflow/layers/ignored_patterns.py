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
from converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from converters.tensorflow.sequences.ignored import (
    ignored_sequence_1,
    ignored_sequence_2,
    dropout_cell_sequence,
    real_div_sequence,
    identity_sequence,
    placeholder_with_default_sequence,
)


class IgnoredLayersResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes):
            super(IgnoredLayersResolver.Descriptor, self).__init__('IgnoredLayer', name, nodes)
            # define pattern one to be ignored

    def __init__(self):
        self.sequences = [
            ignored_sequence_1,
            ignored_sequence_2,
            dropout_cell_sequence,
            real_div_sequence,
            identity_sequence,
            placeholder_with_default_sequence,
        ]

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for pattern_output_nodes in self.sequences:
            matches = graph_matcher.match_sequence(pattern_output_nodes)
            if len(matches) == 0:
                continue

            for match in matches:
                consumed_nodes = match.consumed_nodes
                d = IgnoredLayersResolver.Descriptor(str(consumed_nodes[0].name), consumed_nodes)
                descriptors.append(d)

        return descriptors


class IgnoredLayersBuilder(LayerBuilder):

    def transform_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        descriptor.set_ignored(True)

    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ConcatLayerResolver.Descriptor
        :rtype: int
        """
        return None
