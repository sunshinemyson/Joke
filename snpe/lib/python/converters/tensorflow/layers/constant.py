#!/usr/bin/env python
# //=============================================================================
# //  @@-COPYRIGHT-START-@@
# //
# //  Copyright 2015-2018 Qualcomm Technologies, Inc. All rights reserved.
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
from converters.tensorflow.layers.ignored_patterns import IgnoredLayersResolver
from converters.tensorflow.util import ConverterError


class ConstantLayerResolver(LayerResolver, object):
    def resolve_layer(self, graph_matcher, graph_helper):
        raise ConverterError('Constant layers are resolved by other resolvers!')

    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, value, shape, consumer):
            super(ConstantLayerResolver.Descriptor, self).__init__('Constant', name, nodes)
            self.value = value
            self.shape = shape
            self.consumer = consumer


class ConstantLayerBuilder(LayerBuilder):

    def transform_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        ignored = [d for d in output_descriptors if isinstance(d, IgnoredLayersResolver.Descriptor)]
        if ignored == output_descriptors:
            descriptor.set_ignored(True)

        if len(output_descriptors) == 1 and not descriptor.consumer == output_descriptors[0]:
            descriptor.set_ignored(True)

    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: FillLayerResolver.Descriptor
        :rtype: int
        """
        if not isinstance(descriptor.value, np.ndarray):
            array = np.zeros(descriptor.shape, dtype=np.float32)
            array[...] = descriptor.value
            descriptor.value = array
            descriptor.shape = list(array.shape)
        return converter_context.model.add_const_layer(descriptor.output_names[0],
                                                       list(np.shape(descriptor.value)),
                                                       descriptor.value)
