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
import numpy as np
from converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from converters.tensorflow.util import ConverterError, expand_to_rank
from converters.tensorflow.layers.constant import ConstantLayerResolver
from abc import ABCMeta
from abc import abstractmethod
from converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)


class EltWiseLayerResolver(LayerResolver, object):
    __metaclass__ = ABCMeta

    def __init__(self, layer_type, op_type, descriptor_class):
        super(EltWiseLayerResolver, self).__init__()
        self._layer_type = layer_type
        self._op_type = op_type
        self._descriptor_class = descriptor_class

        self.sequence = GraphSequence([ConverterSequenceNode('root', [self._op_type])])
        self.sequence.set_outputs(['root'])

        self.sequence_with_identity = GraphSequence([
            ConverterSequenceNode('root', [self._op_type]),
            ConverterSequenceNode('identity', ['Identity'])
        ])
        self.sequence_with_identity.set_inputs('identity', ['root'])
        self.sequence_with_identity.set_outputs(['identity'])

        self.sequence_with_const_input = GraphSequence([
            ConverterSequenceNode('root', [self._op_type]),
            NonConsumableConverterSequenceNode('const', ['Const', 'Identity']),
            NonConsumableConverterSequenceNode('other', ['?'])
        ])
        self.sequence_with_const_input.set_inputs('root', ['const', 'other'])
        self.sequence_with_const_input.set_outputs(['root'])

        self.sequence_with_const_input_and_identity = GraphSequence([
            ConverterSequenceNode('root', [self._op_type]),
            ConverterSequenceNode('identity', ['Identity']),
            NonConsumableConverterSequenceNode('const', ['Const']),
            NonConsumableConverterSequenceNode('other', ['?'])
        ])
        self.sequence_with_const_input_and_identity.set_inputs('root', ['const', 'other'])
        self.sequence_with_const_input_and_identity.set_inputs('identity', ['root'])
        self.sequence_with_const_input_and_identity.set_outputs(['identity'])

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        non_const_input_sequences = [self.sequence_with_identity, self.sequence]
        for sequence in non_const_input_sequences:
            for match in graph_matcher.match_sequence(sequence):
                eltwise_op = match['root']
                consumed_nodes = match.consumed_nodes
                descriptor = self._descriptor_class(self._layer_type, str(eltwise_op.name), consumed_nodes)
                descriptors.append(descriptor)

        const_input_sequences = [self.sequence_with_const_input_and_identity, self.sequence_with_const_input]
        for sequence in const_input_sequences:
            for match in graph_matcher.match_sequence(sequence):
                eltwise_op = match['root']
                const_op = match['const']
                const_tensor = graph_helper.evaluate_tensor_output(const_op.outputs[0])
                eltwise_shape = graph_helper.get_op_output_shape(eltwise_op)
                eltwise_shape = expand_to_rank(eltwise_shape, 3)
                if len(eltwise_shape) > 3:
                    eltwise_shape = eltwise_shape[-3:]
                if list(const_tensor.shape) != eltwise_shape:
                    const_tensor = self._broadcast_tensor(const_tensor, eltwise_shape)

                eltwise_descriptor = self._descriptor_class(self._layer_type, str(eltwise_op.name), match.consumed_nodes)
                const_descriptor = ConstantLayerResolver.Descriptor(str(const_op.name), [const_op], const_tensor,
                                                                    eltwise_shape, eltwise_descriptor)
                descriptors.extend([eltwise_descriptor, const_descriptor])

        return descriptors

    def _broadcast_tensor(self, tensor, shape):
        raise ConverterError('ElementWise resolver must implement broadcast method.')


class EltWiseLayerBuilder(LayerBuilder):
    __metaclass__ = ABCMeta

    @abstractmethod
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        pass


class EltWiseSumLayerResolver(EltWiseLayerResolver):
    class Descriptor(LayerDescriptor):
        pass

    def __init__(self):
        super(EltWiseSumLayerResolver, self).__init__('ElementWiseSum', 'Add', EltWiseSumLayerResolver.Descriptor)

    def _broadcast_tensor(self, tensor, shape):
        broadcasted_tensor = np.zeros(shape, dtype=np.float32)
        broadcasted_tensor += tensor
        return broadcasted_tensor


class EltWiseSumLayerBuilder(LayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: EltWiseSumLayerResolver.Descriptor
        :rtype: int
        """
        input_names = self.get_input_names(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return converter_context.model.add_elementwise_sum_layer(descriptor.layer_name,
                                                                 [1.0 for _ in input_names],
                                                                 input_names,
                                                                 output_name)


class EltWiseMulLayerResolver(EltWiseLayerResolver):
    class Descriptor(LayerDescriptor):
        pass

    def __init__(self):
        super(EltWiseMulLayerResolver, self).__init__('ElementWiseMul', 'Mul', EltWiseMulLayerResolver.Descriptor)

    def _broadcast_tensor(self, tensor, shape):
        broadcasted_tensor = np.ones(shape, dtype=np.float32)
        broadcasted_tensor *= tensor
        return broadcasted_tensor


class EltWiseMulLayerBuilder(LayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: EltWiseMulLayerResolver.Descriptor
        :rtype: int
        """
        input_names = self.get_input_names(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return converter_context.model.add_elementwise_product_layer(descriptor.layer_name,
                                                                     input_names,
                                                                     output_name)


class EltWiseMaxLayerResolver(EltWiseLayerResolver):
    class Descriptor(LayerDescriptor):
        pass

    def __init__(self):
        super(EltWiseMaxLayerResolver, self).__init__('ElementWiseMax', 'Maximum', EltWiseMaxLayerResolver.Descriptor)


class EltWiseMaxLayerBuilder(LayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: EltWiseMaxLayerResolver.Descriptor
        :rtype: int
        """
        input_names = self.get_input_names(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return converter_context.model.add_elementwise_max_layer(descriptor.layer_name,
                                                                 input_names,
                                                                 output_name)
