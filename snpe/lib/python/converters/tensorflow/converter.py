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
import logging
from collections import OrderedDict
import snpe

import converters.code_to_message as code_to_message
import converters.tensorflow.layers as layers
from converters.tensorflow.layers.ignored_patterns import IgnoredLayersResolver
from converters.tensorflow.common import LayerDescriptor
from converters.tensorflow.common import InputLayerDescriptor
from converters.tensorflow.util import (
    ConverterError,
    GraphHelper,
    uniques
)
from converters.tensorflow.graph_matcher import (
    GraphMatcher,
    TFGraphBuilder
)


class TopologyResolver(object):

    class Topology(object):
        def __init__(self):
            self.inputs = []
            self.outputs = []

    def __init__(self):
        self._descriptor_topology_map = dict()
        """
        :type: dict(LayerDescriptor,TopologyResolver.Topology)
        """
        self._descriptor_ops_map = dict()
        """
        :type: dict(tensorflow.Operation,LayerDescriptor)
        """

    @property
    def descriptor_ops_map(self):
        """ :rtype: descriptor_ops_map """
        return self._descriptor_ops_map

    def resolve_topology(self, descriptors):
        """
        :type descriptors: list(LayerDescriptor)
        :rtype: list(LayerDescriptor)
        """
        self._descriptor_topology_map.clear()
        self._descriptor_ops_map.clear()

        for d in descriptors:
            self._descriptor_topology_map[d.layer_name] = TopologyResolver.Topology()
            for op in d.child_ops:
                self._descriptor_ops_map[op] = d

        for d in descriptors:
            topology = self._descriptor_topology_map[d.layer_name]
            inputs = self._get_input_layers_for(d)
            for i in inputs:
                input_topology = self._descriptor_topology_map[i.layer_name]
                input_topology.outputs.append(d)
            topology.inputs.extend(inputs)

    def get_input_layers_for(self, descriptor):
        return self._descriptor_topology_map[descriptor.layer_name].inputs

    def get_output_layers_for(self, descriptor):
        return self._descriptor_topology_map[descriptor.layer_name].outputs

    def sort_descriptors_in_execution_order(self, _descriptors, _input_descriptors):
        """
        :type _descriptors: list(LayerDescriptor)
        :type _input_descriptors: list(LayerDescriptor)
        :rtype: list(LayerDescriptor)
        """
        sorted_descriptors = []
        queue = list(_input_descriptors)
        visited = set()
        ready = set()
        while len(queue) > 0:
            head = queue.pop(0)
            visited.add(head)
            input_descriptors = self.get_input_layers_for(head)
            if all(i in ready for i in input_descriptors):
                if head in ready:
                    continue

                sorted_descriptors.append(head)
                ready.add(head)

                for o in self.get_output_layers_for(head):
                    if o not in _descriptors or o in visited:
                        continue
                    queue.append(o)
            else:
                for i in input_descriptors:
                    if i in ready or i in visited:
                        continue
                    queue.append(i)
                queue.append(head)

        return sorted_descriptors[len(_input_descriptors):]

    def _get_input_layers_for(self, descriptor):
        """
        :type descriptor: LayerDescriptor
        :rtype: list[LayerDescriptor]
        """
        predecessors = []
        descriptor_input_ops = [op for op in descriptor.child_ops if descriptor.is_input_op(op)]
        for o in descriptor_input_ops:
            q = [t.op for t in o.inputs]
            visited = set()
            while len(q) > 0:
                next_op = q.pop(0)
                if next_op in visited:
                    continue
                visited.add(next_op)

                d = self._descriptor_ops_map.get(next_op, None)
                if d is None or d == descriptor:
                    continue

                if d.is_ignored:
                    q.extend([t.op for t in next_op.inputs])
                else:
                    predecessors.append(d)
        return uniques(predecessors)


class ConverterContext(object):
    def __init__(self, converter_model, dnn_model, graph_helper, topology_resolver, logger):
        """
        This class contains state information pertaining a model during conversion.
        It is shared with LayerBuilder instances in order to retrieve layer connectivity, etc.
        :type converter_model: converters.tensorflow.loader.Model
        :type dnn_model: snpe.modeltools.Model
        :type graph_helper: converters.tensorflow.util.GraphHelper
        :type topology_resolver: converters.tensorflow.converter.TopologyResolver
        :type logger: logging.Logger
        """
        super(ConverterContext, self).__init__()
        self.__converter_model = converter_model
        self.__dnn_model = dnn_model
        self.__logger = logger
        self.__graph_helper = graph_helper
        self._topology_resolver = topology_resolver  # type: converters.tensorflow.converter.TopologyResolver

    @property
    def session(self):
        """ :rtype: tensorflow.Session """
        return self.__converter_model.session

    @property
    def graph(self):
        """ :rtype tensorflow.Graph """
        return self.session.graph

    @property
    def model(self):
        """ :rtype: snpe.modeltools.Model """
        return self.__dnn_model

    @property
    def logger(self):
        """ :rtype: logging.Logger """
        return self.__logger

    @property
    def inputs(self):
        """ :rtype: list[converters.tensorflow.loader.Model.Input] """
        return self.__converter_model.inputs

    @property
    def graph_helper(self):
        """ :rtype: converters.tensorflow.util.GraphHelper """
        return self.__graph_helper

    def get_input_layer_output_shape_for(self, operation):
        """
        :type operation: tensorflow.Operation
        :rtype: [int]
        """
        output_op = self._get_input_layer_output_op_for(operation)
        return self.__graph_helper.get_op_output_shape(output_op)

    def get_output_tensors_between(self, descriptor_from, descriptor_to):
        """
        :type descriptor_from: LayerDescriptor
        :type descriptor_to: LayerDescriptor
        :rtype: list[tensorflow.Tensor]
        """
        tensors = []
        for o in descriptor_to.child_ops:
            ts = self._get_input_layers_output_tensors_for(o)
            for t in ts:
                d = self._topology_resolver.descriptor_ops_map.get(t.op, None)
                if d == descriptor_from:
                    tensors.append(t)
        return uniques(tensors)

    @classmethod
    def merge_descriptors(cls, source, destination):
        destination.child_ops.extend(source.child_ops)
        source.child_ops = []
        source.set_ignored(True)

    def _get_input_layers_output_tensors_for(self, operation):
        """
        :type operation: tensorflow.Operation
        :rtype: list[tensorflow.Tensor]
        """
        descriptor = self._topology_resolver.descriptor_ops_map.get(operation, None)
        if descriptor is None:
            raise ConverterError('Unable to find input layer for operation not in layer.')

        output_tensors = []

        input_descriptors = self._topology_resolver.get_input_layers_for(descriptor)
        input_descriptors_outputs = [o for d in input_descriptors for o in d.child_ops if d.is_output_op(o)]

        visited = set()
        op_queue = [operation]
        while len(op_queue) > 0:
            next_op = op_queue.pop(0)
            visited.add(next_op)
            for input_tensor in next_op.inputs:
                input_op = input_tensor.op
                if input_op in input_descriptors_outputs:
                    output_tensors.append(input_tensor)
                elif input_op not in visited:
                    op_queue.insert(0, input_op)

        return uniques(output_tensors)

    def _get_input_layer_output_op_for(self, operation):
        """
        :type operation: tensorflow.Operation
        :rtype: tensorflow.Operation
        """
        input_tensors = self._get_input_layers_output_tensors_for(operation)
        ops = uniques([t.op for t in input_tensors])
        if len(ops) == 0:
            raise ConverterError(code_to_message.get_message('ERROR_TF_INPUT_OPERATION_NOT_FOUND')(operation.name))
        if len(ops) != 1:
            raise ConverterError(code_to_message.get_message('ERROR_TF_EXPECTED_SINGLE_OUTPUT_FROM_PREVIOUS_LAYER'))
        return ops[0]


class DlcConverter(object):

    def __init__(self, model, strict_node_resolution):
        """
        :type model: converters.tensorflow.loader.Model
        :type strict_node_resolution: bool
        """
        self._logger = logging.getLogger()  # type: logging.Logger
        self._context = None  # type: ConverterContext
        self._model = model
        self._strict_node_resolution = strict_node_resolution
        self._ops = self._resolve_graph_operations_from_model(model)
        self._graph_helper = None
        self._input_descriptors = []
        self._topology_resolver = None

    def convert(self, dlc_output_path, model_version, converter_command):
        """
        :type dlc_output_path: str
        :type model_version: str
        :type converter_command: str
        :rtype: None
        """
        self._graph_helper = GraphHelper(self._model.session, self._model, self._ops)
        self._topology_resolver = TopologyResolver()
        self._context = ConverterContext(self._model, snpe.modeltools.Model(), self._graph_helper,
                                         self._topology_resolver, self._logger)
        self._logger.info(code_to_message.get_progress_message('INFO_ALL_BUILDING_NETWORK'))
        self._convert_input_layers()
        self._convert_layers()
        self._set_model_version(model_version)
        self._context.model.set_converter_command(converter_command)
        self._context.model.save(dlc_output_path)

    def _convert_input_layers(self):
        """
        :rtype: None
        """
        for model_input in self._context.inputs:
            input_operation = self._context.graph.get_operation_by_name(model_input.name)
            shape = self._graph_helper.get_op_output_shape(input_operation)
            if None in shape:
                message = code_to_message.get_message('ERROR_TF_UNABLE_TO_RESOLVE_GRAPH_INPUT_DIMS')
                raise ConverterError(message(model_input.name))
            if model_input.shape != shape:
                message = code_to_message.get_message('ERROR_TF_UNEXPECTED_INPUT_SHAPE')
                raise ConverterError(message(model_input.shape, shape))

            if len(shape) > 3:
                shape = shape[-3:]

            self._logger.info(
                code_to_message.get_progress_message('INFO_TF_BUILDING_INPUT_LAYER')(input_operation.name, shape))

            layer_name = str(input_operation.outputs[0].name)
            descriptor = InputLayerDescriptor(layer_name, [input_operation])
            self._input_descriptors.append(descriptor)
            self._ops.remove(input_operation)
            self._context.model.add_data_layer(descriptor.output_names[0], shape, 'rgb', 'rgb', model_input.type)

    def _convert_layers(self):
        """
        :rtype: None
        """
        graph_ops = list(self._ops)
        descriptors = self._resolve_descriptors_from_nodes(graph_ops)
        descriptors = self._resolve_hierarchical_resolution_conflicts(descriptors)

        self._topology_resolver.resolve_topology(self._input_descriptors + descriptors)
        descriptors = self._topology_resolver.sort_descriptors_in_execution_order(descriptors, self._input_descriptors)
        descriptors = self._filter_disconnected_descriptors(descriptors)
        self._transform_descriptors(descriptors)

        self._topology_resolver.resolve_topology(self._input_descriptors + descriptors)
        descriptors = [d for d in descriptors if not d.is_ignored]

        if self._strict_node_resolution:
            self._assert_all_ops_consumed(descriptors, graph_ops)

        self._create_layers(descriptors)

    def _assert_all_ops_consumed(self, descriptors, graph_ops):
        graph_ops = self._filter_unconsumed_ops(descriptors, graph_ops)

        def is_parameter_op(o):
            return o.type in ['Const', 'Identity', 'Variable']

        remaining_ops = [op for op in graph_ops if not is_parameter_op(op)]
        for op in remaining_ops:
            self._logger.warning(code_to_message.get_warning_message('WARNING_TF_SCOPE_OP_NOT_CONSUMED')(op.name,
                                                                                                         op.type))
        if len(remaining_ops) > 0:
            raise ConverterError(code_to_message.get_message('ERROR_TF_OPERATION_NOT_MAPPED_TO_LAYER'))

    def _filter_disconnected_descriptors(self, descriptors):
        output_descriptors = [descriptor for op, descriptor in self._topology_resolver.descriptor_ops_map.iteritems() if
                              op.name in self._model.out_nodes_names]
        descriptors_queue = list(set(output_descriptors))
        result = list(output_descriptors)
        while len(descriptors_queue) > 0:
            current_descriptor = descriptors_queue.pop(0)
            inputs = self._topology_resolver.get_input_layers_for(current_descriptor)
            for input_descriptor in inputs:
                if input_descriptor in descriptors and input_descriptor not in result:
                    descriptors_queue.append(input_descriptor)
                    result.append(input_descriptor)
        descriptors_to_ignore = set(descriptors) - set(result)
        for descriptor in descriptors:
            if descriptor in descriptors_to_ignore:
                descriptor.set_ignored(True)
        return descriptors

    def _create_layers(self, descriptors):
        for descriptor in descriptors:
            layer_builder = self._create_layer_builder(descriptor)
            self._create_layer(layer_builder, descriptor)

    def _transform_descriptors(self, descriptors):
        for descriptor in descriptors:
            layer_builder = self._create_layer_builder(descriptor)
            inputs = self._topology_resolver.get_input_layers_for(descriptor)
            outputs = self._topology_resolver.get_output_layers_for(descriptor)
            layer_builder.transform_layer(self._context, descriptor, inputs, outputs)

    @classmethod
    def _resolve_hierarchical_resolution_conflicts(cls, descriptors):
        """
        :type descriptors: list(LayerDescriptor)
        :rtype: list(LayerDescriptor)
        """
        op_to_descriptor = OrderedDict()
        for d in descriptors:
            for o in d.child_ops:
                current_descriptor = op_to_descriptor.get(o, None)
                if current_descriptor:
                    if len(d.child_ops) > len(current_descriptor.child_ops):
                        op_to_descriptor[o] = d
                    elif len(d.child_ops) == len(current_descriptor.child_ops) and isinstance(current_descriptor, IgnoredLayersResolver.Descriptor):
                        op_to_descriptor[o] = d
                else:
                    op_to_descriptor[o] = d
        return uniques(op_to_descriptor.values())

    @classmethod
    def _filter_unconsumed_ops(cls, descriptors, ops):
        filtered = ops[:]
        for d in descriptors:
            for o in d.child_ops:
                filtered.remove(o)
        return filtered

    def _resolve_descriptors_from_nodes(self, ops):
        """
        :type nodes: list(tf.Operations)
        :rtype: list(LayerDescriptor)
        """
        descriptors = []
        resolvers = self._create_layer_resolvers()

        constructor = TFGraphBuilder(ops)
        constructor.link_nodes()

        graph_matcher = GraphMatcher(constructor.nodes)
        for resolver in resolvers:
            resolved_descriptors = resolver.resolve_layer(graph_matcher, self._graph_helper)
            if len(resolved_descriptors) == 0:
                continue
            if resolver.is_final_resolution():
                ops_to_remove = [n for d in resolved_descriptors for n in d.child_ops]
                constructor = TFGraphBuilder([o for o in ops if o not in ops_to_remove])
                constructor.link_nodes()
                graph_matcher = GraphMatcher(constructor.nodes)
            descriptors.extend(resolved_descriptors)
        return descriptors

    @classmethod
    def _create_layer_resolvers(cls):
        return [resolver_class() for resolver_class in layers.layer_resolvers]

    def _create_layer(self, layer_builder, descriptor):
        """
        :type descriptor: converters.tensorflow.common.LayerDescriptor
        :rtype: None
        """
        self._logger.info(code_to_message.get_progress_message('INFO_ALL_BUILDING_LAYER_W_NODES')(
            descriptor.layer_type, [op.name for op in descriptor.child_ops]))

        inputs = self._topology_resolver.get_input_layers_for(descriptor)
        outputs = self._topology_resolver.get_output_layers_for(descriptor)
        layer_builder.build_layer(self._context, descriptor, inputs, outputs)

    @classmethod
    def _create_layer_builder(cls, descriptor):
        builder_class = layers.layer_builders.get(type(descriptor), None)
        if builder_class is None:
            raise ConverterError(code_to_message.get_message('ERROR_TF_NO_INPUT_TO_CREATE_LAYER')(type(descriptor)))
        return builder_class()

    def _set_model_version(self, model_version):
        """
        :type model_version:  str
        :rtype:
        """
        if model_version is not None:
            self._context.model.set_model_version(model_version[:64])

    @classmethod
    def _resolve_graph_operations_from_model(cls, model):
        """
        :type model: converters.tensorflow.loader.Model
        :rtype: list[tensorflow.Operation]
        """
        operations_map = dict()
        for op in model.session.graph.get_operations():
            operations_map[str(op.name)] = op

        input_ops = set()
        for i in model.inputs:
            input_ops.add(operations_map[i.name])

        all_ops_in_paths = set()
        for output_op_name in model.out_nodes_names:
            queue = [operations_map[output_op_name]]
            visited = set()
            while len(queue) > 0:
                head = queue.pop(0)
                if head in visited:
                    continue
                visited.add(head)

                if head in input_ops:
                    continue

                for t in head.inputs:
                    queue.append(t.op)

            all_ops_in_paths.update(visited)

        return list(all_ops_in_paths)
