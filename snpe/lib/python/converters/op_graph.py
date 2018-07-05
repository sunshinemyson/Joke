#==============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright 2018 Qualcomm Technologies, Inc. All rights reserved.
#  Confidential & Proprietary - Qualcomm Technologies, Inc. ("QTI")
#
#  The party receiving this software directly from QTI (the "Recipient")
#  may use this software as reasonably necessary solely for the purposes
#  set forth in the agreement between the Recipient and QTI (the
#  "Agreement"). The software may be used in source code form solely by
#  the Recipient's employees (if any) authorized by the Agreement. Unless
#  expressly authorized in the Agreement, the Recipient may not sublicense,
#  assign, transfer or otherwise provide the source code to any third
#  party. Qualcomm Technologies, Inc. retains all ownership rights in and
#  to the software
#
#  This notice supersedes any other QTI notices contained within the software
#  except copyright notices indicating different years of publication for
#  different portions of the software. This notice does not supersede the
#  application of any third party copyright notice to that third party's
#  code.
#
#  @@-COPYRIGHT-END-@@
#==============================================================================
import op_adapter
from snpe.common.snpe_axis_transformer import AxisAnnotation

class OpNode(object):
    def __init__(self, op, input_names, output_names):
        self.op = op
        self.input_names = input_names
        self.output_names = output_names

class AxisFormat(object):
    # Batch,Channel,Spatial. With one batch and two spatial dimensions,
    # equivalent to NCHW
    NCS = 'NCS'
    # Batch,Spatial,Channel. With one batch and two spatial dimensions,
    # equivalent to NHWC. This is the native data order for SNPE ops which
    # output feature maps.
    NSC = 'NSC'
    # Time,Batch,Feature.
    TBF = 'TBF'
    # Batch,Time,Feature. This is the native data order for SNPE RNN ops.
    BTF = 'BTF'
    # Batch,Feature.
    FEATURE = 'FEATURE'
    # Op specific data format.
    NONTRIVIAL = 'NONTRIVIAL'
    # Enum value used by buffers which have not yet undergone axis tracking.
    NOT_YET_DEFINED = 'NOT_YET_DEFINED'

    @classmethod
    def get_permute_order(cls, src_order, target_order, rank):
        if src_order == cls.NCS:
            if target_order == cls.NSC:
                num_spatial = rank-2
                return [0] + [ i+2 for i in xrange(num_spatial) ] + [1]
        elif src_order == cls.NSC:
            if target_order == cls.NCS:
                num_spatial = rank-2
                return [0, rank-1] + [ i+1 for i in xrange(num_spatial)]
        elif src_order == cls.TBF:
            if target_order == cls.BTF:
                return [1,0,2]
        elif src_order == cls.BTF:
            if target_order == cls.TBF:
                return [1,0,2]
        else:
            raise ValueError("No permutation from %s to %s" % (src_order, target_order))

class Buffer(object):
    def __init__(self, name, shape, producer):
        self.name = name
        self.producer = producer
        self.consumers = set()
        self.shape = shape
        self.axis_format = AxisFormat.NOT_YET_DEFINED

    def rank(self):
        return len(self.shape)

    def get_axis_order(self):
        """Translate AxisFormat enum to modeltools axis order list"""
        if self.axis_format == 'NSC':
            return [AxisAnnotation.CHANNEL, AxisAnnotation.HEIGHT, AxisAnnotation.WIDTH]
        elif self.axis_format == 'FEATURE':
            return [AxisAnnotation.FEATURE]
        elif self.axis_format == 'BTF':
            return [AxisAnnotation.BATCH, AxisAnnotation.TIME, AxisAnnotation.FEATURE]
        else:
            raise ValueError("Encountered unexpected axis format for get_axis_order: %s" % self.axis_format)

class OpGraph(object):
    def __init__(self, naming_policy, shape_inference_policy):
        self.naming_policy = naming_policy
        self.shape_inference_policy = shape_inference_policy

        self.nodes_by_name = {}
        self.nodes_in_order = []
        self.buffers = {}

    def __insert_node(self, node, output_shapes, idx=-1):
        """Insert a node into the graph's internal data structures.

        node: Node to be inserted
        output_shapes: shapes of the node's output buffers, which must be created.
        idx: index in nodes_in_order at which to insert. By default, appends to
             the list."""
        for name, shape in zip(node.output_names, output_shapes):
            self.buffers[name] = Buffer(name, shape, node)

        for name in node.input_names:
            self.buffers[name].consumers.add(node)

        self.nodes_by_name[node.op.name] = node
        if idx == -1:
            self.nodes_in_order.append(node)
        else:
            self.nodes_in_order.insert(idx, node)


    def add(self, op, input_names, output_names):
        op.name = self.naming_policy.get_op_name(op)

        if not isinstance(input_names, list):
            input_names = [input_names]
        input_names = self.naming_policy.get_input_names(op, input_names)

        input_shapes = []
        for name in input_names:
            if not name in self.buffers:
                raise KeyError("Graph has no buffer %s, referred to as input for %s" % (name, op.name))
            input_shapes.append(self.buffers[name].shape)



        if not isinstance(output_names, list):
            output_names = [output_names]
        output_names = self.naming_policy.get_output_names(op, output_names)

        node = OpNode(op, input_names, output_names)

        output_shapes = self.shape_inference_policy.infer_shape(op, input_shapes)
        if len(output_shapes) != len(output_names):
            raise ValueError("Op %s: produced %d output shapes, but have %d outputs" % (op.name, len(output_shapes), len(output_names)))

        # at this point everything should be error free, so it's fine to actually
        # touch the data structures
        self.__insert_node(node, output_shapes)

    def add_input(self, name, shape, encoding, input_type):
        op = op_adapter.InputOp(name, shape,
                                image_encoding_in=encoding,
                                image_encoding_out=encoding,
                                image_type=input_type)
        output_names = self.naming_policy.get_output_names(op, [name])

        node = OpNode(op, [], output_names)
        self.__insert_node(node, [shape])

    def inject(self, op, input_name, output_name, consumer_names=None):
        op.name = self.naming_policy.get_op_name(op)
        if not input_name in self.buffers:
            raise KeyError("Cannot inject op %s onto nonexistent buffer %s" % (op.name, input_name))

        input_buffer = self.buffers[input_name]
        if consumer_names is None:
            old_consumers = list(input_buffer.consumers)
            input_buffer.consumers.clear();
        else:
            old_consumers = []
            for name in consumer_names:
                if not name in self.nodes_by_name:
                    raise KeyError("Cannot inject op %s with nonexistent consumer %s" % (op.name, name))
                consumer = self.nodes_by_name[name]
                if not consumer in input_buffer.consumers:
                    raise KeyError("Cannot inject op %s, specified consumer %s does not actually consume input buffer %s" % (op.name, name, input_name))

                old_consumers.append(consumer)
                input_buffer.consumers.remove(consumer)

        output_name = self.naming_policy.get_output_names(op, [output_name])[0]
        producer_idx = self.nodes_in_order.index(input_buffer.producer)
        output_shapes = self.shape_inference_policy.infer_shape(op, [input_buffer.shape])
        node = OpNode(op, [input_name], [output_name])
        self.__insert_node(node, output_shapes, producer_idx+1)

        output_buffer = self.buffers[output_name]
        for consumer in old_consumers:
            output_buffer.consumers.add(consumer)
            for i, name in enumerate(consumer.input_names):
                if name == input_name:
                    consumer.input_names[i] = output_name

    def prune(self, node):
        """Remove a node and its output buffers from the graph completely.
        Will raise an exception if the node has any successors."""

        output_buffers = self.get_output_buffers(node)
        consumers = []
        for buf in output_buffers:
            consumers.extend(buf.consumers)
        consumers = [c.op.name for c in consumers]
        if len(consumers) > 0:
            raise RuntimeError("Cannot prune node %s, which has the following successors: %s" % (node.op.name, consumers))

        for buf in output_buffers:
            del self.buffers[buf.name]
        for buf in self.get_input_buffers(node):
            buf.consumers.remove(node)
        del self.nodes_by_name[node.op.name]
        self.nodes_in_order.remove(node)

    def squash(self, node, input_name):
        # remove the input buffer, causing that buffer's
        # producer to producer the output buffer instead.
        if not input_name in self.buffers:
            raise KeyError("Cannot squash node %s onto non-existent input buffer %s" % (node.op.name, input_name))
        input_buffer = self.buffers[input_name]
        output_buffer = self.buffers[node.output_names[0]]

        if len(input_buffer.consumers) > 1:
            raise ValueError("Cannot squash node %s onto input buffer %s, which has more than one consumer" % (node.op.name, input_name))
        if not node in input_buffer.consumers:
            raise ValueError("Cannot squash node %s onto input buffer %s that it doesn't consumer" % (node.op.name, input_name))


        prev = input_buffer.producer
        output_idx = prev.output_names.index(input_name)
        prev.output_names[output_idx] = output_buffer.name
        output_buffer.producer = prev

        del self.buffers[input_name]
        del self.nodes_by_name[node.op.name]
        self.nodes_in_order.remove(node)

    def get_input_buffers(self, node):
        return [self.buffers[name] for name in node.input_names]

    def get_output_buffers(self, node):
        return [self.buffers[name] for name in node.output_names]

    def get_buffer(self, buffer_name):
        return self.buffers[buffer_name]

    def list_nodes(self):
        return self.nodes_in_order[:]

    def list_buffers(self):
        return list(self.buffers.values())
