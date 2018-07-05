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

from onnx_translations import *

#------------------------------------------------------------------------------
#   Clip
#------------------------------------------------------------------------------
class OnnxClipTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op,
                                    ('max','f'),
                                    ('min','f'))
        return op_adapter.NeuronOp(src_op.name,
                                   modeltools.NEURON_RELU_MIN_MAX,
                                   min_clamp=params.min,
                                   max_clamp=params.max)

    def axes_to_snpe_order(self, node, graph):
        eltwise_to_snpe_order(node, graph)

OnnxTranslations.register(OnnxClipTranslation(), onnx_type('Clip'))

#------------------------------------------------------------------------------
#   Concat
#------------------------------------------------------------------------------
class OnnxConcatTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op,
                                    ('axis','i'))
        return op_adapter.ConcatOp(src_op.name, params.axis)

    def infer_output_shapes(self, op, input_shapes):
        axis = op.axis
        output_shape = input_shapes[0][:]
        output_shape[axis] = sum(shape[axis] for shape in input_shapes)
        return [output_shape]

    def axes_to_snpe_order(self, node, graph):
        eltwise_to_snpe_order(node, graph)
        buf = graph.get_buffer(node.output_names[0])
        if buf.axis_format == AxisFormat.NSC:
            node.op.axis = NCS_TO_NSC[node.op.axis]

OnnxTranslations.register(OnnxConcatTranslation(),
                          onnx_type('Concat'),
                          op_adapter.ConcatOp.TRANSLATION_KEY)

#------------------------------------------------------------------------------
#   Flatten
#------------------------------------------------------------------------------
class OnnxFlattenTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, ('axis','i',1))
        axis = params.axis

        input_buf = graph.get_buffer(str(src_op.input[0]))
        input_shape = input_buf.shape

        pre_axes = input_shape[:axis]
        post_axes = input_shape[axis:]
        output_shape = [ product(pre_axes), product(post_axes) ]
        return op_adapter.ReshapeOp(src_op.name, output_shape)

    # NB the reshape translation handles everything besides parameter
    # extraction, because flatten is just a special case of reshape.
    def axes_to_snpe_order(self):
        raise NotImplemented()

OnnxTranslations.register(OnnxFlattenTranslation(), onnx_type('Flatten'))

#------------------------------------------------------------------------------
#   Reshape
#------------------------------------------------------------------------------
class OnnxReshapeTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, ('shape','li'))
        input_name = str(src_op.input[0])
        if graph.weights.has(input_name):
            # static reshape of weight parameters
            output_name = str(src_op.output[0])
            LOG_INFO(INFO_STATIC_RESHAPE,
                     input_name,
                     output_name,
                     params.shape)

            w = graph.weights.fetch(input_name).copy()
            w = numpy.reshape(w, params.shape)
            graph.weights.insert(output_name, w)
            return OnnxStaticOp(src_op.name)

        else:
            # dynamic reshape of activations
            input_buf = graph.get_buffer(input_name)
            input_shape = input_buf.shape

            remainder_size = product(input_shape)
            remainder_index = -1
            output_shape = []
            for i, s in enumerate(params.shape):
                if s == -1:
                    remainder_index = i
                    output_shape.append(0)
                elif s == 0:
                    remainder_size /= input_shape[i]
                    output_shape.append(input_shape[i])
                else:
                    remainder_size /= s
                    output_shape.append(s)
            if remainder_index >= 0:
                output_shape[remainder_index] = remainder_size

            return op_adapter.ReshapeOp(src_op.name, output_shape)

    def infer_output_shapes(self, op, input_shapes):
        return [op.output_shape]

    def axes_to_snpe_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_buffer(input_name)
        # force convergence if necessary
        # use the 'bacwkwards' permute orders because they are self-inverses.
        if input_buf.axis_format == AxisFormat.NSC:
            inject_implicit_permute(graph, input_name, AxisFormat.NCS, NCS_TO_NSC,[node.op.name])
        elif input_buf.axis_format == AxisFormat.BTF:
            inject_implicit_permute(graph, input_name, AxisFormat.TBF, TBF_TO_BTF,[node.op.name])
        elif input_buf.axis_format == AxisFormat.NONTRIVIAL:
            pass
        elif input_buf.axis_format == AxisFormat.FEATURE:
            pass
        else:
            raise ValueError(ERROR_RESHAPE_UNEXPECTED_INPUT_ORDER.format(input_buf.axis_format))
        output_buf = graph.get_output_buffers(node)[0]
        if output_buf.rank() >= 4:
            ASSERT(product(output_buf.shape[:-3]) == 1,
                   ERROR_RESHAPE_BATCH_UNSUPPORTED)
            output_buf.shape = output_buf.shape[-3:]
        output_buf.axis_format = AxisFormat.NONTRIVIAL

OnnxTranslations.register(OnnxReshapeTranslation(),
                          onnx_type('Reshape'),
                          op_adapter.ReshapeOp.TRANSLATION_KEY)

#------------------------------------------------------------------------------
#   Slice, Crop
#------------------------------------------------------------------------------
class OnnxSliceTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        input_buf = graph.get_buffer(str(src_op.input[0]))
        rank = len(input_buf.shape)
        default_axes = list(range(rank))
        params = extract_attributes(src_op,
                                    ('axes','li',default_axes),
                                    ('ends','li'),
                                    ('starts','li'))
        LOG_ASSERT(len(params.starts) == len(params.axes),
                   "Node %s: expected same number of starts as axes",
                   src_op.name)
        LOG_ASSERT(len(params.ends) == len(params.axes),
                   "Node %s: expected same number of ends as axes",
                   src_op.name)

        # canonicalize the axes
        offsets = [0]*rank
        output_shape = input_buf.shape[:]
        for i, axis in enumerate(params.axes):
            start = params.starts[i]
            end = parms.ends[i]
            dim = input_buf.shape[axis]
            # Negative values mean wrap around, like in python
            if start < 0:
                start %= dim
            if end < 0:
                end %= dim
            # higher than the size, however, means stop at the end.
            start = min(start, dim)
            end = min(end, dim)

            offsets[axis] = start
            output_shape[axis] = end-start

        return op_adapter.CropOp(src_op.name, offsets, output_shape)

    def infer_output_shapes(self, op, input_shapes):
        return [op.output_shape]

    def axes_to_snpe_order(self, node, graph):
        eltwise_to_snpe_order(node, graph)

# Onnx Crop should go here as well, but the documentation is really
# ambiguous so we won't add it until we see an example.
OnnxTranslations.register(OnnxSliceTranslation(),
                          onnx_type('Slice'),
                          op_adapter.CropOp.TRANSLATION_KEY)

#------------------------------------------------------------------------------
#   Split
#------------------------------------------------------------------------------
class OnnxSplitTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op,
                                    ('axis','i'),
                                    ('splits','li',[]))
        input_buf = graph.get_buffer(str(src_op.input[0]))
        if not params.splits:
            params.splits = [input_buf.shape[axis]/len(src_op.output)]

        slice_points = []
        next_slice_point = 0
        for split in params.splits[1:]:
            next_slice_point += split
            slice_points.append(next_slice_point)
        return op_adapter.SliceOp(src_op.name,
                                  axis=params.axis,
                                  slice_points=slice_points)

    def axes_to_snpe_order(self, node, graph):
        eltwise_to_snpe_order(node, graph)

OnnxTranslations.register(OnnxSplitTranslation(),
                          onnx_type('Split'),
                          op_adapter.SliceOp.TRANSLATION_KEY)

#------------------------------------------------------------------------------
#   Transpose
#------------------------------------------------------------------------------
class OnnxTransposeTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.index_method(REMOVE_NOOP, self.remove_noop)

    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op, ('perm','li'))

        return op_adapter.PermuteOp(src_op.name, params.perm)

    def infer_output_shapes(self, op, input_shapes):
        output_shape = [input_shapes[0][i] for i in op.order]
        return [output_shape]

    def axes_to_snpe_order(self, node, graph):
        input_buf = graph.get_input_buffers(node)[0]
        output_buf = graph.get_output_buffers(node)[0]
        # check for trivial cases first, which will end up
        # in removal. Otherwise, just set output order to nontrivial
        if input_buf.axis_format == AxisFormat.NSC:
            # special case: transforming to NSC, will become noop
            if node.op.order == [0,2,3,1]:
                node.op.order = [0,1,2]
                output_buf.shape = output_buf.shape[1:]
                output_buf.axis_format = AxisFormat.NSC
                return
            else:
                # going to nontrivial
                node.op.order.remove(0)
                node.op.order = map(lambda x: x-1, node.op.order)
                output_buf.shape = output_buf.shape[1:]
                output_buf.axis_format = AxisFormat.NONTRIVIAL
        elif input_buf.axis_format == AxisFormat.BTF:
            if node.op.order == [0,2,3,1]:
                node.op.order = [0,1,2]
                output_buf.shape = output_buf.shape[1:]
                output_buf.axis_format = AxisFormat.BTF
            else:
                node.op.order.remove(0)
                node.op.order = map(lambda x: x-1, node.op.order)
                output_buf.shape = output_buf.shape[1:]
                output_buf.axis_format = AxisFormat.NONTRIVIAL
        elif input_buf.axis_format == AxisFormat.NONTRIVIAL:
            if len(node.op.order) == 4:
                node.op.order.remove(0)
                node.op.order = map(lambda x: x-1, node.op.order)
                output_buf.shape = output_buf.shape[1:]
                output_buf.axis_format = AxisFormat.NONTRIVIAL
            elif len(node.op.order) > 3:
                raise ValueError(ERROR_PERMUTE_TOO_MANY_DIMENSIONS)
            else:
                # nothing to be done
                output_buf.axis_format = AxisFormat.NONTRIVIAL
        else:
            raise ValueError(ERROR_PERMUTE_UNEXPECTED_INPUT_ORDER.format(intput_buf.axis_format))

    def remove_noop(self, node, graph):
        input_buffer = graph.get_input_buffers(node)[0]
        output_buffer = graph.get_output_buffers(node)[0]
        if input_buffer.axis_format == output_buffer.axis_format and \
           node.op.order == range(len(node.op.order)):
            # this permute is trivial, remove it
            graph.squash(node, input_buffer.name)

OnnxTranslations.register(OnnxTransposeTranslation(),
                          onnx_type('Transpose'),
                          op_adapter.PermuteOp.TRANSLATION_KEY)
