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
#   AveragePool, MaxPool
#------------------------------------------------------------------------------
class OnnxPoolTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op,
                                    ('auto_pad','s',''),
                                    ('kernel_shape','li'),
                                    ('pads','li',[0,0,0,0]),
                                    ('strides','li',[1,1]))


        ASSERT(pads_symmetric(params.pads) or pads_righthanded(params.pads),
               ERROR_ASYMMETRIC_PADS_VALUES)

        padding_size_strategy = extract_padding_mode(params.auto_pad, src_op.name)
        if pads_righthanded(params.pads):
            padding_size_strategy = modeltools.PADDING_SIZE_EXPLICIT_ASYMMETRIC
        if str(src_op.op_type) == 'AveragePool':
            pool_type = modeltools.POOL_AVG
        else:
            pool_type = modeltools.POOL_MAX

        return op_adapter.PoolOp(src_op.name,
                                 pool_type=pool_type,
                                 size_y=params.kernel_shape[0],
                                 size_x=params.kernel_shape[1],
                                 stride_y=params.strides[0],
                                 stride_x=params.strides[1],
                                 pad_y=params.pads[2],
                                 pad_x=params.pads[3],
                                 padding_size_strategy=padding_size_strategy,
                                 pool_region_include_padding=False)

    def infer_output_shapes(self, op, input_shapes):
        input_shape = input_shapes[0]
        input_height = input_shape[2]
        input_width = input_shape[3]
        output_height = modeltools.calc_pool_output_dim(input_height,
                                                        op.size_y,
                                                        op.pad_y,
                                                        op.stride_y,
                                                        op.padding_size_strategy)
        output_width = modeltools.calc_pool_output_dim(input_width,
                                                       op.size_x,
                                                       op.pad_x,
                                                       op.stride_x,
                                                       op.padding_size_strategy)
        output_shape = input_shape[0:2] + [output_height, output_width]
        LOG_DEBUG(DEBUG_INFERRED_SHAPE, op.name, output_shape)
        return [output_shape]

    def axes_to_snpe_order(self, node, graph):
        log_axes_to_snpe_order(node, graph)
        image_to_snpe_order(node, graph)


OnnxTranslations.register(OnnxPoolTranslation(),
                          onnx_type('AveragePool'),
                          onnx_type('MaxPool'),
                          op_adapter.PoolOp.TRANSLATION_KEY)

#------------------------------------------------------------------------------
#   BatchNormalization
#------------------------------------------------------------------------------
class OnnxBatchNormalizationTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op,
                                    ('epsilon','f',1e-5),
                                    ('is_test','i',0),
                                    ('spatial','i',1))
        ASSERT(params.is_test, ERROR_BATCHNORM_TEST_ONLY)
        input_names = list(src_op.input)
        gamma, beta, mu, var = graph.weights.fetch(*input_names[1:])
        # y = gamma*( (x-mu)/sqrt(var+epsilon) ) + beta
        # weights = gamma/sqrt(var+epsilon)
        weights = gamma/numpy.sqrt(var+params.epsilon)
        # bias = -mu*gamma/sqrt(var+epsilon) + beta = -mu*weights + beta
        bias = -mu*weights + beta

        return op_adapter.BatchnormOp(src_op.name,
                                      weights,
                                      bias,
                                      across_spatial=bool(params.spatial))

    def extract_input_names(self, src_op):
        return [src_op.input[0]]

    def axes_to_snpe_order(self, node, graph):
        image_to_snpe_order(node, graph)

OnnxTranslations.register(OnnxBatchNormalizationTranslation(),
                          onnx_type('BatchNormalization'),
                          op_adapter.BatchnormOp.TRANSLATION_KEY)

#------------------------------------------------------------------------------
#   Conv
#------------------------------------------------------------------------------
class OnnxConvTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        input_names = map(str, src_op.input)

        weights = graph.weights.fetch(input_names[1])

        if len(input_names) > 2:
            bias = graph.weights.fetch(input_names[2])
        else:
            input_buf = graph.get_buffer(input_names[0])
            bias = numpy.zeros(weights.shape[0], dtype=numpy.float32)

        params = extract_attributes(src_op,
                                    ('auto_pad','s',''),
                                    ('dilations','li',[1,1]),
                                    ('group','i',1),
                                    ('kernel_shape','li',[]),
                                    ('pads','li',[0,0,0,0]),
                                    ('strides','li',[1,1]))

        ASSERT(pads_symmetric(params.pads),ERROR_ASYMMETRIC_PADS_VALUES)


        if params.kernel_shape:
            ASSERT(tuple(params.kernel_shape) == weights.shape[2:],
                   ERROR_KERNEL_SHAPE_DIFFERS_FROM_WEIGHTS)

        padding_mode = extract_padding_mode(params.auto_pad, src_op.name)

        return op_adapter.ConvolutionOp(src_op.name,
                                        weights,
                                        bias,
                                        padx=params.pads[1],
                                        pady=params.pads[0],
                                        padding_size_strategy=padding_mode,
                                        stridex=params.strides[1],
                                        stridey=params.strides[0],
                                        dilationx=params.dilations[1],
                                        dilationy=params.dilations[0],
                                        groups=params.group)

    def extract_input_names(self, src_op):
        return [src_op.input[0]]

    def infer_output_shapes(self, op, input_shapes):
        input_height = input_shapes[0][2]
        input_width = input_shapes[0][3]
        output_height = modeltools.calc_conv_output_dim(input_height,
                                                        op.weights.shape[2],
                                                        op.pady,
                                                        op.stridey,
                                                        op.dilationy,
                                                        op.padding_size_strategy)
        output_width = modeltools.calc_conv_output_dim(input_width,
                                                       op.weights.shape[3],
                                                       op.padx,
                                                       op.stridex,
                                                       op.dilationx,
                                                       op.padding_size_strategy)
        output_depth = op.bias.shape[0]
        batch = input_shapes[0][0]
        output_shape = [batch, output_depth, output_height, output_width]
        LOG_DEBUG(DEBUG_INFERRED_SHAPE, op.name, output_shape)
        return [output_shape]

    def axes_to_snpe_order(self, node, graph):
        log_axes_to_snpe_order(node, graph)
        image_to_snpe_order(node, graph)
        # weight order for ONNX is NCHW, want HWCN
        weights = numpy.transpose(node.op.weights, (2,3,1,0))
        node.op.weights = numpy.ascontiguousarray(weights, dtype=numpy.float32)

OnnxTranslations.register(OnnxConvTranslation(),
                          onnx_type('Conv'),
                          op_adapter.ConvolutionOp.TRANSLATION_KEY)

#------------------------------------------------------------------------------
#   ConvTranspose
#------------------------------------------------------------------------------
class OnnxConvTransposeTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        input_names = map(str, src_op.input)
        weights = graph.weights.fetch(input_names[1])
        if len(input_names) > 2:
            bias = graph.weights.fetch(input_names[2])
        else:
            input_buf = graph.get_buffer(input_names[0])
            bias = numpy.zeros(weights.shape[0], dtype=numpy.float32)

        params = extract_attributes(src_op,
                                    ('auto_pad','s',''),
                                    ('dilations','li',[1,1]),
                                    ('group','i',1),
                                    ('kernel_shape','li',[]),
                                    ('output_padding','li',[]),
                                    ('output_shape','li',[0,0]),
                                    ('pads','li',[0,0,0,0]),
                                    ('strides','li',[1,1,]))

        padding_mode = extract_padding_mode(params.auto_pad, src_op.name)
        LOG_ASSERT(pads_symmetric(params.pads), ERROR_ASYMMETRIC_PADS_VALUES)
        if params.kernel_shape:
            LOG_ASSERT(tuple(params.kernel_shape) == weights.shape[2:],
                       ERROR_KERNEL_SHAPE_DIFFERS_FROM_WEIGHTS)

        LOG_ASSERT(params.strides[0] == params.strides[1],
                   ERROR_DECONV_RECTANGULAR_STRIDE_UNSUPPORTED)

        op = op_adapter.DeconvolutionOp(src_op.name,
                                        weights,
                                        bias,
                                        stride=params.strides[0],
                                        padding=params.pads[0],
                                        padding_size_strategy=padding_mode,
                                        output_height=params.output_shape[0],
                                        output_width=params.output_shape[1],
                                        groups=params.group)

        ASSERT(not params.output_padding,
               ERROR_DECONV_OUTPUT_PADDING_UNSUPPORTED)
        return op

    def extract_input_names(self, src_op):
        return [src_op.input[0]]

    def infer_output_shape(self, op, input_shapes):
        if op.output_height == 0:
            # calculate according to provided formula
            input_shape = input_shapes[0]
            input_height = input_shape[2]
            input_width = input_shape[3]
            def calc_output_dim(input_size,
                                filter_size,
                                stride,
                                pad):
                return stride*(input_size-1) + filter_size - 2*pad # + output_pad

            output_height = calc_output_dim(input_height,
                                            op.weights.shape[2],
                                            op.stride,
                                            op.padding)
            op['output_height'] = output_height

            output_width = calc_output_dim(input_width,
                                           op.weights.shape[3],
                                           op.stride,
                                           op.padding)
            op['output_width'] = output_width
        else:
            output_height = op.output_height
            output_width = op.output_width


        return [ input_shape[0:2]  + [output_height, output_width] ]

    def axes_to_snpe_order(self, node, graph):
        image_to_snpe_order(node, graph)
        # weights are in CNHW, want HWCN
        weights = numpy.transpose(node.op.weights, (2,3,0,1))
        node.op.weights = numpy.ascontiguousarray(weights, dtype=numpy.float32)

OnnxTranslations.register(OnnxConvTransposeTranslation(),
                          onnx_type('ConvTranspose'),
                          op_adapter.DeconvolutionOp.TRANSLATION_KEY)

#------------------------------------------------------------------------------
#   FC
#------------------------------------------------------------------------------
class OnnxFCTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op,
                                    ('axis','i',1),
                                    ('axis_w','i',1))
        LOG_ASSERT(params.axis == 1, ERROR_FC_AXIS_UNSUPPORTED)
        LOG_ASSERT(params.axis == 1, ERROR_FC_AXIS_W_UNSUPPORTED)

        input_names = get_input_names(src_op)
        weights, bias = graph.weights.fetch(*input_names[1:3])
        return op_adpater.FullyConnectedOp(src_op.name, [weights], bias)

    def extract_input_names(self, src_op):
        return [str(src_op.input[0])]

    def infer_output_shapes(self, op, input_shapes):
        N = op.weights_list[0].shape[1]
        M = input_shapes[0][0]
        return [ [M,N] ]

    def axes_to_snpe_order(self, node, graph):
        log_axes_to_snpe_order(node, graph)
        input_buf = graph.get_input_buffers(node)[0]
        if input_buf.rank() == 3:
            enforce_input_format(node, graph, input_buf.name, AxisFormat.NSC, NCS_TO_NSC)
            # weights expect NCHW order, need to permute
            input_buf = graph.get_input_buffers(node)[0]
            height, width, depth = input_buf.shape
            weights = node.op.weights_list[0]
            # ONNX defines FC as W^Tx + b,
            # so the weights have shape (input_size, output_size)
            input_size = weights.shape[0]
            output_size = weights.shape[1]
            ASSERT(input_size == depth*height*width,
                   ERROR_FC_WRONG_INPUT_SIZE,
                   node.op.name,
                   input_size,
                   (depth,height,width))


            weights.shape = (depth, height, width, output_size)
            weights = numpy.transpose(weights, (3, 1, 2, 0))
            weights = numpy.ascontiguousarray(weights, dtype=numpy.float32)
            weights.shape = (output_size, input_size)
            node.op.weights_list[0] = weights
        elif input_buf.rank() == 1:
            # again, need to transpose weights for snpe order
            weights = node.op.weights_list[0]
            weights = numpy.ascontiguousarray(numpy.transpose(weights, (1,0)))
            node.op.weights_list[0] = weights

        output_buf = graph.get_output_buffers(node)[0]
        output_buf.shape = output_buf.shape[1:]
        output_buf.axis_format = AxisFormat.FEATURE



OnnxTranslations.register(OnnxFCTranslation(),
                          onnx_type('FC'),
                          op_adapter.FullyConnectedOp.TRANSLATION_KEY)

#------------------------------------------------------------------------------
#   GlobalAveragePool, GlobalMaxPool
#------------------------------------------------------------------------------
class OnnxGlobalPoolTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        input_buf = graph.get_buffer(str(src_op.input[0]))

        if str(src_op.op_type) == 'GlobalAveragePool':
            pool_type = modeltools.POOL_AVG
        else:
            pool_type = modeltools.POOL_MAX

        return op_adapter.PoolOp(src_op.name,
                                 pool_type=pool_type,
                                 size_x=input_buf.shape[3],
                                 size_y=input_buf.shape[2],
                                 stride_x=input_buf.shape[3],
                                 stride_y=input_buf.shape[2])

    # remainder of operations handled by OnnxPoolTranslation
    def axes_to_snpe_order(self):
        raise NotImplemented()

OnnxTranslations.register(OnnxGlobalPoolTranslation(),
                          onnx_type('GlobalAveragePool'),
                          onnx_type('GlobalMaxPool'))

#------------------------------------------------------------------------------
#   InstanceNormalization
#------------------------------------------------------------------------------
class OnnxInstanceNormalizationTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        input_names = map(str, src_op.input)
        weights, bias = graph.weights.fetch(*input_names[1:])
        return op_adapter.BatchnormOp(src_op.name,
                                      weights,
                                      bias,
                                      compute_statistics=True,
                                      use_mu_sigma=True,
                                      across_spatial=True)
    # rest is handled by OnnxBatchNormalizationTranslation

#------------------------------------------------------------------------------
#   MaxRoiPool
#------------------------------------------------------------------------------
class OnnxMaxRoiPoolTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op,
                                    ('spatial_scale','f',1.0),
                                    ('pooled_shape','li'))

        input_names = map(str, src_op.input)
        input_buf = graph.get_buffer(input_names[0])
        roi_buf = graph.get_buffer(input_names[1])
        output_shape = [ roi_buf.shape[0],
                         input_buf.shape[1],
                         params.pooled_shape[0],
                         params.pooled_shape[1] ]

        return op_adapter.RoiPoolingOp(src_op.name,
                                       output_shape,
                                       pooled_size_h=params.pooled_shape[0],
                                       pooled_size_w=params.pooled_shape[1],
                                       spatial_scale=params.spatial_scale)

    def infer_output_shapes(self, op, input_shapes):
        return [op.output_shape]

    def axes_to_snpe_order(self, node, graph):
        enforce_input_format(node, graph, node.input_names[0], AxisFormat.NSC, NCS_TO_NSC)
        output_buf = graph.get_output_buffers(node)[0]
        ASSERT(output_buf.shape[0] == 1,
               ERROR_MAX_ROI_POOL_BATCH_UNSUPPORTED)
        output_buf.shape = permute_shape(output_buf.shape[1:], NCS_TO_NSC)
        output_buf.axis_format = AxisFormat.NSC

OnnxTranslations.register(OnnxMaxRoiPoolTranslation(),
                          onnx_type('MaxRoiPool'),
                          op_adapter.RoiPoolingOp.TRANSLATION_KEY)

#------------------------------------------------------------------------------
#   Prelu, LeakyRelu
#------------------------------------------------------------------------------
# Also handles LeakyRelu as a bonus.
class OnnxPreluTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        input_names = map(str, src_op.input)
        input_buf = graph.get_buffer(input_names[0])

        if str(src_op.type) == 'LeakyRelu':
            params = extract_attribtutes(src_op, ('alpha','f',0.01))
            bias = numpy.ones(input_buf.shape[1], dtype=numpy.float32)
            bias *= params.alpha
        else:
            slope = graph.weights.fetch(input_names[1])
            if len(slope) == 1:
                bias = numpy.ones(input_buf.shape[1], dtype=numpy.float32)
                bias *= slope[0]
            else:
                bias = numpy.require(slope, dtype=numpy.float32)

        return op_adapter.PreluOp(src_op.name, bias)

    def extract_input_names(self, src_op, graph):
        return [src_op.input[0]]

    def axes_to_snpe_order(self, node, graph):
        eltwise_to_snpe_order(node, graph)

OnnxTranslations.register(OnnxPreluTranslation(),
                          onnx_type('Prelu'),
                          onnx_type('LeakyRelu'),
                          op_adapter.PreluOp.TRANSLATION_KEY)

#------------------------------------------------------------------------------
#   Lrn
#------------------------------------------------------------------------------
class OnnxLrnTranslation(OnnxTranslationBase):
    def extract_parameters(self, src_op, graph):
        params = extract_attributes(src_op,
                                    ('alpha','f'),
                                    ('beta','f'),
                                    ('bias','f',1.0),
                                    ('size','i'))

        return op_adapter.RNormOp(src_op.name,
                                  params.size,
                                  params.alpha/params.size,
                                  params.beta,
                                  params.bias,
                                  across_channels=True)

    def axes_to_snpe_order(self, node, graph):
        image_to_snpe_order(node, graph)

OnnxTranslations.register(OnnxLrnTranslation(),
                          onnx_type('LRN'),
                          op_adapter.RNormOp.TRANSLATION_KEY)

