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
from snpe import modeltools

class Op(object):
    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.attrs = {}

    def addattr(self, key, source, default):
        self.attrs[key] = source.get(key, default)

    def assertattr(self, key, source):
        if key in source:
            self.attrs[key] = source[key]
        else:
            raise KeyError("Op %s missing required argument %s" % (self.name, key))

    def __getitem__(self, key):
        return self.attrs[key]

    def __setitem__(self, key, value):
        self.attrs[key] = value

    def __getattr__(self, name):
        try:
            return self.attrs[name]
        except KeyError:
            raise KeyError("op %s has no attribute %s" % (self.name, name))

class InputOp(Op):
    TRANSLATION_KEY = 'input'
    def __init__(self, name, shape, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.shape = shape
        self.addattr('image_encoding_in', kargs, 'bgr')
        self.addattr('image_encoding_out', kargs, 'bgr')
        self.addattr('image_type', kargs, 'default')

class BatchnormOp(Op):
    TRANSLATION_KEY = 'batchnorm'
    def __init__(self, name, weights, bias, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.weights = weights
        self.bias = bias
        self.addattr('compute_statistics',kargs,False)
        self.addattr('use_mu_sigma',kargs,False)
        self.addattr('across_spatial',kargs,False)


class ConvolutionOp(Op):
    TRANSLATION_KEY = 'convolution'
    def __init__(self, name, weights, bias, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.weights = weights
        self.bias = bias
        self.assertattr('padx', kargs)
        self.assertattr('pady', kargs)
        self.addattr('padding_mode',kargs,modeltools.PADDING_ZERO)
        self.addattr('padding_size_strategy',kargs,modeltools.PADDING_SIZE_EXPLICIT)
        self.assertattr('stridex',kargs)
        self.assertattr('stridey',kargs)
        self.assertattr('dilationx',kargs)
        self.assertattr('dilationy',kargs)
        self.addattr('groups',kargs,1)


class ConcatOp(Op):
    TRANSLATION_KEY = 'concatenation'
    def __init__(self, name, axis):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.axis = axis


class CropOp(Op):
    TRANSLATION_KEY = 'crop'
    def __init__(self, name, offsets, output_shape):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.offsets = offsets
        self.output_shape = output_shape


class CrossCorrelationOp(Op):
    TRANSLATION_KEY = 'cross_correlation'
    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)

class DeconvolutionOp(Op):
    TRANSLATION_KEY = 'deconvolution'
    def __init__(self, name, weights, bias, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.weights = weights
        self.bias = bias
        self.addattr('stride', kargs, 1)
        self.addattr('padding', kargs, 0)
        self.addattr('padding_size_strategy', kargs, modeltools.PADDING_SIZE_EXPLICIT)
        self.assertattr('output_height', kargs)
        self.assertattr('output_width', kargs)
        self.addattr('groups', kargs, 1)

class DropoutOp(Op):
    TRANSLATION_KEY = 'dropout'
    def __init__(self, name, keep):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.keep = keep

class ElementwiseMaxOp(Op):
    TRANSLATION_KEY = 'elementwise_max'
    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)

class ElementwiseProductOp(Op):
    TRANSLATION_KEY = 'elementwise_product'
    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)

class ElementwiseSumOp(Op):
    TRANSLATION_KEY = 'elementwise_sum',
    def __init__(self, name, coeffs = []):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.coeffs = coeffs

class FullyConnectedOp(Op):
    TRANSLATION_KEY = 'fully_connected'
    def __init__(self, name, weights_list, bias):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.weights_list = weights_list
        self.bias = bias

class GenerateProposalsOp(Op):
    TRANSLATION_KEY = 'generate_proposals'
    def __init__(self, name, anchors, im_info, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('spatial_scale', kargs)
        self.assertattr('pre_nms_top_n', kargs)
        self.assertattr('post_nms_top_n', kargs)
        self.assertattr('nms_thresh', kargs)
        self.assertattr('min_size', kargs)
        self.addattr('correct_transform_coords', kargs, True)

class GruOp(Op):
    TRANSLATION_KEY = 'gru'
    def __init__(self, name, state_gate, forget_gate, control_gate, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.state_gate = state_gate
        self.forget_gate = forget_gate
        self.control_gate = control_gate
        self.addattr('activation', kargs, modeltools.NEURON_LOGISTIC)
        self.addattr('gate_activation', kargs, modeltools.NEURON_LOGISTIC)
        self.addattr('rec_gate_activation', kargs, modeltools.NEURON_TANH)
        self.addattr('backwards', kargs, False)

class LstmOp(Op):
    TRANSLATION_KEY = 'lstm'
    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('gate_weights', kargs)
        self.assertattr('gate_bias', kargs)
        self.assertattr('recurrent_weights', kargs)
        self.addattr('w_xc_static', kargs, None)
        self.addattr('backward', kargs, False)
        self.addattr('reset_state_at_time_step_0', kargs, False)

class MaxYOp(Op):
    TRANSLATION_KEY = 'max_y'
    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)

class NeuronOp(Op):
    TRANSLATION_KEY = 'neuron'
    def __init__(self, name, neuron_type, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.neuron_type = neuron_type
        self.addattr('a', kargs, 0.0)
        self.addattr('b', kargs, 0.0)
        self.addattr('min_clamp', kargs, 0.0)
        self.addattr('max_clamp', kargs, 0.0)

class PoolOp(Op):
    TRANSLATION_KEY = 'pool'
    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('pool_type', kargs)
        self.assertattr('size_x', kargs)
        self.assertattr('size_y', kargs)
        self.addattr('stride_x', kargs, 1)
        self.addattr('stride_y', kargs, 1)
        self.addattr('pad_x', kargs, 0)
        self.addattr('pad_y', kargs, 0)
        self.addattr('padding_size_strategy', kargs, modeltools.PADDING_SIZE_EXPLICIT)
        self.addattr('pool_region_include_padding', kargs, True)

class PermuteOp(Op):
    TRANSLATION_KEY = 'permute'
    def __init__(self, name, order):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.order = order

class PreluOp(Op):
    TRANSLATION_KEY = 'prelu'
    def __init__(self, name, coeff):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.coeff = coeff

class ProposalOp(Op):
    TRANSLATION_KEY = 'proposal'
    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('feat_stride', kargs)
        self.assertattr('scales', kargs)
        self.assertattr('ratios', kargs)
        self.assertattr('anchor_bas_size', kargs)
        self.assertattr('min_bbox_size', kargs)
        self.assertattr('max_num_proposals', kargs)
        self.assertattr('max_num_rois', kargs)
        self.assertattr('iou_threshold_nms', kargs)

class ReshapeOp(Op):
    TRANSLATION_KEY = 'reshape'
    def __init__(self, name, output_shape):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.output_shape = output_shape

class RNormOp(Op):
    TRANSLATION_KEY = 'rnorm'
    def __init__(self, name, size, alpha, beta, k, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.addattr('across_channels',kargs,True)

class RoiAlignOp(Op):
    TRANSLATION_KEY = 'roi_align'
    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('spatial_scale', kargs)
        self.assertattr('pooled_size_h', kargs)
        self.assertattr('pooled_size_w', kargs)
        self.assertattr('sampling_ratio', kargs)
        # implode batch parameters
        self.addattr('tiled_batch_h', kargs, -1)
        self.addattr('tiled_batch_w', kargs, -1)
        self.addattr('batch_pad_h', kargs, -1)
        self.addattr('batch_pad_w', kargs, -1)
        self.addattr('pad_value', kargs, 0.0)

class RoiPoolingOp(Op):
    TRANSLATION_KEY = 'roi_pooling'
    def __init__(self, name, output_shape, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('pooled_size_h', kargs)
        self.assertattr('pooled_size_w', kargs)
        self.assertattr('spatial_scale', kargs)
        self.output_shape = output_shape

class ResizeOp(Op):
    TRANSLATION_KEY = 'resize'
    def __init__(self, name, output_shape, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.output_shape = output_shape
        self.addattr('pad_value', kargs, 0.0)
        self.addattr('maintain_aspect_ratio', kargs, False)
        self.addattr('resize_mode', kargs, modeltools.RESIZE_BILINEAR)
        self.addattr('scaled_height', kargs, 0.0)
        self.addattr('scaled_width', kargs, 0.0)
        self.addattr('align_corners', kargs, False)

class RnnTransformationOp(Op):
    TRANSLATION_KEY = 'rnn_transformation'
    def __init__(self, name, weights, bias, activation):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.weights = weights
        self.bias = bias
        self.activation = activation

class SliceOp(Op):
    TRANSLATION_KEY = 'slice'
    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('axis', kargs)
        self.assertattr('slice_points', kargs)

class SoftmaxOp(Op):
    TRANSLATION_KEY = 'softmax'
    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)

class SubtractMeanOp(Op):
    TRANSLATION_KEY = 'subtract_mean'
    def __init__(self, name, mean_values):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.mean_values = mean_values

class UpsampleIndexBasedOp(Op):
    TRANSLATION_KEY = 'upsample_index_based'
    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('pool_size', kargs)
        self.addattr('pool_stride', kargs, 1)
        self.addattr('pad', kargs, 0)
        self.addattr('output_height', kargs, -1)
        self.addattr('output_width', kargs, -1)

class UpsampleSparseOp(Op):
    TRANSLATION_KEY = 'upsample_sparse'
    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('pool_size', kargs)
        self.addattr('pool_stride', kargs, 1)
        self.addattr('pad', kargs, 0)
        self.addattr('output_height', kargs, -1)
        self.addattr('output_width', kargs, -1)
