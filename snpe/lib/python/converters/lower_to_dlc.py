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

import translation
import op_adapter
from snpe import modeltools

#------------------------------------------------------------------------------
#   Module Level Functions
#------------------------------------------------------------------------------
def lower(graph):
    model = modeltools.Model()
    DlcTranslations.apply_total(LOWER_TO_DLC, graph, model)
    for buf in graph.list_buffers():
        model.set_buffer_axis_order(buf.name, buf.get_axis_order())
    return model

#------------------------------------------------------------------------------
#   Translations
#------------------------------------------------------------------------------
DlcTranslations = translation.TranslationBank()
LOWER_TO_DLC = 'lower_to_dlc'

def register(dlc_translation):
    DlcTranslations.register(dlc_translation(), dlc_translation.TARGET)
    return dlc_translation

class DlcTranslationBase(translation.Translation):
    def __init__(self):
        translation.Translation.__init__(self)
        self.index_method(LOWER_TO_DLC, self.lower)

@register
class DlcInputTranslation(DlcTranslationBase):
    TARGET = op_adapter.InputOp.TRANSLATION_KEY
    def lower(self, node, graph, model):
        model.add_data_layer(node.op.name,
                             node.op.shape,
                             node.op.image_encoding_in,
                             node.op.image_encoding_out,
                             node.op.image_type)

@register
class DlcBatchnormTranslation(DlcTranslationBase):
    TARGET = op_adapter.BatchnormOp.TRANSLATION_KEY
    def lower(self, node, graph, model):
        model.add_batchnorm_layer(node.op.name,
                                  node.op.weights,
                                  node.op.bias,
                                  node.op.compute_statistics,
                                  node.op.use_mu_sigma,
                                  node.op.across_spatial,
                                  node.input_names[0],
                                  node.output_names[0])
@register
class DlcConvolutionTranslation(DlcTranslationBase):
    TARGET = op_adapter.ConvolutionOp.TRANSLATION_KEY
    def lower(self, node, graph, model):
        model.add_conv_layer(node.op.name,
                             node.op.weights,
                             node.op.bias,
                             node.op.padx,
                             node.op.pady,
                             node.op.padding_mode,
                             node.op.padding_size_strategy,
                             node.op.stridex,
                             node.op.stridey,
                             node.op.dilationx,
                             node.op.dilationy,
                             node.input_names[0],
                             node.output_names[0],
                             node.op.groups)
@register
class DlcConcatTranslation(DlcTranslationBase):
    TARGET = op_adapter.ConcatOp.TRANSLATION_KEY
    def lower(self, node, graph, model):
        model.add_concatenation_layer(node.op.name,
                                      node.input_names,
                                      node.output_names[0],
                                      node.op.axis)
@register
class DlcCropTranslation(DlcTranslationBase):
    TARGET = op_adapter.CropOp.TRANSLATION_KEY
    def lower(self, node, graph, model):
        model.add_crop_layer(node.op.name,
                             node.op.offsets,
                             node.op.output_dim,
                             node.input_names[0],
                             node.output_names[0])
@register
class DlcCrossCorrelationTranslation(DlcTranslationBase):
    TARGET = op_adapter.CrossCorrelationOp.TRANSLATION_KEY
    def lower(self, node, graph, model):
        model.add_cross_correlation_layer(node.op.name,
                                          node.input_names[0],
                                          node.input_names[1],
                                          node.output_names[0])
@register
class DlcDeconvolutionTranslation(DlcTranslationBase):
    TARGET = op_adapter.DeconvolutionOp.TRANSLATION_KEY
    def lower(self, node, graph, model):
        model.add_deconvolution_layer(node.op.name,
                                      node.op.weights,
                                      node.op.bias,
                                      node.op.stride,
                                      node.op.padding_size_strategy,
                                      node.op.padding,
                                      node.input_names[0],
                                      node.output_names[0],
                                      node.op.output_width,
                                      node.op.output_height,
                                      node.op.groups)
@register
class DlcDropoutTranslation(DlcTranslationBase):
    TARGET = op_adapter.DropoutOp.TRANSLATION_KEY
    def lower(self, node, graph, model):
        model.add_dropout_layer(node.op.name,
                                node.op.keep,
                                node.input_names[0],
                                node.output_names[0])
@register
class DlcElementwiseMaxTranslation(DlcTranslationBase):
    TARGET = op_adapter.ElementwiseMaxOp.TRANSLATION_KEY
    def lower(self, node, graph, model):
        model.add_elementwise_max_layer(node.op.name,
                                        node.input_names,
                                        node.output_names[0])
@register
class DlcElementwiseProductTranslation(DlcTranslationBase):
    TARGET = op_adapter.ElementwiseProductOp.TRANSLATION_KEY
    def lower(self, node, graph, model):
        model.add_elementwise_product_layer(node.op.name,
                                            node.input_names,
                                            node.output_names[0])
@register
class DlcElementwiseSumTranslation(DlcTranslationBase):
    TARGET = op_adapter.ElementwiseSumOp.TRANSLATION_KEY
    def lower(self, node, graph, model):
        coeffs = node.op.coeffs[:]
        num_missing_coeffs = len(node.input_names) - len(coeffs)
        if num_missing_coeffs > 0:
            coeffs.extend( [1.0]*num_missing_coeffs )
        model.add_elementwise_sum_layer(node.op.name,
                                        coeffs,
                                        node.input_names,
                                        node.output_names[0])
@register
class DlcFullyConnectedTranslation(DlcTranslationBase):
    TARGET = op_adapter.FullyConnectedOp.TRANSLATION_KEY
    def lower(self, node, graph, model):
        model.add_fc_layer(node.op.name,
                           node.op.weights_list,
                           node.op.bias,
                           node.input_names,
                           node.output_names[0])
@register
class DlcGenerateProposalsOp(DlcTranslationBase):
    TARGET = op_adapter.GenerateProposalsOp.TRANSLATION_KEY
    def lower(self, node, graph, model):
        model.add_generate_proposals_layer(node.op.name,
                                           node.op.spatial_scale,
                                           node.op.pre_nms_top_n,
                                           node.op.post_nms_top_n,
                                           node.op.nms_thresh,
                                           node.op.min_size,
                                           node.op.correct_transform_coords,
                                           node.op.anchors,
                                           node.op.im_info,
                                           node.input_names[0],
                                           node.input_names[1],
                                           node.output_names[0],
                                           node.ouput_names[1])
@register
class DlcGruTranslation(DlcTranslationBase):
    TARGET = op_adapter.GruOp.TRANSLATION_KEY
    def lower(self, node, graph, model):
        model.add_gru_layer(node.op.name,
                            node.op.state_gate,
                            node.op.forget_gate,
                            node.op.control_gate,
                            node.op.activation,
                            node.op.gate_activation,
                            node.op.rec_gate_activation,
                            node.op.backwards,
                            node.input_names[0],
                            node.output_names[0])
@register
class DlcLstmTranslation(DlcTranslationBase):
    TARGET = op_adapter.LstmOp.TRANSLATION_KEY
    def lower(self, node, graph, model):
        input_name = node.input_names[0]
        if len(node.input_names) > 1:
            sequence_continuation_name = node.input_names[1]
        else:
            sequence_continuation_name = ''
        if len(node.input_names) > 3:
            c_0_input_name = node.input_names[-2]
            h_0_input_name = node.input_names[-1]
        else:
            c_0_input_name = ''
            h_0_input_name = ''
        if len(node.input_names) in (3,5):
            x_static_name = node.input_names[2]
        else:
            x_static_name = ''
        model.add_lstm_layer(node.op.name,
                             node.op.gate_weights,
                             node.op.gate_bias,
                             node.op.recurrent_weights,
                             node.op.w_xc_static,
                             node.op.backward,
                             node.op.reset_state_at_time_step_0,
                             input_name,
                             sequence_continuation_name,
                             x_static_input_name,
                             c_0_input_name,
                             h_0_input_name,
                             node.output_names)
@register
class DlcMaxYTranslation(DlcTranslationBase):
    TARGET = op_adapter.MaxYOp.TRANSLATION_KEY
    def lower(self, node, graph, model):
        model.add_max_y_layer(node.op.name,
                              node.input_names[0],
                              node.output_names[0])
@register
class DlcNeuronTranslation(DlcTranslationBase):
    TARGET = op_adapter.NeuronOp.TRANSLATION_KEY
    def lower(self, node, graph, model):
        model.add_neuron_layer(node.op.name,
                               node.op.neuron_type,
                               node.input_names[0],
                               node.output_names[0],
                               node.op.a,
                               node.op.b,
                               node.op.min_clamp,
                               node.op.max_clamp)
@register
class DlcPoolTranslation(DlcTranslationBase):
    TARGET = op_adapter.PoolOp.TRANSLATION_KEY
    def lower(self, node, graph, model):
        model.add_pooling_layer(node.op.name,
                                node.op.pool_type,
                                node.op.size_x,
                                node.op.size_y,
                                node.op.stride_x,
                                node.op.stride_y,
                                node.op.pad_x,
                                node.op.pad_y,
                                node.op.padding_size_strategy,
                                node.input_names[0],
                                node.output_names[0],
                                node.op.pool_region_include_padding)
@register
class DlcPermuteTranslation(DlcTranslationBase):
    TARGET = op_adapter.PermuteOp.TRANSLATION_KEY
    def lower(self, node, graph, model):
        model.add_permute_layer(node.op.name,
                                node.op.order,
                                node.input_names[0],
                                node.output_names[0])
@register
class DlcPreluTranslation(DlcTranslationBase):
    TARGET = op_adapter.PreluOp.TRANSLATION_KEY
    def lower(self, node, graph, model):
        model.add_prelu_layer(node.op.name,
                              node.op.coeff,
                              node.input_name[0],
                              node.output_name[0])
@register
class DlcProposalTranslation(DlcTranslationBase):
    TARGET = op_adapter.ProposalOp.TRANSLATION_KEY
    def lower(self, node, graph, model):
        model.add_proposal_layer(node.op.name,
                                 node.op.feat_stride,
                                 node.op.scales,
                                 node.op.ratios,
                                 node.op.anchor_base_size,
                                 node.op.min_bbox_size,
                                 node.op.max_num_proposals,
                                 node.op.max_num_rois,
                                 node.op.iou_threshold_nms,
                                 node.input_names,
                                 node.output_names[0])
@register
class DlcReshapeTranslation(DlcTranslationBase):
    TARGET = op_adapter.ReshapeOp.TRANSLATION_KEY
    def lower(self, node, graph, model):
        model.add_reshape_layer(node.op.name,
                                node.op.output_shape,
                                node.input_names[0],
                                node.output_names[0])
@register
class DlcRNormTranslation(DlcTranslationBase):
    TARGET = op_adapter.RNormOp.TRANSLATION_KEY
    def lower(self, node, graph, model):
        if node.op.across_channels:
            add_method = model.add_cmrn_layer
        else:
            add_method = model.add_local_norm_layer

        add_method(node.op.name,
                   node.op.size,
                   node.op.alpha,
                   node.op.beta,
                   node.op.k,
                   node.input_names[0],
                   node.output_names[0])
@register
class DlcRoiAlignTranslation(DlcTranslationBase):
    TARGET = op_adapter.RoiAlignOp.TRANSLATION_KEY
    def lower(self, node, graph, model):
        model.add_roialign_layer(node.op.name,
                                 node.op.spatial_scale,
                                 node.op.pooled_size_h,
                                 node.op.pooled_size_w,
                                 node.op.sampling_ratio,
                                 node.input_names[0],
                                 node.input_names[1],
                                 node.output_names[0],
                                 node.output_names[1] if len(node.output_names) > 1 else "",
                                 node.op.tiled_batch_h,
                                 node.op.tiled_batch_w,
                                 node.op.batch_pad_h,
                                 node.op.batch_pad_w,
                                 node.op.pad_value)
@register
class DlcRoiPoolingTranslation(DlcTranslationBase):
    TARGET = op_adapter.RoiPoolingOp.TRANSLATION_KEY
    def lower(self, node, graph, model):
        model.add_roipooling_layer(node.op.name,
                                   node.op.pooled_size_w,
                                   node.op.pooled_size_h,
                                   node.op.spatial_scale,
                                   node.op.output_shape,
                                   node.input_names,
                                   node.output_names[0])
@register
class DlcResizeTranslation(DlcTranslationBase):
    TARGET = op_adapter.ResizeOp.TRANSLATION_KEY
    def lower(self, node, graph, model):
        model.add_scaling_layer(node.op.name,
                                node.op.output_shape,
                                node.op.pad_value,
                                node.op.maintain_aspect_ratio,
                                node.op.resize_mode,
                                node.op.scale_height,
                                node.op.scale_width,
                                node.input_names[0],
                                node.output_names[0],
                                node.op.align_corners)

@register
class DlcRnnTransformationTranslation(DlcTranslationBase):
    TARGET = op_adapter.RnnTransformationOp.TRANSLATION_KEY
    def lower(self, node, graph, model):
        model.add_tx_layer(node.op.name,
                           node.op.weights,
                           node.op.bias,
                           node.op.activation,
                           node.input_names[0],
                           node.output_names[0])
@register
class DlcSliceTranslation(DlcTranslationBase):
    TARGET = op_adapter.SliceOp.TRANSLATION_KEY
    def lower(self, node, graph, model):
        model.add_slice_layer(node.op.name,
                              node.input_names[0],
                              node.op.axis,
                              node.op.slice_points,
                              node.output_names)
@register
class DlcSoftmaxTranslation(DlcTranslationBase):
    TARGET = op_adapter.SoftmaxOp.TRANSLATION_KEY
    def lower(self, node, graph, model):
        model.add_softmax_layer(node.op.name,
                                node.input_names[0],
                                node.output_names[0])
@register
class DlcSubtractMeanTranslation(DlcTranslationBase):
    TARGET = op_adapter.SubtractMeanOp.TRANSLATION_KEY
    def lower(self, node, graph, model):
        model.add_subtract_mean_layer(node.op.name,
                                      node.op.mean_values,
                                      node.input_names[0],
                                      node.output_names[0])
@register
class DlcUpsampleIndexBaseTranslation(DlcTranslationBase):
    TARGET = op_adapter.UpsampleIndexBasedOp.TRANSLATION_KEY
    def lower(self, node, graph, model):
        pool_id = model.get_layer_id(node.input_names[1])
        model.add_upsample_layer(node.op.name,
                                 node.op.pool_size,
                                 node.op.pool_stride,
                                 node.op.pad,
                                 node.op.output_height,
                                 node.op.output_width,
                                 node.input_names[0],
                                 node.output_names[0],
                                 pool_id)
@register
class DlcUpsampleSparseTranslation(DlcTranslationBase):
    TARGET = op_adapter.UpsampleSparseOp.TRANSLATION_KEY
    def lower(self, node, graph, model):
        model.add_upsample_layer(node.op.name,
                                 node.op.pool_size,
                                 node.op.pool_stride,
                                 node.op.pad,
                                 node.op.output_height,
                                 node.op.output_width,
                                 node.input_names[0],
                                 node.output_names[0])
