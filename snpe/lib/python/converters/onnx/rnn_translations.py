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
#   GRU
#------------------------------------------------------------------------------
# class BidirectionalGruOp(op_adapter.Op):
#     TRANSLATION_KEY = 'bidirectional_gru'
#     def __init__(self, name, **kargs):
#         op_adapter.Op.__init__(self, name, self.TRANSLATION_KEY)
#         self.assertattr('weights', kargs)
#         self.assertattr('recurrence_weights', kargs)
#         self.addattr('bias', kargs, None)
#         self.addattr('activations',kargs,[])
#         self.assertattr('hidden_size',kargs)

# SPLIT_GRU='split_gru'

# class OnnxGruTranslation(OnnxTranslationBase):
#     def __init__(self):
#         OnnxTranslationBase.__init__(self)
#         self.index_method(SPLIT_GRU, self.split_gru)

#     def extract_parameters(self, src_op, graph):
#         params = extract_attributes(src_op,
#                                     ('activation_alpha','lf',None),
#                                     ('activation_beta', 'lf',None),
#                                     ('activations','ls',[]),
#                                     ('clip','f',None),
#                                     ('direction','s','forward'),
#                                     ('hidden_size','i'),
#                                     ('linear_before_reset','i',0),
#                                     ('output_sequence','i',0))
#         LOG_ASSERT(not params.activation_alpha,
#                    "Node %s: SNPE does not support activation scale parameters in GRU ops",
#                    src_op.name)
#         LOG_ASSERT(not params.activation_beta,
#                    "Node %s: SNPE does not support activation scale parameters in GRU ops",
#                    src_op.name)
#         LOG_ASSERT(not params.clip,
#                    "Node %s: SNPE does not support pre-activation clipping in GRU ops",
#                    src_op.name)
#         LOG_ASSERT(params.linear_before_reset,
#                    "Node %s: SNPE only supports linear_before_reset != 0 in GRU ops",
#                    src_op.name)
#         LOG_ASSERT(not params.output_sequence,
#                    "Node %s: SNPE does not support outputting sequence values from GRU ops",
#                    src_op.name)

#         input_names = map(str, src_op.input)
#         LOG_ASSERT(len(input_names) <= 4,
#                    "Node %s: SNPE does not support sequence_lens or initial_h inputs to GRU ops",
#                    src_op.name)
#         weights, rec_weights = graph.weights.fetch(*input_names[1:3])
#         bias = None
#         if len(input_names) == 4:
#             bias = graph.weights.fetch(input_names[3])
#         if params.direction == 'bidirectional':
#             return BidirectionalGruOp(src_op.name,
#                                       weights=weights,
#                                       recurrence_weights=rec_weights,
#                                       bias=bias,
#                                       activations=map(extract_activation, params.activations),
#                                       hidden_size=params.hidden_size)
#         else:
#             return self.create_unidirectional_gru(src_op.name,
#                                                   weights,
#                                                   rec_weights,
#                                                   bias,
#                                                   params.hidden_size,
#                                                   map(extract_activation, params.activations),
#                                                   direction=='reverse')

#     def create_unidirectional_gru(self,
#                                   name,
#                                   weights,
#                                   rec_weights,
#                                   bias,
#                                   size,
#                                   activations,
#                                   backward):

#         if bias is None:
#             bias = numpy.zeros((3,size),dtype=numpy.float32)
#         else:
#             # for probably vendor specific reasons, ONNX defines GRU bias to
#             # be separated into forward and recurrent parts, that are always
#             # added together (unless linear_before_reset is false, but we
#             # don't support that. So we will combine here.
#             bias.shape = (6,size)
#             new_bias = numpy.empty((3,size), dtype=numpy.float32)
#             # z stored in parts 0 and 3
#             numpy.add(bias.shape[0,:],bias.shape[3,:],out=new_bias[0,:])
#             # r stored in parts 1 and 4
#             numpy.add(bias.shape[1,:],bias.shape[4,:],out=new_bias[1,:])
#             # h stored in parts 2 and 5
#             numpy.add(bias.shape[2,:],bias.shape[5,:],out=new_bias[2,:])

#             bias = new_bias

#         # weights laid out as [zrh*hidden_size,input_size]
#         # z maps to control, r to forget, h to state
#         weights.shape = (3,size,weights.shape[-1])
#         rec_weights.shape = (3,size,size)
#         control_gate = { 'weights':weights[0,:,:],
#                          'rec_weights':rec_weights[0,:,:],
#                          'bias':bias[0,:] }
#         forget_gate = { 'weights':weights[1,:,:],
#                         'rec_weights':rec_weights[1,:,:],
#                         'bias':bias[1,:] }
#         state_gate = { 'weights':weights[2,:,:],
#                        'rec_weights':rec_weights[2,:,:],
#                        'bias':bias[2,:] }
#         return op_adapter.GruOp(src_op.name,
#                                 state_gate,
#                                 forget_gate,
#                                 control_gate,
#                                 activation=activations[0],
#                                 gate_activation=activations[1],
#                                 rec_gate_activation=activations[2],
#                                 backwards = backward)

#     def split_gru(self, node, graph):
#         name = node.op.name
#         weights = node.op.weights
#         rec_weights = node.op.rec_weights,
#         activations = node.op.activations
#         forward_op = self.create_unidirectional_gru(name + '_forward',
#                                                     weights[0,:,:],
#                                                     rec_weights[0,:,:],
#                                                     bias[0,:] if bias else None,
#                                                     node.op.hidden_size,
#                                                     activations,
#                                                     False)

# the GRU translation is incomplete. Don't register it until there's a test for it.

#------------------------------------------------------------------------------
#   LSTM
#------------------------------------------------------------------------------
# TBD

#------------------------------------------------------------------------------
#   RNN
#------------------------------------------------------------------------------
# Pending test model
