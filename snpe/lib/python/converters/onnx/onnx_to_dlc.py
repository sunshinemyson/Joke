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
import argparse
import sys
import traceback
from .. import translation, op_adapter, op_graph, lower_to_dlc
from util import *
try:
    import onnx
    parse_model = onnx.load
except ImportError:
    def parse_model(model_path):
        raise Exception(ERROR_ONNX_NOT_FOUND.format(str(sys.path)))

import onnx_translations


#------------------------------------------------------------------------------
#   Command Line Processing
#------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path","-m",help="Path to the source ONNX model.",
                        required=True)
    parser.add_argument("--dlc_path","-d",help="Path where the converted DLC model should be saved.")
    parser.add_argument('--encoding',
                        help='Set the image encoding for an input buffer. This should be specifed in the format "--encoding <input name> <encoding>", where encoding is one of: "argb32", "rgba", "nv21", "opaque", or "bgr". The defautl encoding for all inputs not so described is "bgr". "opaque" inputs will be interpreted as-is by SNPE, and not subject to order transformations.',
                        nargs=2, action='append')
    parser.add_argument("--debug",help="Run the converter in debug mode", action="store_true")
    args = parser.parse_args()
    return args

def sanitize_args(args):
    sanitized_args = []
    for k, v in vars(args).iteritems():
        if k in ['model_path','m','dlc_path','d']:
            continue
        sanitized_args.append('{}={}'.format(k,v))
    return "{} {}".format(sys.argv[0].split('/')[-1], ' '.join(sanitized_args))

#------------------------------------------------------------------------------
#   The Converter Class
#------------------------------------------------------------------------------
class OnnxConverter(object):
    def __init__(self):
        self.translations = onnx_translations.OnnxTranslations
        self.graph = op_graph.OpGraph(naming_policy=OnnxNamePolicy(),
                                      shape_inference_policy=OnnxShapeInferencePolicy())

        self.input_model_path = ''
        self.output_model_path = ''
        self.debug = False

    def __call__(self, args):
        self.set_options(args)
        self.convert(parse_model(args.model_path))
        self.save()

    def set_options(self, args):
        self.input_model_path = args.model_path
        self.output_model_path = args.dlc_path
        self.debug = args.debug
        self.converter_command = sanitize_args(args)
        setup_logging(args)

    def convert(self, model):
        self.graph.weights = WeightProvider(model)
        # extract inputs
        parameter_names = set()
        for tensor in model.graph.initializer:
            parameter_names.add(str(tensor.name))

        for value_info in model.graph.input:
            name = str(value_info.name)
            if name in parameter_names:
                # weights are usually listed as inputs too.
                continue
            tensor_shape = value_info.type.tensor_type.shape
            shape = [int(dim.dim_value) for dim in tensor_shape.dim]
            self.graph.add_input(name, shape, 'bgr', 'default')

        # extract parameters, infer shapes, etc.
        for i, src_op in enumerate(model.graph.node):
            LOG_DEBUG(DEBUG_CONVERTING_NODE, i, src_op.op_type)
            src_type = onnx_type(src_op.op_type)
            try:
                self.translations.apply_specific(src_type,
                                                 onnx_translations.ADD_OP,
                                                 src_op,
                                                 self.graph)
            except Exception, e:
                if self.debug:
                    traceback.print_exc()
                LOG_ERROR("Node %s: %s" % (src_op.name, e))
                sys.exit(-1)

        try:
            # apply graph transformations
            self.translations.apply_partial(onnx_translations.SQUASH_SCALE, self.graph)
            self.translations.apply_partial(onnx_translations.SQUASH_BATCHNORM, self.graph)

            # transition to HWC
            self.translations.apply_total(onnx_translations.AXES_TO_SNPE_ORDER, self.graph)

            # remove NOOPs, which may include trivial permutes at this point
            self.translations.apply_partial(onnx_translations.REMOVE_NOOP, self.graph)
        except Exception, e:
            if self.debug:
                traceback.print_exc()
            LOG_ERROR(str(e))
            sys.exit(-1)

    def save(self):
        if not self.output_model_path:
            output_path = self.input_model_path + '.dlc'
        else:
            output_path = self.output_model_path
        LOG_INFO(INFO_DLC_SAVE_LOCATION, output_path)
        model = lower_to_dlc.lower(self.graph)
        model.set_converter_command(self.converter_command)
        model.save(output_path)

#------------------------------------------------------------------------------
#   Policies
#------------------------------------------------------------------------------
class OnnxNamePolicy(object):
    def __init__(self):
        self.type_count = {}

    def get_op_name(self, op):
        if op.name:
            return str(op.name)
        else:
            count = self.type_count.get(op.type, 0)
            self.type_count[op.type] = count+1
            return "%s_%d" % (op.type, count)

    def get_input_names(self, op, input_names):
        return map(str, input_names)

    def get_output_names(self, op, output_names):
        return map(str, output_names)

class OnnxShapeInferencePolicy(object):

    def infer_shape(self, op, input_shapes):
        return onnx_translations.OnnxTranslations.apply_specific(op.type,
                                                                 onnx_translations.INFER_SHAPE,
                                                                 op,
                                                                 input_shapes)
