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
import logging
import numpy
from snpe import modeltools
from messages import *
try:
    import onnx
    from onnx.numpy_helper import to_array as extract_onnx_tensor
except:
    onnx = None # converter will throw before we try anything in here

def is_broadcast(onnx_op):
    return getattr(onnx_op, 'axis', 0) != 0 or getattr(onnx_op, 'broadcast', 1) == 1

def assert_no_broadcast(onnx_op):
    ASSERT(not is_broadcast(onnx_op),
           ERROR_BROADCAST_NOT_SUPPORTED,
           onnx_op.name)

class NamedDict(dict):
    def __getattr__(self, key):
        return self[key]

def extract_attributes(onnx_op, *attr_infos):
    """Ensure the existence and extract well typed attributes from an onnx
    NodeProto.

    Each entry in attr_info should be either a 2- or 3-tuple.
    * The first element should be the string name of an attribute.
    * The second element should by a type code for the attribute corresponding to:
      - i for int attributes
      - f for float attributes
      - s for string attributes
      - t for tensor attributes (returned as a numpy array)
      - g for graph attributes
      - lx, where x is one of the preceeding attribute type identifiers, for list valued attributes
    * The third element, if present, specifies a default value should the attribute not be present.
      If no default is specified, this function will thrown an error.

    The return object will have a named property for each attribute info."""
    onnx_attrs = {}
    for attr in onnx_op.attribute:
        onnx_attrs[attr.name] = attr

    code_to_enum = { 'i':onnx.AttributeProto.INT,
                     'f':onnx.AttributeProto.FLOAT,
                     's':onnx.AttributeProto.STRING,
                     't':onnx.AttributeProto.TENSOR,
                     'g':onnx.AttributeProto.GRAPH,
                     'li':onnx.AttributeProto.INTS,
                     'lf':onnx.AttributeProto.FLOATS,
                     'ls':onnx.AttributeProto.STRINGS,
                     'lt':onnx.AttributeProto.TENSORS,
                     'lg':onnx.AttributeProto.GRAPHS }

    ret = NamedDict()
    for attr_info in attr_infos:
        name = attr_info[0]
        if not name in onnx_attrs:
            if len(attr_info) == 3:
                ret[name] = attr_info[2]
                continue
            else:
                raise ValueError(ERROR_ATTRIBUTE_MISSING.format(onnx_op.name, name))
        attr = onnx_attrs[name]
        code = attr_info[1]
        requested_type = code_to_enum[code]
        if attr.type != requested_type:
            msg=ERROR_ATTRIBUTE_WRONG_TYPE.format(onnx_op.name,
                                                  name,
                                                  onnx.AttributeProto.AttributeType.Name(requested_type),
                                                  onnx.AttributeProto.AttributeType.Name(attr.type))
            raise ValueError(msg)
        if code == 'i':
            ret[name] = int(attr.i)
        elif code == 'f':
            ret[name] = float(attr.f)
        elif code == 's':
            ret[name] = str(attr.s)
        elif code == 'g':
            ret[name] = attr.g
        elif code == 't':
            ret[name] = extract_onnx_tensor(attr.t)
        elif code == 'li':
            ret[name] = map(int, attr.ints)
        elif code == 'lf':
            ret[name] = map(float, attr.floats)
        elif code == 'ls':
            ret[name] = map(str, attr.strings)
        elif code == 'lg':
            ret[name] = list(attr.graphs)
        elif code == 'lt':
            ret[name] = map(extract_onnx_tensor, attr.tensors)

    return ret

def extract_activation(onnx_activation):
    acts = { 'Relu':modeltools.NEURON_RELU,
             'Tanh':modeltools.NEURON_TANH,
             'Sigmoid':modeltools.NEURON_LOGISTIC }
    try:
        return acts[str(onnx_activation)]
    except KeyError:
        raise ValueError(ERROR_ACTIVATION_FUNCTION_UNSUPPORTED.format(onnx_activation))

def extract_padding_mode(auto_pad, node_name):
    if auto_pad == 'VALID':
        return modeltools.PADDING_SIZE_IMPLICIT_VALID
    elif auto_pad == 'SAME_LOWER':
        return modeltools.PADDING_SIZE_IMPLICIT_SAME
    elif auto_pad == '':
        return modeltools.PADDING_SIZE_EXPLICIT_FLOOR
    else:
        raise ValueError(ERROR_PADDING_TYPE_UNSUPPORTED.format(node_name, auto_pad))

def onnx_type(type_name):
    """Convert an onnx type name string to a namespaced format"""
    return 'onnx_' + str(type_name).lower()

def pads_symmetric(pads):
    num_dims = len(pads)/2
    for i in xrange(num_dims):
        if pads[i] != pads[i+num_dims]:
            return False
    return True

def pads_righthanded(pads):
    num_dims = len(pads)/2
    for i in xrange(num_dims):
        if pads[i] != 0:
            return False
    # don't call all zeros righthanded
    return not all(x == 0 for x in pads)

def product(nums):
    if len(nums) == 0:
        return 1
    else:
        return reduce(int.__mul__, nums)

#------------------------------------------------------------------------------
#   WeightProvider
#------------------------------------------------------------------------------
class WeightProvider(object):
    def __init__(self, model):
        self.weight_map = {}
        for tensor in model.graph.initializer:
            self.weight_map[str(tensor.name)] = extract_onnx_tensor(tensor)

    def fetch(self, *keys):
        ret = []
        for key in keys:
            key = str(key)
            if not key in self.weight_map:
                raise KeyError(ERROR_WEIGHTS_MISSING_KEY.format(key))
            ret.append(numpy.require(self.weight_map[key].copy(), dtype=numpy.float32))
        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def has(self, key):
        return key in self.weight_map

    def insert(self, key, weights):
        self.weight_map[key] = weights

#------------------------------------------------------------------------------
#   Logging
#------------------------------------------------------------------------------
def ASSERT(cond, msg, *args):
    assert cond, msg.format(*args)

LOGGER = None
def setup_logging(args):
    global LOGGER

    if args.debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(lineno)d - %(levelname)s - %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(log_level)
    LOGGER.addHandler(handler)

def LOG_DEBUG(msg, *args):
    if LOGGER:
        LOGGER.debug(msg.format(*args))

def LOG_ERROR(msg, *args):
    if LOGGER:
        LOGGER.error(msg.format(*args))

def LOG_INFO(msg, *args):
    if LOGGER:
        LOGGER.info(msg.format(*args))

def LOG_WARNING(msg, *args):
    if LOGGER:
        LOGGER.warning(msg.format(*args))
