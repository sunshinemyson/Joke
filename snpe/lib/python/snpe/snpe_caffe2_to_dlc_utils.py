#!/usr/bin/env python2.7
# -*- mode: python -*-
#//=============================================================================
#//  @@
#//
#//  Copyright 2017 Qualcomm Technologies, Inc. All rights reserved.
#//  Confidential & Proprietary - Qualcomm Technologies, Inc. ("QTI")
#//
#//  The party receiving this software directly from QTI (the "Recipient")
#//  may use this software as reasonably necessary solely for the purposes
#//  set forth in the agreement between the Recipient and QTI (the
#//  "Agreement"). The software may be used in source code form solely by
#//  the Recipient's employees (if any) authorized by the Agreement. Unless
#//  expressly authorized in the Agreement, the Recipient may not sublicense,
#//  assign, transfer or otherwise provide the source code to any third
#//  party. Qualcomm Technologies, Inc. retains all ownership rights in and
#//  to the software
#//
#//  This notice supersedes any other QTI notices contained within the software
#//  except copyright notices indicating different years of publication for
#//  different portions of the software. This notice does not supersede the
#//  application of any third party copyright notice to that third party's
#//  code.
#//
#//  @@
#//=============================================================================

import argparse
import logging
import numpy
import os

import snpe
from snpe.common import snpe_udl_utils

class SNPEUtils(object):
    def blob2arr(self, blob):
        if hasattr(blob, "shape"):
            return numpy.ndarray(buffer=blob.data, shape=blob.shape, dtype=numpy.float32)
        else:
       #Caffe-Segnet fork doesn't have shape field exposed on blob.
            return numpy.ndarray(buffer=blob.data, shape=blob.data.shape, dtype=numpy.float32)

    def setUpLogger(self, verbose):
        formatter = '%(asctime)s - %(lineno)d - %(levelname)s - %(message)s'
        lvl = logging.INFO
        if verbose:
             lvl = logging.DEBUG
        logger = logging.getLogger()
        logger.setLevel(lvl)
        formatter = logging.Formatter(formatter)

        ch = logging.StreamHandler()
        ch.setLevel(lvl)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    def getArgs(self):
        logger = logging.getLogger()
        logger.debug("Parsing the arguments")

        parser = argparse.ArgumentParser(
            description=
            'Script to convert caffe2 networks into a DLC file.')

        required = parser.add_argument_group('required arguments')
        required.add_argument('-p', '--predict_net', type=str, required=True,
                              help='Input caffe2 binary network definition protobuf')
        required.add_argument('-e', '--exec_net', type=str, required=True,
                              help='Input caffe2 binary file containing the weight data')
        required.add_argument('-i', '--input_dim', nargs=2, action='append', required=True,
                            help='The names and dimensions of the network input layers specified in the format "input_name" C,H,W. Ex "data" 3,224,224. Note that the quotes should always be included in order to handle special characters, spaces, etc. For multiple inputs specify multiple --input_dim on the command line like: --input_dim "data1" 3,224,224 --input_dim "data2" 3,50,100 We currently assume that all inputs have 3 dimensions.')

        optional = parser.add_argument_group('optional arguments')
        optional.add_argument('-d', '--dlc', type=str,
                            help='Output DLC file containing the model. If not specified, the data will be written to a file with same name and location as the predict_net file with a .dlc extension')

        # The "enable_preprocessing" option only works when ImageInputOp is specified. Otherwise preprocessing must occur prior to passing the input to SNPE
        optional.add_argument('--enable_preprocessing', action="store_const", const=True, default=False,
                            help="If specified, the converter will enable image mean subtraction and cropping specified by ImageInputOp. Do NOT enable if there is not a ImageInputOp present in the Caffe2 network.")
        optional.add_argument('--encoding', type=str, choices=['argb32', 'rgba', 'nv21', 'bgr'], default='bgr',
                            help='Image encoding of the source images. Default is bgr if not specified')
        optional.add_argument('--opaque_input', type=str, help="A space separated list of input blob names which should be treated as opaque (non-image) data. These inputs will be consumed as-is by SNPE. Any input blob not listed will be assumed to be image data.", nargs='*', default=[])
        optional.add_argument('--model_version', type=str,
                            help='User-defined ASCII string to identify the model, only first 64 bytes will be stored')
        optional.add_argument('--reorder_list', nargs='+',
                            help='A list of external inputs or outputs that SNPE should automatically reorder to match the specified Caffe2 channel ordering. Note that this feature is only enabled for the GPU runtime.', default = []) 
        optional.add_argument("--verbose", dest="verbose", action="store_true",
                            help="Verbose printing", default = False)

        args = parser.parse_args()
        if args.dlc is None:
            filename, fileext = os.path.splitext(os.path.realpath(args.predict_net))
            args.dlc = filename + ".dlc"

        return args
