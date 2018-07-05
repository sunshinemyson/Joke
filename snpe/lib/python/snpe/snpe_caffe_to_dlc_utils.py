#!/usr/bin/env python2.7
# -*- mode: python -*-
#//=============================================================================
#//  @@
#//
#//  Copyright 2016-2017 Qualcomm Technologies, Inc. All rights reserved.
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
from snpe.common import snpe_validation_utils

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
            'Script to convert caffe protobuf configuration into a DLC file.')
        parser._action_groups.pop()

        required = parser.add_argument_group('required arguments')
        required.add_argument('-c', '--caffe_txt', type=str, required=True,
                            help='Input caffe proto txt configuration file')

        optional = parser.add_argument_group('optional arguments')


        optional.add_argument('-b','--caffe_bin', type=str,
                            help='Input caffe binary file containing the weight data')
        optional.add_argument('-d', '--dlc', type=str,
                            help='Output DLC file containing the model. If not specified, the data will be written to a file with same name as the caffetxt file with a .dlc extension')
        # The "omit_preprocessing" argument populates a variable called "enable_preprocessing" with its opposite value, so that
        # we avoid "double-negatives" all over the code when using it.
        optional.add_argument('--omit_preprocessing', dest="enable_preprocessing", action="store_const", const=False, default=True,
                            help="If specified, converter will disable preprocessing specified by a data layer transform_param or any preprocessing command line options")
        optional.add_argument('--encoding', type=str, choices=['argb32', 'rgba', 'nv21', 'bgr'], default='bgr',
                            help='Image encoding of the source images. Default is bgr if not specified')
        optional.add_argument('--input_size', type=int, nargs=2, metavar=('WIDTH','HEIGHT'),
                            help='Dimensions of the source images for scaling, if different from the network input.')
        optional.add_argument('--model_version', type=str,
                            help='User-defined ASCII string to identify the model, only first 64 bytes will be stored')
        optional.add_argument('--disable_batchnorm_folding', dest="disable_batchnorm_folding", action="store_true",
                            help="If not specified, converter will try to fold batchnorm into previous convolution layer")
        optional.add_argument('--in_layer', type=str, action='append', dest='input_layers',
                              help='Name of the input layer')
        optional.add_argument('--in_type', type=str, choices=['default', 'image', 'opaque'], action='append', dest='input_types',
                              help='Type of data expected by input layer. Type is default if not specified.')
        optional.add_argument('--validation_target', nargs=2, metavar=('RUNTIME_TARGET','PROCESSOR_TARGET'), default = [], action=snpe_validation_utils.ValidateTargetArgs,
                            help="A combination of processor and runtime target against which model will be validated."
                            "Choices for RUNTIME_TARGET: {cpu, gpu, dsp}."
                            "Choices for PROCESSOR_TARGET: {snapdragon_801, snapdragon_820, snapdragon_835}."
                            "If not specified, will validate model against {snapdragon_820, snapdragon_835} across all runtime targets.")
        optional.add_argument('--strict', dest="enable_strict_validation", action="store_true", default=False,
                            help="If specified, will validate in strict mode whereby model will not be produced if it violates constraints of the specified validation target."
                                 "If not specified, will validate model in permissive mode against the specified validation target.")
        optional.add_argument("--verbose", dest="verbose", action="store_true",
                            help="Verbose printing", default = False)

        args = parser.parse_args()
        if args.dlc is None:
            filename, fileext = os.path.splitext(os.path.realpath(args.caffe_txt))
            args.dlc = filename + ".dlc"

        return args
