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

valid_processor_choices = ('snapdragon_801', 'snapdragon_820', 'snapdragon_835')
valid_runtime_choices = ('cpu','gpu', 'dsp')

class ValidateTargetArgs(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        specified_runtime, specified_processor = values
        if specified_runtime not in valid_runtime_choices:
            raise ValueError('invalid runtime_target {s1!r}. Valid values are {s2}'.format(s1=specified_runtime, s2=valid_runtime_choices))
        if specified_processor not in valid_processor_choices:
            raise ValueError('invalid processor_target {s1!r}. Valid values are {s2}'.format(s1=specified_processor, s2=valid_processor_choices))
        setattr(args, self.dest, values)
