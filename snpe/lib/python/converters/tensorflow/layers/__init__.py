#!/usr/bin/env python
# //=============================================================================
# //  @@-COPYRIGHT-START-@@
# //
# //  Copyright 2015-2016 Qualcomm Technologies, Inc. All rights reserved.
# //  Confidential & Proprietary - Qualcomm Technologies, Inc. ("QTI")
# //
# //  The party receiving this software directly from QTI (the "Recipient")
# //  may use this software as reasonably necessary solely for the purposes
# //  set forth in the agreement between the Recipient and QTI (the
# //  "Agreement"). The software may be used in source code form solely by
# //  the Recipient's employees (if any) authorized by the Agreement. Unless
# //  expressly authorized in the Agreement, the Recipient may not sublicense,
# //  assign, transfer or otherwise provide the source code to any third
# //  party. Qualcomm Technologies, Inc. retains all ownership rights in and
# //  to the software
# //
# //  This notice supersedes any other QTI notices contained within the software
# //  except copyright notices indicating different years of publication for
# //  different portions of the software. This notice does not supersede the
# //  application of any third party copyright notice to that third party's
# //  code.
# //
# //  @@-COPYRIGHT-END-@@
# //=============================================================================
import numpy as np


from converters.tensorflow.layers.fullyconnected import (
    FullyConnectedLayerResolver,
    FullyConnectedLayerBuilder
)
from converters.tensorflow.layers.convolution import (
    ConvolutionLayerResolver,
    ConvolutionLayerBuilder,
    GroupedConvolutionLayerResolver,
    DilatedConvolutionLayerResolver,
    DepthwiseConvolutionLayerResolver,
    DilatedDepthwiseConvolutionLayerResolver
)
from converters.tensorflow.layers.concat import (
    ConcatLayerResolver,
    ConcatLayerBuilder
)
from converters.tensorflow.layers.relu import (
    ReluLayerResolver,
    ReluLayerBuilder
)
from converters.tensorflow.layers.relu_min_max import (
    ReluMinMaxLayerResolver,
    ReluMinMaxLayerBuilder
)
from converters.tensorflow.layers.relu6 import (
    Relu6LayerResolver
)
from converters.tensorflow.layers.sigmoid import (
    SigmoidLayerResolver,
    SigmoidLayerBuilder
)
from converters.tensorflow.layers.tanh import (
    TanhLayerResolver,
    TanhLayerBuilder
)
from converters.tensorflow.layers.softmax import (
    SoftmaxLayerResolver,
    SoftmaxLayerBuilder
)
from converters.tensorflow.layers.lrn import (
    LrnLayerResolver,
    LrnLayerBuilder
)
from converters.tensorflow.layers.deconvolution import (
    DeConvolutionOptimizedLayerResolver,
    DeConvolutionLayerBuilder
)
from converters.tensorflow.layers.batchnorm import (
    BatchNormLayerResolver,
    UnscaledBatchNormLayerResolver,
    ScaledBatchNormLayerResolver,
    BatchNormWithGlobalNormLayerResolver,
    BatchNormLayerBuilder,
    FusedBatchNormNormLayerResolver,
    GenericBatchNormLayerResolver
)
from converters.tensorflow.layers.pooling import (
    AvgPoolingLayerResolver,
    MaxPoolingLayerResolver,
    PoolingLayerBuilder
)
from converters.tensorflow.layers.eltwise import (
    EltWiseSumLayerResolver,
    EltWiseSumLayerBuilder,
    EltWiseMulLayerResolver,
    EltWiseMulLayerBuilder,
    EltWiseMaxLayerResolver,
    EltWiseMaxLayerBuilder
)

from converters.tensorflow.layers.add_n import (
    AddNLayerResolver,
    AddNLayerBuilder
)

from converters.tensorflow.layers.slice import (
    SliceLayerResolver,
    SliceLayerBuilder
)

from converters.tensorflow.layers.prelu import (
    PReLuLayerResolver,
    PReLuLayerBuilder
)

from converters.tensorflow.layers.reshape import (
    ReshapeLayerResolver,
    ReshapeLayerBuilder
)

from converters.tensorflow.layers.resize import (
    ResizeNearestNeighborLayerResolver,
    ResizeBilinearLayerResolver,
    ResizeLayerBuilder
)

from converters.tensorflow.layers.lstm import (
    LstmLayerResolver,
    LstmLayerBuilder
)
from converters.tensorflow.layers.ignored_patterns import (
    IgnoredLayersResolver,
    IgnoredLayersBuilder
)

from converters.tensorflow.layers.fill import (
    FillLayerResolver,
    FillLayerBuilder
)

from converters.tensorflow.layers.ssd import (
    SSDDecoderResolver,
    SSDDecoderLayersBuilder,
    SSDNmsResolver,
    SSDNmsLayersBuilder,
    SSDAnchorGeneratorResolver,
)

from converters.tensorflow.layers.crop import (
    CropLayerResolver,
    CropLayerBuilder
)

from converters.tensorflow.layers.constant import (
    ConstantLayerResolver,
    ConstantLayerBuilder
)

from converters.tensorflow.common import (
    LayerDescriptor,
    LayerResolver,
    LayerBuilder
)

layer_resolvers = [
    IgnoredLayersResolver,
    SSDAnchorGeneratorResolver,
    SSDNmsResolver,
    ConvolutionLayerResolver,
    ConcatLayerResolver,
    FullyConnectedLayerResolver,
    ReluLayerResolver,
    Relu6LayerResolver,
    SigmoidLayerResolver,
    TanhLayerResolver,
    AvgPoolingLayerResolver,
    MaxPoolingLayerResolver,
    SoftmaxLayerResolver,
    LrnLayerResolver,
    DeConvolutionOptimizedLayerResolver,
    EltWiseSumLayerResolver,
    EltWiseMulLayerResolver,
    EltWiseMaxLayerResolver,
    UnscaledBatchNormLayerResolver,
    ScaledBatchNormLayerResolver,
    BatchNormWithGlobalNormLayerResolver,
    GenericBatchNormLayerResolver,
    GroupedConvolutionLayerResolver,
    SliceLayerResolver,
    PReLuLayerResolver,
    DilatedConvolutionLayerResolver,
    ReshapeLayerResolver,
    ResizeBilinearLayerResolver,
    ResizeNearestNeighborLayerResolver,
    DepthwiseConvolutionLayerResolver,
    DilatedDepthwiseConvolutionLayerResolver,
    AddNLayerResolver,
    LstmLayerResolver,
    FillLayerResolver,
    SSDDecoderResolver,
    CropLayerResolver,
    FusedBatchNormNormLayerResolver,
]
"""
type: list[type(LayerResolver)]
"""

layer_builders = {
    BatchNormLayerResolver.Descriptor: BatchNormLayerBuilder,
    BatchNormWithGlobalNormLayerResolver.Descriptor: BatchNormLayerBuilder,
    GenericBatchNormLayerResolver.Descriptor: BatchNormLayerBuilder,
    ConcatLayerResolver.Descriptor: ConcatLayerBuilder,
    ConvolutionLayerResolver.Descriptor: ConvolutionLayerBuilder,
    DeConvolutionOptimizedLayerResolver.Descriptor: DeConvolutionLayerBuilder,
    EltWiseMaxLayerResolver.Descriptor: EltWiseMaxLayerBuilder,
    EltWiseMulLayerResolver.Descriptor: EltWiseMulLayerBuilder,
    EltWiseSumLayerResolver.Descriptor: EltWiseSumLayerBuilder,
    AddNLayerResolver.Descriptor: AddNLayerBuilder,
    FullyConnectedLayerResolver.Descriptor: FullyConnectedLayerBuilder,
    LrnLayerResolver.Descriptor: LrnLayerBuilder,
    ReluLayerResolver.Descriptor: ReluLayerBuilder,
    Relu6LayerResolver.Descriptor: ReluMinMaxLayerBuilder,
    SigmoidLayerResolver.Descriptor: SigmoidLayerBuilder,
    SoftmaxLayerResolver.Descriptor: SoftmaxLayerBuilder,
    TanhLayerResolver.Descriptor: TanhLayerBuilder,
    AvgPoolingLayerResolver.Descriptor: PoolingLayerBuilder,
    MaxPoolingLayerResolver.Descriptor: PoolingLayerBuilder,
    GroupedConvolutionLayerResolver.Descriptor: ConvolutionLayerBuilder,
    SliceLayerResolver.Descriptor: SliceLayerBuilder,
    PReLuLayerResolver.Descriptor: PReLuLayerBuilder,
    DilatedConvolutionLayerResolver.Descriptor: ConvolutionLayerBuilder,
    ReshapeLayerResolver.Descriptor: ReshapeLayerBuilder,
    ResizeBilinearLayerResolver.Descriptor: ResizeLayerBuilder,
    ResizeNearestNeighborLayerResolver.Descriptor: ResizeLayerBuilder,
    LstmLayerResolver.UnrolledTimeStepDescriptor: LstmLayerBuilder,
    LstmLayerResolver.StateDescriptor: LstmLayerBuilder,
    IgnoredLayersResolver.Descriptor: IgnoredLayersBuilder,
    FillLayerResolver.Descriptor: FillLayerBuilder,
    SSDDecoderResolver.Descriptor: SSDDecoderLayersBuilder,
    CropLayerResolver.Descriptor: CropLayerBuilder,
    SSDNmsResolver.Descriptor: SSDNmsLayersBuilder,
    ConstantLayerResolver.Descriptor: ConstantLayerBuilder,
    FusedBatchNormNormLayerResolver.Descriptor: BatchNormLayerBuilder,
}
"""
type: dict[type(LayerDescriptor), type(LayerBuilder)]
"""
