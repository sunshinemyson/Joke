//==============================================================================
//
//  @@
//
//  Copyright 2017 Qualcomm Technologies, Inc. All rights reserved.
//  Confidential & Proprietary - Qualcomm Technologies, Inc. ("QTI")
//
//  The party receiving this software directly from QTI (the "Recipient")
//  may use this software as reasonably necessary solely for the purposes
//  set forth in the agreement between the Recipient and QTI (the
//  "Agreement"). The software may be used in source code form solely by
//  the Recipient's employees (if any) authorized by the Agreement. Unless
//  expressly authorized in the Agreement, the Recipient may not sublicense,
//  assign, transfer or otherwise provide the source code to any third
//  party. Qualcomm Technologies, Inc. retains all ownership rights in and
//  to the software
//
//  This notice supersedes any other QTI notices contained within the software
//  except copyright notices indicating different years of publication for
//  different portions of the software. This notice does not supersede the
//  application of any third party copyright notice to that third party's
//  code.
//
//  @@
//
//==============================================================================

#include "SetBuilderOptions.hpp"

#include "SNPE/SNPE.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "SNPE/SNPEBuilder.hpp"

std::unique_ptr<zdl::SNPE::SNPE> setBuilderOptions(std::unique_ptr<zdl::DlContainer::IDlContainer> & container,
                                                   zdl::DlSystem::Runtime_t runtime,
                                                   zdl::DlSystem::UDLBundle udlBundle,
                                                   bool useUserSuppliedBuffers,
                                                   const zdl::DlSystem::TensorShapeMap& inputDim)
{
    std::unique_ptr<zdl::SNPE::SNPE> snpe;
    zdl::SNPE::SNPEBuilder snpeBuilder(container.get());
    // snpeBuilder.setInputDimensions(inputDim);
    snpe = snpeBuilder.setOutputLayers({})
       .setRuntimeProcessor(runtime)
       .setUdlBundle(udlBundle)
       .setUseUserSuppliedBuffers(useUserSuppliedBuffers)
       .build();

    return snpe;
}
