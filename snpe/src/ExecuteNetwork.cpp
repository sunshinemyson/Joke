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

#include <iostream>
#include <algorithm>
#include <sstream>
#include <unordered_map>

#include "ExecuteNetwork.hpp"
#include "Util.hpp"

#include "SNPE/SNPE.hpp"
#include "DlSystem/ITensor.hpp"
#include "DlSystem/StringList.hpp"
#include "DlSystem/TensorMap.hpp"
#include "DlSystem/TensorShape.hpp"

void executeNetwork (std::unique_ptr<zdl::SNPE::SNPE> & snpe,
                     std::unique_ptr<zdl::DlSystem::ITensor> & input,
                     const std::string& outputDir,
                     int num)
{
    // Execute the network and store the outputs that were specified when creating the network in a TensorMap
    static zdl::DlSystem::TensorMap outputTensorMap;
    snpe->execute(input.get(), outputTensorMap);
    zdl::DlSystem::StringList tensorNames = outputTensorMap.getTensorNames();

    // Iterate through the output Tensor map, and print each output layer name
    std::for_each( tensorNames.begin(), tensorNames.end(), [&](const char* name)
    {
        std::ostringstream path;
        path << outputDir << "/"
        << "Result_" << num << "/"
        << name << ".raw";
        auto tensorPtr = outputTensorMap.getTensor(name);
        SaveITensor(path.str(), tensorPtr);
    });
}

void executeNetwork(std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                    zdl::DlSystem::UserBufferMap& inputMap,
                    zdl::DlSystem::UserBufferMap& outputMap,
                    std::unordered_map<std::string,std::vector<uint8_t>>& applicationOutputBuffers,
                    const std::string& outputDir,
                    int num)
{
    // Execute the network and store the outputs in user buffers specified in outputMap
    snpe->execute(inputMap, outputMap);

    // Get all output buffer names from the network
    const zdl::DlSystem::StringList& outputBufferNames = outputMap.getUserBufferNames();

    // Iterate through output buffers and print each output to a raw file
    if (outputDir.empty()) return;
    std::for_each(outputBufferNames.begin(), outputBufferNames.end(), [&](const char* name)
    {
       std::ostringstream path;
       path << outputDir << "/Result_" << num << "/" << name << ".raw";

       SaveUserBuffer(path.str(), applicationOutputBuffers.at(name));
    });
}
