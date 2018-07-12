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

#ifndef EXECUTE_H
#define EXECUTE_H

#include <string>
#include <unordered_map>
#include <vector>

#include "SNPE/SNPE.hpp"
#include "DlSystem/ITensor.hpp"
#include "DlSystem/UserBufferMap.hpp"

void executeNetwork(std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                    std::unique_ptr<zdl::DlSystem::ITensor>& input,
                    const std::string& outputDir,
                    int num);

void executeNetwork(std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                    zdl::DlSystem::UserBufferMap& inputMap,
                    zdl::DlSystem::UserBufferMap& outputMap,
                    std::unordered_map<std::string,std::vector<uint8_t>>& applicationOutputBuffers,
                    const std::string& outputDir,
                    int num);

#endif
