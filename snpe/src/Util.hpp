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

#ifndef UTIL_H
#define UTIL_H

#include <vector>
#include <string>
#include <sstream>

#include "DlSystem/ITensorFactory.hpp"
#include "DlSystem/TensorShape.hpp"

template <typename Container> Container& split(Container& result, const typename Container::value_type & s, typename Container::value_type::value_type delimiter )
{
  result.clear();
  std::istringstream ss( s );
  while (!ss.eof())
  {
    typename Container::value_type field;
    getline( ss, field, delimiter );
    if (field.empty()) continue;
    result.push_back( field );
  }
  return result;
}

size_t calcSizeFromDims(const zdl::DlSystem::Dimension *dims, size_t rank, size_t elementSize);

std::vector<float> loadFloatDataFile(const std::string& inputFile);
std::vector<unsigned char> loadByteDataFile(const std::string& inputFile);
template<typename T> void loadByteDataFile(const std::string& inputFile, std::vector<T>& loadVector);

void SaveITensor(const std::string& path, const zdl::DlSystem::ITensor* tensor);
void SaveUserBuffer(const std::string& path, const std::vector<uint8_t>& buffer);
bool EnsureDirectory(const std::string& dir);

#endif

