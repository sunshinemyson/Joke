#include <android/bitmap.h>
#include <jni.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <unordered_map>

#include <algorithm>
#include "CheckRuntime.hpp"
#include "CreateUserBuffer.hpp"
#include "ExecuteNetwork.hpp"
#include "LoadContainer.hpp"
#include "SetBuilderOptions.hpp"
#include "Util.hpp"
#include "android_log.hpp"
#include "DlSystem/TensorShapeMap.hpp"

namespace {
static const char* TAG = "snpe_jni";
}

bool feedNetworkInput(std::unordered_map<std::string, std::vector<uint8_t>>&,
                      void*, const AndroidBitmapInfo&);

void displayOutput(std::unordered_map<std::string, std::vector<uint8_t>>&);

// Java_org_tensorflow_demo_SnpeDemoActivity_snpe_demo_main
extern "C" JNIEXPORT void JNICALL
Java_org_tensorflow_demo_SnpeDemoActivity_snpe_1demo_1main(
    JNIEnv* env, jobject* instance, jstring dlcFilePath, jstring dspRuntimePath,
    jobject jbitmap) {
  {
    LOGI("SNPE Native: Setup DSP runtime ");
    std::string dspRuntimePath_(
        env->GetStringUTFChars(dspRuntimePath, nullptr));
    dspRuntimePath_ +=
        ";/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp";
    if (0 !=
        setenv("ADSP_LIBRARY_PATH", dspRuntimePath_.c_str(), 1 /*override*/)) {
      LOGI("SNPE Native: Failed at setup env variable");
      return;
    }
    LOGI("SNPE Native: Setup DSP runtime Done");
  }

  LOGI("SNPE Native: Check runtime");
  zdl::DlSystem::Runtime_t rt = checkRuntime();
  LOGI("SNPE Native: choose %s as backend",
       rt == zdl::DlSystem::Runtime_t::GPU
           ? "GPU"
           : zdl::DlSystem::Runtime_t::DSP == rt ? "DSP" : "CPU");
  LOGI("SNPE Native: Check runtime done");

  // todo: memory leak & codec issue
  std::string dlcFName(env->GetStringUTFChars(dlcFilePath, nullptr));
  LOGI("SNPE Native: Load DLC file from %s", dlcFName.c_str());
  std::unique_ptr<zdl::DlContainer::IDlContainer> dlcContainer =
      loadContainerFromFile(dlcFName);
  if (!dlcContainer) {
    LOGE("SNPE Native: load DLF file failed");
  }
  LOGI("SNPE Native: Load DLC file done");

  zdl::DlSystem::TensorShapeMap inputShape;
  {
    // change input size ?
    inputShape.add("Preprocessor/sub:0",
                   zdl::DlSystem::TensorShape({6, 300, 300, 2}));
  }

  LOGI("SNPE Native: Set Network Builder Options");
  std::unique_ptr<zdl::SNPE::SNPE> snpe =
      setBuilderOptions(dlcContainer, rt, zdl::DlSystem::UDLBundle(),
                        true /*useUserSuppliedBuffers*/, inputShape);
  LOGI("SNPE Native: Set Network Builder Options done");

  LOGI("SNPE Native: Create UserBuffer");
  zdl::DlSystem::UserBufferMap inputMap, outputMap;
  std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>
      snpeUserBackedInputBuffers, snpeUserBackedOutputBuffers;
  std::unordered_map<std::string, std::vector<uint8_t>> applicationInputBuffers,
      applicationOutputBuffers;
  {
    createInputBufferMap(inputMap, applicationInputBuffers,
                         snpeUserBackedInputBuffers, snpe);
    createOutputBufferMap(outputMap, applicationOutputBuffers,
                          snpeUserBackedOutputBuffers, snpe);
    for (auto itor = snpeUserBackedInputBuffers.cbegin();
         itor != snpeUserBackedInputBuffers.cend(); ++itor) {
      LOGI("SNPE Native: snpeUserBackedOutputBuffers[].size=%d(bytes)",
           (*itor)->getSize());
    }

    for (auto itor = snpeUserBackedOutputBuffers.cbegin();
         itor != snpeUserBackedOutputBuffers.cend(); ++itor) {
      LOGI("SNPE Native: snpeUserBackedOutputBuffers[].size=%d(bytes)",
           (*itor)->getSize());
    }
    //   std::for_each(snpeUserBackedInputBuffers.cbegin(),
    //   snpeUserBackedInputBuffers.cend(), [])

    for (auto itor = applicationInputBuffers.cbegin();
         itor != applicationInputBuffers.cend(); ++itor) {
      LOGI("SNPE Native: input buffer name = %s, buf_size = %d",
           itor->first.c_str(), itor->second.size());
    }

    for (auto itor = applicationOutputBuffers.cbegin();
         itor != applicationOutputBuffers.cend(); ++itor) {
      LOGI("SNPE Native: output buffer name = %s, buf_size = %d",
           itor->first.c_str(), itor->second.size());
    }
    LOGI("SNPE Native: Create UserBuffer done");
  }

  LOGI("SNPE Native: Feed In data");
  {
    void* framebuf_addr = nullptr;
    AndroidBitmap_lockPixels(env, jbitmap, &framebuf_addr);
    AndroidBitmapInfo bmpInfo;
    AndroidBitmap_getInfo(env, jbitmap, &bmpInfo);
    LOGI("SNPE Native: bmpInfo(w=%d,h=%d,stride = %u(Bytes/Row), pixel_fmt=%s",
         bmpInfo.width, bmpInfo.height, bmpInfo.stride,
         bmpInfo.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ? "RGBA_8888"
                                                           : "Other");
    if (framebuf_addr == nullptr) {
      LOGE("SNPE Native: can not lock frame  buffer");
      return;
    }

    feedNetworkInput(applicationInputBuffers,
                     /*buffer_addr*/ framebuf_addr, bmpInfo);

    AndroidBitmap_unlockPixels(env, jbitmap);
  }

  {
    LOGI("SNPE Native: execute NN inference");
    executeNetwork(snpe, inputMap, outputMap, applicationOutputBuffers,
                   std::string(""), 1);
    LOGI("SNPE Native: execute NN inference Done");
  }

  displayOutput(applicationOutputBuffers);

  LOGI("SNPE Native: Feed In data done");
}

// todo-shape not match
bool feedNetworkInput(
    std::unordered_map<std::string, std::vector<uint8_t>>& appBuffer,
    void* bitmap, const AndroidBitmapInfo& bmpInfo) {
  // todo
  for (auto itor = appBuffer.begin(); itor != appBuffer.end(); ++itor) {
    auto& vecBuf = itor->second;
    auto vecBuf_Size = vecBuf.size();
    decltype(bmpInfo.width) bmpBufSizeInBytes =
        (bmpInfo.stride / bmpInfo.width) * bmpInfo.width * bmpInfo.height;

    LOGI("SNPE Native: NN input buf size = %d", vecBuf_Size);
    LOGI("SNPE Native: BMP size = %d", bmpBufSizeInBytes);
    auto repeatCnt = vecBuf_Size / bmpBufSizeInBytes;
    auto vecBuf_Ptr = &vecBuf[0];
    while (repeatCnt) {
      LOGI("SNPE Native: fill NN Input by chunks");
      memcpy(vecBuf_Ptr, (decltype(vecBuf_Ptr))bitmap, bmpBufSizeInBytes);
      --repeatCnt;
      vecBuf_Ptr += bmpBufSizeInBytes;
    }
    auto restRoom = vecBuf_Size % bmpBufSizeInBytes;
    if (0 != restRoom) {
      LOGI("SNPE Native: copy %d byte to NN input buf", restRoom);
      memcpy(vecBuf_Ptr, (decltype(vecBuf_Ptr))bitmap, restRoom);
    }
  }

  LOGI("SNPE Native:%s done", __FUNCTION__);

  return false;
}

void displayOutput(
    std::unordered_map<std::string, std::vector<uint8_t>>& outputMap) {
  std::for_each(
      outputMap.begin(), outputMap.end(),
      [&](const std::pair<std::string, std::vector<uint8_t>> outPair) {
        // todo
      });
}