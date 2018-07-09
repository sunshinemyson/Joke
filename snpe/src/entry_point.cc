#include <iostream>
#include <string>
#include <jni.h>
//#include <bitmap.h>
#include <unordered_map>

#include "android_log.hpp"
#include "CheckRuntime.hpp"
#include "Util.hpp"
#include "LoadContainer.hpp"
#include "SetBuilderOptions.hpp"
#include "CreateUserBuffer.hpp"

namespace {
    static const char* TAG = "snpe_jni";
}

int _fake_main(int argc, char* argv[]) {

    zdl::DlSystem::Runtime_t rt = checkRuntime();

    std::cout<<"Fake_main"<<std::endl;
    return 0;
}

// Java_org_tensorflow_demo_SnpeDemoActivity_snpe_demo_main
extern "C"
JNIEXPORT void JNICALL
Java_org_tensorflow_demo_SnpeDemoActivity_snpe_1demo_1main(JNIEnv* env,
                                                         jobject* instance,
                                                         jstring dlcFilePath,
                                                         jobject jbitmap
                                                         )
{
    LOGI("%s:%s:line:%d", __FILE__, __FUNCTION__, __LINE__);
    LOGI("Going to Read DLC file from %s", env->GetStringUTFChars(dlcFilePath, nullptr));

    LOGI("SNPE Native: Check runtime");
    zdl::DlSystem::Runtime_t rt = checkRuntime();
    LOGI("SNPE Native: choose %s as backend", rt == zdl::DlSystem::Runtime_t::GPU ? "GPU" : zdl::DlSystem::Runtime_t::DSP == rt ? "DSP" : "CPU");
    LOGI("SNPE Native: Check runtime done");

    // todo: memory leak & codec issue
    std::string dlcFName(env->GetStringUTFChars(dlcFilePath, nullptr));
    LOGI("SNPE Native: Load DLC file from %s", dlcFName.c_str());
    std::unique_ptr<zdl::DlContainer::IDlContainer> dlcContainer = loadContainerFromFile(dlcFName);
    if (!dlcContainer) {
        LOGE("SNPE Native: load DLF file failed");
    }
    LOGI("SNPE Native: Load DLC file done");

    LOGI("SNPE Native: Set Network Builder Options");
    std::unique_ptr<zdl::SNPE::SNPE> builder = setBuilderOptions(dlcContainer,
                                                                rt,
                                                                zdl::DlSystem::UDLBundle(),
                                                                true
                                                                );
    LOGI("SNPE Native: Set Network Builder Options done");

    LOGI("SNPE Native: Create UserBuffer");
    zdl::DlSystem::UserBufferMap inputMap, outputMap;
    std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>> snpeUserBackedInputBuffers, snpeUserBackedOutputBuffers;
    std::unordered_map<std::string, std::vector<uint8_t>> applicationInputBuffers, applicationOutputBuffers;

    createInputBufferMap(inputMap, applicationInputBuffers, snpeUserBackedInputBuffers, builder);
    createOutputBufferMap(outputMap, applicationOutputBuffers, snpeUserBackedOutputBuffers, builder);

    for (auto itor = applicationInputBuffers.cbegin(); itor != applicationInputBuffers.cend(); ++ itor) {
        LOGI("SNPE Native: input buffer name = %s, buf_size = %d", itor->first.c_str(), itor->second.size());
    }
    for (auto itor = applicationOutputBuffers.cbegin(); itor != applicationOutputBuffers.cend(); ++ itor) {
        LOGI("SNPE Native: output buffer name = %s, buf_size = %d", itor->first.c_str(), itor->second.size());
    }
    LOGI("SNPE Native: Create UserBuffer done");

    LOGI("SNPE Native: Feed In data");
    void * framebuf_addr = nullptr;
    //AndroidBitmap_lockPixels(env, jbitmap, &framebuf_addr);
    // if (*fremebuf_add == nullptr) {
    //     LOGE("SNPE Native: can not lock frame  buffer");
    //     return;
    // }
    //AndroidBitmap_unlockPixels(env, jbitmap);

    LOGI("SNPE Native: Feed In data done");

}
