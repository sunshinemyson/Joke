#include <iostream>
#include "CheckRuntime.hpp"
#include "Util.hpp"

#include <jni.h>
#include "android_log.hpp"

namespace {
    static const char* TAG = "Snpe_jni";
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
                                                         jobject* instance)
{
    LOGI("%s:%s:line:%d", __FILE__, __FUNCTION__, __LINE__);
}
//Java_org_tensorflow_demo_SnpeDemoActivity_snpe_1demo_1main