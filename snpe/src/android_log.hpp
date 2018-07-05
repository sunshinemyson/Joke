#ifndef __ANDROID_LOG__HPP__
#define __ANDROID_LOG__HPP__
#include <android/log.h>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)
#endif