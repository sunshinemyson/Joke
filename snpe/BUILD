# Description:
#   TensorFlow camera demo app for Android.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
    "tf_opts_nortti_if_android",
)

exports_files(["LICENSE"])

LINKER_SCRIPT = "//tensorflow/contrib/android:jni/version_script.lds"

cc_binary(
    name = "libsnpe_demo.so",
    srcs = glob([
	"src/*.hpp",
        "src/*.cpp",
        "src/*.h",
        "src/*.cc",
    ]),
    includes = [
        "include/zdl",
    ],
    copts = tf_copts() + ["-fexceptions"],
    linkopts = [
        "-landroid",
        "-ldl",
        "-ljnigraphics",
        "-llog",
        "-lm",
        "-z defs",
        "-s",
        "-Wl,--version-script",  # This line must be directly followed by LINKER_SCRIPT.
        LINKER_SCRIPT,
    ],
    linkshared = 1,
    linkstatic = 1,
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        LINKER_SCRIPT,
        ":snpe",
        ":snpe_adsp",
        ":symphony-cpu",
        ":symphonypower",
        ":stdcpp",
    ],
)

cc_import(
    name = "snpe",
    shared_library = "lib/arm-android-gcc4.9/libSNPE.so"
)
cc_import(
    name = "snpe_adsp",
    shared_library = "lib/arm-android-gcc4.9/libsnpe_adsp.so"
)

cc_import(
    name = "symphony-cpu",
    shared_library = "lib/arm-android-gcc4.9/libsymphony-cpu.so"
)

cc_import(
    name = "symphonypower",
    shared_library = "lib/arm-android-gcc4.9/libsymphonypower.so"
)

cc_import(
    name = "stdcpp",
    shared_library = "lib/arm-android-gcc4.9/libgnustl_shared.so"
)