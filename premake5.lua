workspace "HandPose"
    architecture "x64"
    staticruntime "on"
    flags { "MultiProcessorCompile" }
    configurations { "Debug", "Release" }

project "HandPose"
    kind "ConsoleApp"
    language "C++"
    cppdialect "C++17"
    systemversion "latest"

    outdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"
    targetdir ("bin/" .. outdir)
    objdir ("bin-int/" .. outdir)

    OPENCV_DIR = "deps/OpenCV/install/"
    OPENCV_VERSION = "453"
    CV_LIB = iif(os.istarget("windows"), "x64/vc16/staticlib", "lib")
    CV_INC = iif(os.istarget("windows"), "include", "include/opencv4")
    OPENPOSE_DIR = "deps/OpenPose/"
    EIGEN_DIR = "deps/Eigen/"
    CERES_DIR = "deps/Ceres/"

    filter "system:linux"
        libdirs
        {
            OPENCV_DIR .. CV_LIB .. "/opencv4/3rdparty"
        }
    filter "system:windows"
        includedirs
        {
            OPENPOSE_DIR .. "3rdparty/windows/caffe/include",
            OPENPOSE_DIR .. "3rdparty/windows/caffe_cpu/include",
            OPENPOSE_DIR .. "3rdparty/windows/caffe3rdparty/include"
        }
        libdirs
        {
            OPENPOSE_DIR .. "3rdparty/windows/caffe/lib",
            OPENPOSE_DIR .. "3rdparty/windows/caffe_cpu/lib",
            OPENPOSE_DIR .. "3rdparty/windows/caffe3rdparty/lib"
        }
    filter {} --reset

    includedirs 
    {
        "src",
        OPENCV_DIR .. CV_INC,
        OPENPOSE_DIR .. "include",
        EIGEN_DIR,
	CERES_DIR .. "include"
    }
   
    libdirs
    {
        OPENCV_DIR .. CV_LIB,
	CERES_DIR .. "lib"
    }

    files { "src/**.h", "src/**.hpp", "src/**.cpp"}

    pchheader "pch.h"
    pchsource "src/pch.cpp"

    filter {"configurations:Debug"}
        defines { "DEBUG", "_DEBUG" }
        symbols "On"
	libdirs
        {
	    OPENPOSE_DIR .. "build/src/openpose/Debug"
        }

    filter {"configurations:Release"}
        defines { "NDEBUG" }
        optimize "On"
	libdirs
        {
	    OPENPOSE_DIR .. "build/src/openpose/Release"
        }

    filter {"system:windows"}
        postbuildcommands
        {
            ("{COPY} " .. OPENPOSE_DIR .. "build/bin/ bin/" .. outdir),
        }

    filter {"system:windows", "configurations:Debug"}
        links
        { 
            "opencv_world" .. OPENCV_VERSION .. "d",
            "aded", "IlmImfd", "ippicvmt", "ippiwd", "ittnotifyd", "libjpeg-turbod", "libopenjp2d", "libpngd", "libprotobufd", "libtiffd", "libwebpd", "quircd", "zlibd",
            "openposed", "caffe-d", "caffeproto-d", "snappyd", "lmdbd", "glogd", "gflagsd", "caffezlibd", "caffehdf5_D", "caffehdf5_hl_D",
            "boost_date_time-vc142-mt-gd-x64-1_74", "boost_filesystem-vc142-mt-gd-x64-1_74", "boost_system-vc142-mt-gd-x64-1_74", "boost_thread-vc142-mt-gd-x64-1_74",
	    "ceres-debug"
        }
        postbuildcommands
        {
            ("{COPY} " .. OPENPOSE_DIR .. "build/x64/Debug/openposed.dll bin/" .. outdir),
            ("{COPY} deps/misc/ bin/" .. outdir),
        }

    filter {"system:windows", "configurations:Release"}
        links 
        { 
            "opencv_world" .. OPENCV_VERSION,
            "ade", "IlmImf", "ippicvmt", "ippiw", "ittnotify", "libjpeg-turbo", "libopenjp2", "libpng", "libprotobuf", "libtiff", "libwebp", "quirc", "zlib",
            "openpose", "caffe", "caffeproto", "snappy", "lmdb", "glog", "gflags", "caffezlib", "caffehdf5", "caffehdf5_hl",
            "boost_date_time-vc142-mt-x64-1_74", "boost_filesystem-vc142-mt-x64-1_74", "boost_system-vc142-mt-x64-1_74", "boost_thread-vc142-mt-x64-1_74",
	    "ceres"
        }
        postbuildcommands
        {
            ("{COPY} " .. OPENPOSE_DIR .. "build/x64/Release/openpose.dll bin/" .. outdir),
        }
 
    filter {"system:linux"}
        links {"opencv_world", "dl", "pthread", "ade", "ippiw", "ippicv", "ittnotify", "libopenjp2", "libprotobuf", "libwebp", "quirc", "z", "IlmImf", "tiff", "png", "jpeg"}
