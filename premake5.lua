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

    filter "system:linux"
        libdirs
        {
            OPENCV_DIR .. CV_LIB .. "/opencv4/3rdparty"
        }
    filter {} --reset

    includedirs 
    {
        "src",
        OPENCV_DIR .. CV_INC,
    }
   
    libdirs
    {
        OPENCV_DIR .. CV_LIB,
    }

    files { "src/**.h", "src/**.cpp"}

    pchheader "pch.h"
    pchsource "src/pch.cpp"

    filter {"configurations:Debug"}
        defines { "DEBUG", "_DEBUG" }
        symbols "On"

    filter {"configurations:Release"}
        defines { "NDEBUG" }
        optimize "On"

    filter {"system:windows", "configurations:Debug"}
        links { "opencv_world" .. OPENCV_VERSION .. "d", "aded", "IlmImfd", "ippicvmt", "ippiwd", "ittnotifyd", "libjpeg-turbod", "libopenjp2d", "libpngd", "libprotobufd", "libtiffd", "libwebpd", "quircd", "zlibd" }
        postbuildcommands
        {
            -- ("{COPY} " .. OPENCV_DIR .. "x64/vc16/bin/opencv_world" .. OPENCV_VERSION .. "d.dll bin/" .. outdir),
        }

    filter {"system:windows", "configurations:Release"}
        links { "opencv_world" .. OPENCV_VERSION, "ade", "IlmImf", "ippicvmt", "ippiw", "ittnotify", "libjpeg-turbo", "libopenjp2", "libpng", "libprotobuf", "libtiff", "libwebp", "quirc", "zlib" }
        postbuildcommands
        {
            -- ("{COPY} " .. OPENCV_DIR .. "x64/vc15/bin/opencv_world" .. OPENCV_VERSION .. ".dll bin/" .. outdir),
        }
 
    filter {"system:linux"}
        links {"opencv_world", "dl", "pthread", "ade", "ippiw", "ippicv", "ittnotify", "libopenjp2", "libprotobuf", "libwebp", "quirc", "z", "IlmImf", "tiff", "png", "jpeg"}
