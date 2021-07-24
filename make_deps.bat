if exist .\deps\OpenCV\build\ (
    echo Skipping OpenCV build - folder already exists.
) else (
    cmake -G "Visual Studio 16 2019" -S .\deps\OpenCV -B .\deps\OpenCV\build -D BUILD_SHARED_LIBS=ON -D BUILD_opencv_world=ON
    cmake --build .\deps\OpenCV\build --config "Debug"
    cmake --build .\deps\OpenCV\build --config "Release"
)

if exist .\deps\OpenCV\install\ (
    echo Skipping OpenCV install - folder already exists.
) else (
    cmake --install .\deps\OpenCV\build --config "Debug" --prefix .\deps\OpenCV\install
    cmake --install .\deps\OpenCV\build --config "Release" --prefix .\deps\OpenCV\install 
)


if exist .\deps\OpenPose\build\ (
    echo Skipping OpenPose build - folder already exists.
) else (
    where /q nvcc --version
    IF ERRORLEVEL 1 (
        cmake -G "Visual Studio 16 2019" -S .\deps\Openpose -B .\deps\OpenPose\build -D GPU_MODE=CPU_ONLY -D BUILD_EXAMPLES=OFF
    ) else (
        cmake -G "Visual Studio 16 2019" -S .\deps\Openpose -B .\deps\OpenPose\build -D GPU_MODE=CUDA -D BUILD_EXAMPLES=OFF
    )
    cmake --build .\deps\OpenPose\build --config "Debug" 
    cmake --build .\deps\OpenPose\build --config "Release"
)

PAUSE