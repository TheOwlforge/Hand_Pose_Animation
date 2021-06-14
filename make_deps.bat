git submodule init
cmake -G "Visual Studio 16 2019" -S deps\OpenCV -B deps\OpenCV\build -D BUILD_SHARED_LIBS=OFF -D BUILD_opencv_world=ON
cmake --build deps\OpenCV\build --config "Debug" 
cmake --install deps\OpenCV\build --config "Debug" --prefix deps\OpenCV\install
cmake --build deps\OpenCV\build --config "Release" 
cmake --install deps\OpenCV\build --config "Release" --prefix deps\OpenCV\install

PAUSE