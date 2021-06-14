git submodule init

mkdir deps/OpenCV/build
mkdir deps/OpenCV/install
cd deps/OpenCV/build

# Debug
cmake .. -G "Unix Makefiles" -DBUILD_SHARED_LIBS=OFF -DBUILD_opencv_world=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=../install -DOPENCV_BUILD_3RDPARTY_LIBS=ON
make
make install

# Release
cmake .. -G "Unix Makefiles" -DBUILD_SHARED_LIBS=OFF -DBUILD_opencv_world=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install -DOPENCV_BUILD_3RDPARTY_LIBS=ON
make
make install