# generic-tflmicro
- This repository provides:
	- cmake project to generate TensorFlow Lite for Microcontrollers ( TFLM ) as a library
		- You can easily use TFLM in your project by adding this repository as git-submodule
	- sample project (hello world project) to use the TFLM library

- The TFLM library in this repo doesn't have platform-specific hardware optimizations. So the library should run on any platform
	- I made this repository mainly for development on PC (Windows and Linux on x64)
	- When you use a specific hardware like Arduino, it' better to create a new library for the device to get better performance

## How to build
### PC Linux (x64)
```sh
mkdir -p build && cd build/
cmake ..
make -j8
./examples/hello_world/hello_world 
```
### Windows (Visual Studio 2019) (x64)
- Use cmake-gui

## How to use in your project
Refer to [examples/hello_world/CMakeLists.txt](examples/hello_world/CMakeLists.txt)

## How this repository was modified (2024/9 updated)
- Clone code and generate all projects
	- Refer to: https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/docs/new_platform_support.md
```sh
git clone https://github.com/tensorflow/tflite-micro
cd tflite-micro

python3 tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py \
  -e hello_world \
  -e micro_speech \
  -e person_detection \
  /tmp/tflm-tree
```
- This will create a folder that looks like the following at the top-level:
```
examples  LICENSE  tensorflow  third_party
```
- Update `tensorflow` and `third_party` to source
- Update `hello_world` example
