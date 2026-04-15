cmake_minimum_required(VERSION 3.0)

set(CXXFLAGS "-lgpiod -std=c++17 -I/usr/include/opencv4/opencv -I/usr/include/opencv4 -lopencv_gapi -lopencv_stitching -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_dpm -lopencv_face -lopencv_fuzzy -lopencv_hfs -lopencv_img_hash -lopencv_line_descriptor -lopencv_quality -lopencv_reg -lopencv_rgbd -lopencv_saliency -lopencv_sfm -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrapping -lopencv_superres -lopencv_optflow -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_plot -lopencv_videostab -lopencv_video -lopencv_xfeatures2d -lopencv_shape -lopencv_ml -lopencv_ximgproc -lopencv_xobjdetect -lopencv_objdetect -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_flann -lopencv_xphoto -lopencv_photo -lopencv_imgproc -lopencv_core -I/usr/include/c++/8.3.0/riscv64-poky-linux")
set(CMAKE_CXX_FLAGS "${CXXFLAGS}")
set(BinName "tflm")
add_executable(${BinName} "")

target_link_libraries(
  ${BinName}
  "generic-tflmicro"
)

file(GLOB_RECURSE SRC ${CMAKE_CURRENT_LIST_DIR}/*.c ${CMAKE_CURRENT_LIST_DIR}/*.cpp ${CMAKE_CURRENT_LIST_DIR}/*.cc ${CMAKE_CURRENT_LIST_DIR}/*.cxx ${CMAKE_CURRENT_LIST_DIR}/*.h ${CMAKE_CURRENT_LIST_DIR}/*.hpp)

target_sources(${BinName}
  PRIVATE
  ${SRC}
)
