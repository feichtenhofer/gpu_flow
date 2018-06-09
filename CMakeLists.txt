cmake_minimum_required(VERSION 2.4)
#set(OpenCV_DIR /home/pedro/opencv-3.1.0/build)
# QT5 insert

  # Tell CMake to run moc when necessary:
  set(CMAKE_AUTOMOC ON)
  # As moc files are generated in the binary dir, tell CMake
  # to always look for includes there:
  set(CMAKE_INCLUDE_CURRENT_DIR ON)

  # Widgets finds its own dependencies (QtGui and QtCore).
  find_package(Qt5Widgets REQUIRED)

  # The Qt5Widgets_INCLUDES also includes the include directories for
  # dependencies QtCore and QtGui
  include_directories(${Qt5Widgets_INCLUDES})

  # We need add -DQT_WIDGETS_LIB when using QtWidgets in Qt 5.
  add_definitions(${Qt5Widgets_DEFINITIONS})

  # Executables fail to build with Qt 5 in the default configuration
  # without -fPIE. We add that here.
  set(CMAKE_CXX_FLAGS "${Qt5Widgets_EXECUTABLE_COMPILE_FLAGS}")

# OpenCV Config
SET(OpenCV_DIR "/home/pedro/opencv-2.4.13.6/build")
find_package( OpenCV REQUIRED )
set(OpenCV_FOUND 1)
message(STATUS "    version: ${OpenCV_VERSION}")
message("-- OPENCV include:   " ${OpenCV_INCLUDE_DIRS})
message("-- OPENCV libs dir:  " ${OpenCV_LIB_DIR})
message("-- OPENCV libs:   " ${OpenCV_LIBS} )

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
LINK_DIRECTORIES(${OpenCV_LIB_DIR})

ADD_DEFINITIONS(-DDEBUG -ggdb -O3 -Wall -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -fbranch-probabilities)

if(COMMAND cmake_policy)
   cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

################################

option(WARP ON)
if(WARP)
    ADD_EXECUTABLE(
      compute_flow_si_warp
      compute_flow_si_warp.cpp
      ${SRCS}
    )

    TARGET_LINK_LIBRARIES(
      compute_flow_si_warp
      ${OPENCV_LIBRARIES}
      ${OpenCV_LIBS}
      ${Qt5Widgets_LIBRARIES}
    )
else()
    ADD_EXECUTABLE(
      compute_flow
      compute_flow.cpp
      ${SRCS}
    )

    TARGET_LINK_LIBRARIES(
      compute_flow
      ${OPENCV_LIBRARIES}
      ${OpenCV_LIBS}
      ${Qt5Widgets_LIBRARIES}
    )
endif()
################################

foreach(dir ${DIRS})
  FILE(GLOB files ${dir}/*.h*)
  LIST(APPEND h_files ${files})
endforeach(dir)
add_custom_target(TAGS etags --members -l c++ ${h_files} ${SRCS})
