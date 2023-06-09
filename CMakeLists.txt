cmake_minimum_required(VERSION 2.8)
project(da3d)

# The build type "Release" adds some optimizations
if (NOT CMAKE_BUILD_TYPE)
  set (CMAKE_BUILD_TYPE "Release")
endif ()

# Are we using gcc?
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # GCC on MacOs needs this option to use the clang assembler
  if (APPLE)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wa,-q")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wa,-q")
  endif ()
  # Optimize to the current CPU and enable warnings
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -Wall -Wextra -pg")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wall -Wextra -pg")
endif ()

# Enable C99
if (CMAKE_VERSION VERSION_LESS "3.1")
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
else ()
  set (CMAKE_C_STANDARD 99)
endif ()

# Enable OpenMP
 find_package (OpenMP)
 if(OPENMP_FOUND)
   set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
   set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
 endif()

# Link LibTIFF, LibJPEG, LibPNG, OpenMP
find_package (TIFF REQUIRED)
find_package (JPEG REQUIRED)
find_package (PNG REQUIRED)


include_directories (PUBLIC ${EIGEN3_INCLUDE_DIR} PUBLIC ${TIFF_INCLUDE_DIR} PUBLIC ${JPEG_INCLUDE_DIR} PUBLIC ${PNG_INCLUDE_DIRS} PUBLIC ${ALGLIB_INCLUDE_DIR} )
link_libraries (${TIFF_LIBRARIES} ${JPEG_LIBRARIES} ${PNG_LIBRARIES})

set(SOURCE_FILES
    bm3d.cpp           iio.c              lib_transforms.h   utilities.h
    bm3d.h             lib_transforms.cpp main.cpp           utilities.cpp
    )

set_property(SOURCE iio.c PROPERTY COMPILE_FLAGS "-Wno-unused-variable -Wno-unused-parameter -Wno-pointer-sign -Wno-parentheses -Wno-deprecated-declarations -Wno-unused-function")

add_executable(bm3d ${SOURCE_FILES})
