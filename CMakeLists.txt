cmake_minimum_required(VERSION 3.20)
project(iro-cuda-ax LANGUAGES CXX)

include(GNUInstallDirs)

add_library(iro_cuda_ax_headers INTERFACE)
add_library(iro-cuda-ax::headers ALIAS iro_cuda_ax_headers)

target_compile_features(iro_cuda_ax_headers INTERFACE cxx_std_20)
target_include_directories(iro_cuda_ax_headers
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

option(AX_ENABLE_INTERNAL_COMPILE_TARGET "Enable internal manifest-driven instantiation compile target" OFF)
if(AX_ENABLE_INTERNAL_COMPILE_TARGET)
  add_custom_target(ax_internal_compile
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/compile.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running internal AX architecture compile validation"
    VERBATIM)
endif()

install(TARGETS iro_cuda_ax_headers EXPORT iro_cuda_ax_targets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT iro_cuda_ax_targets
  FILE iro-cuda-axTargets.cmake
  NAMESPACE iro-cuda-ax::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/iro-cuda-ax)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/iro-cuda-axConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/iro-cuda-axConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/iro-cuda-ax)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/iro-cuda-axConfigVersion.cmake
  VERSION 0.2.4
  COMPATIBILITY ExactVersion)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/iro-cuda-axConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/iro-cuda-axConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/iro-cuda-ax)
