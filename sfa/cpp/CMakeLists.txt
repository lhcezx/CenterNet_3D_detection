cmake_minimum_required (VERSION 2.6)
project(cpp)

set(CMAKE_CXX_STANDARD 14)
set(BOOST_ROOT "C:/Users/lhcez/Desktop/Code/boost_1_77_0/boost_1_77_0/boost")
include_directories(C:/Users/lhcez/Desktop/Code/boost_1_77_0/boost_1_77_0) # 添加头文件搜索路径
link_directories(C:/Users/lhcez/Desktop/Code/boost_1_77_0/boost_1_77_0/stage/lib)
aux_source_directory(. DIR_SRCS)

add_executable(cpp evaluate_object.cpp)