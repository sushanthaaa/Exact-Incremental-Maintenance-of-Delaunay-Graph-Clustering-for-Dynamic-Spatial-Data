cmake_minimum_required(VERSION 3.16)
project(DelauCluster LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

find_package(CGAL REQUIRED)

add_library(delaucluster_core
  src/DelaunayClusterer.cpp
)
target_include_directories(delaucluster_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(delaucluster_core PUBLIC CGAL::CGAL)

add_executable(cluster
  src/cluster.cpp
)
target_link_libraries(cluster PRIVATE delaucluster_core)

# Same-language incremental-DBSCAN dynamic baseline (no CGAL dependency).
add_executable(incremental_dbscan_stream
  src/incremental_dbscan_stream.cpp
)

message(STATUS "")
message(STATUS "=== DelauCluster build configuration ===")
message(STATUS "  Build type:   ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CGAL version: ${CGAL_VERSION}")
message(STATUS "========================================")
message(STATUS "")
