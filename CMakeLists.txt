cmake_minimum_required(VERSION 3.10)
project(Amora)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fopenmp")

add_library(simplepir STATIC
    simplepir/pir_types.cpp
    simplepir/pir_math.cpp
    simplepir/random.cpp
    simplepir/matrix.cpp
    simplepir/simple_pir.cpp
    simplepir/embedding_pir.cpp
    simplepir/neighbor_pir.cpp
)

target_include_directories(simplepir PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/simplepir
)

target_link_libraries(simplepir PUBLIC pthread gomp)

add_library(graph_partitioner STATIC
    src/graph_partitioner.cpp
)

target_include_directories(graph_partitioner PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(graph_partitioner PUBLIC pthread gomp)

add_library(clustering STATIC
    src/clustering/clustering.cpp
)

target_include_directories(clustering PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/clustering
)

target_link_libraries(clustering PUBLIC pthread gomp graph_partitioner)

option(USE_MTMETIS "Use MT-METIS for multi-threaded graph partitioning" ON)
if(USE_MTMETIS)
    set(MTMETIS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/mt-metis-0.7.2)
    set(MTMETIS_LIB ${MTMETIS_DIR}/build/Linux-x86_64/lib/libmtmetis.a)
    if(EXISTS ${MTMETIS_LIB})
        target_include_directories(clustering PUBLIC ${MTMETIS_DIR}/include)
        target_compile_definitions(clustering PUBLIC USE_MTMETIS)
        target_link_libraries(clustering PUBLIC ${MTMETIS_LIB} m)
        message(STATUS "MT-METIS enabled: ${MTMETIS_DIR}")
    else()
        message(WARNING "MT-METIS not found at ${MTMETIS_LIB}, using built-in partitioner")
    endif()
endif()

add_library(subgroup STATIC
    src/subgroup/subgroup_manager.cpp
)

target_include_directories(subgroup PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/subgroup
    ${CMAKE_CURRENT_SOURCE_DIR}/include/clustering
)

target_link_libraries(subgroup PUBLIC pthread gomp)

add_library(hnsw_quantized STATIC
    src/hnsw_quantized.cpp
    src/io.cpp
)

target_include_directories(hnsw_quantized PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/clustering
    ${CMAKE_CURRENT_SOURCE_DIR}/include/subgroup
)

target_link_libraries(hnsw_quantized PUBLIC pthread gomp graph_partitioner clustering subgroup)

option(USE_HNSWLIB "Use hnswlib for fast graph building" ON)
if(USE_HNSWLIB)
    set(HNSWLIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/hnswlib)
    if(EXISTS ${HNSWLIB_DIR}/hnswlib/hnswlib.h)
        target_include_directories(hnsw_quantized PUBLIC ${HNSWLIB_DIR})
        target_compile_definitions(hnsw_quantized PUBLIC USE_HNSWLIB)
        message(STATUS "hnswlib enabled: ${HNSWLIB_DIR}")
    else()
        message(WARNING "hnswlib not found at ${HNSWLIB_DIR}, building without it")
    endif()
endif()

add_library(hnsw_private_v2 STATIC
    src/private_hnsw_v2.cpp
)

target_include_directories(hnsw_private_v2 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/clustering
    ${CMAKE_CURRENT_SOURCE_DIR}/include/subgroup
    ${CMAKE_CURRENT_SOURCE_DIR}/simplepir
)

target_link_libraries(hnsw_private_v2 PUBLIC hnsw_quantized simplepir)

add_executable(build_index_quantized examples/build_index_quantized.cpp)
target_link_libraries(build_index_quantized hnsw_quantized)

add_executable(private_search_v2 examples/private_search_v2.cpp)
target_link_libraries(private_search_v2 hnsw_private_v2)
