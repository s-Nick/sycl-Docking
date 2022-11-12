set(CMAKE_CXX_COMPILER "g++-11")

set(CMAKE_CUDA_ARCHITECTURES 75)

#set(CMAKE_CUDA_FLAGS "-ccbin=g++-11")

cmake_minimum_required(VERSION 3.18)
project("sycl-Docking" LANGUAGES CUDA CXX)

include(CheckLanguage)
check_language(CUDA)

enable_language(CUDA)
find_package(Boost REQUIRED)
set(CMAKE_PREFIX_PATH $ENV{RDBASE}/../)
find_package(RDKit REQUIRED)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()
add_library(utils STATIC src/distance.cu src/graph.cpp src/rotamer.cu src/rotation.cu )
set_target_properties(utils PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(utils PUBLIC $ENV{RDBASE}/Code 
                                        $ENV{RDBASE}/External
                                        $ENV{BOOST_ROOT}/include
                          )


link_directories($ENV{RDBASE}/lib $ENV{BOOST_ROOT}/lib)
target_link_libraries(utils -lRDKitChemReactions -lRDKitFileParsers -lRDKitSmilesParse -lRDKitDepictor  -lRDKitSubstructMatch -lRDKitGraphMol -lRDKitDataStructs -lRDKitRDGeometryLib -lRDKitRDGeneral)
add_executable(main src/main.cu)

target_link_libraries(main utils)

set(CMAKE_INSTALL_PREFIX "/home/p4rs3c/Documents/projects/sycl-Docking/cuda/")
install(TARGETS main
        CONFIGURATIONS Release
        RUNTIME DESTINATION
        )
