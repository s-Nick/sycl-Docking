set(CMAKE_CXX_COMPILER $ENV{DPCPP_HOME}/clang++)

cmake_minimum_required(VERSION 3.18)

project("sycl-Docking" LANGUAGES CXX)

find_package(Boost REQUIRED COMPONENTS program_options)
set(CMAKE_PREFIX_PATH $ENV{RDBASE}/../)
find_package(RDKit REQUIRED)

add_library(utils STATIC src/distance.cpp src/graph.cpp src/rotamer.cpp src/rotation.cpp )
target_include_directories(utils PUBLIC $ENV{DPCPP_HOME}/../include/
                                        $ENV{DPCPP_HOME}/../include/sycl
                                        $ENV{RDBASE}/Code 
                                        $ENV{RDBASE}/External
                                        $ENV{BOOST_ROOT}/include
                        )
link_directories($ENV{RDBASE}/lib $ENV{BOOST_ROOT}/lib $ENV{DPCPP_HOME}/../install/lib )
target_link_libraries(utils -lRDKitChemReactions -lRDKitFileParsers -lRDKitSmilesParse -lRDKitDepictor  -lRDKitSubstructMatch -lRDKitGraphMol -lRDKitDataStructs -lRDKitRDGeometryLib -lRDKitRDGeneral ${Boost_LIBRARIES})
#target_link_libraries(utils PRIVATE ${Boost_LIBRARIES})
add_executable(main src/main.cpp)

target_link_libraries(main utils Boost::program_options)
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_LIST_DIR} CACHE STRING "" FORCE)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_FLAGS "-fsycl -fsycl-targets=nvptx64-nvidia-cuda" CACHE STRING "" FORCE  )
install(TARGETS main
        CONFIGURATIONS ${CMAKE_BUILD_TYPE}
        RUNTIME DESTINATION
        )
