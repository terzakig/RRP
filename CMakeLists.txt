cmake_minimum_required(VERSION 3.2)

project(ratfit_example)


#find Eigen (TODO: Uncomment for all pertinent entries to use)
#find_package(Eigen3 REQUIRED)

SET(RFEXAMPLE_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/example/src)

set(RATFIT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/rationalfit/src) 
set(RATFIT_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/rationalfit/src) 

#set(EIGEN34_DIR ${RATFIT_SRC_DIR}/eigen3x4)

#set(EIGEN34_HEADERS
#    ${EIGEN34_DIR}/EigenDecompose.h
#    ${EIGEN34_DIR}/PolySolvers.h
#)

set(RATFIT_HEADERS 
${RAT_FIT_SRC_DIR}/RationalFitter.h
${RAT_FIT_SRC_DIR}/minialg.h)

#set(EIGEN34_SOURCES ${EIGEN34_HEADERS})
set(RATFIT_SOURCES ${RATFIT_HEADERS})
    
set(ALL_SOURCES 
   ${RAT_FIT_SRC_SOURCES}
#   ${EIGEN34_SOURCES}
   ${RFEXAMPLE_SRC_DIR}/main.cpp
)

#All include directories
include_directories(
${RATFIT_INCLUDE_DIR}
${RFEXAMPLE_SRC_DIR} 
#${EIGEN3_INCLUDE_DIR}
)


# The example executable     
add_executable(${PROJECT_NAME} ${ALL_SOURCES})
                      
