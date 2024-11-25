# Copyright (c) 2005-2024, University of Oxford.
# All rights reserved.
# 
# University of Oxford means the Chancellor, Masters and Scholars of the
# University of Oxford, having an administrative office at Wellington
# Square, Oxford OX1 2JD, UK.
# 
# This file is part of Chaste.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#  * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  * Neither the name of the University of Oxford nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

if(NOT Chaste_ENABLE_PYCHASTE)
    return()
endif()

find_package(Chaste COMPONENTS ${Chaste_DEPENDS_pychaste})

chaste_do_component(pychaste)

##################################################
# Copy Python package to build directory
##################################################
set(PYCHASTE_PACKAGE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/package)

set(file_paths "")
file(
    GLOB_RECURSE
    file_paths
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/src/py
    ${CMAKE_CURRENT_SOURCE_DIR}/src/py/*
)
list(FILTER file_paths EXCLUDE REGEX ".*\\.(dll|dylib|pyc|so)$") # Remove compiled files

foreach(file_path ${file_paths})
    if(file_path MATCHES ".*\\.(cfg|in|js|py|toml)$") # Files to track changes on
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/src/py/${file_path}
            ${PYCHASTE_PACKAGE_BINARY_DIR}/${file_path}
            COPYONLY
        )
    else()
        file(
            COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/py/${file_path}
            DESTINATION ${PYCHASTE_PACKAGE_BINARY_DIR}/
        )
    endif()
endforeach()

##################################################
# Build Python modules from wrappers
##################################################
# Create a `_pychaste_all` shared library. The library name is the same
# as the module name in the module wrapper `_pychaste_all.main.cppwg.cpp`.
# It is imported in Python code as `from chaste._pychaste_all import *`.

header_dirs("${CMAKE_CURRENT_SOURCE_DIR}/dynamic" PYCHASTE_WRAPPER_INCLUDE_DIRS)
set(PYCHASTE_WRAPPER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dynamic/wrappers")

file(
    GLOB_RECURSE
    PYCHASTE_WRAPPER_SOURCES
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    ${PYCHASTE_WRAPPER_DIR}/all/*.cpp
    ${PYCHASTE_WRAPPER_DIR}/all/*.hpp
)

add_library(_pychaste_all SHARED
    "${PYCHASTE_WRAPPER_DIR}/all/_pychaste_all.main.cppwg.cpp"
    ${PYCHASTE_WRAPPER_SOURCES}
)

set_target_properties(
    _pychaste_all
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${PYCHASTE_PACKAGE_BINARY_DIR}/chaste
        PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${CMAKE_SHARED_LIBRARY_SUFFIX}"
)

target_link_libraries(
    _pychaste_all
    PRIVATE
    ${Python3_LIBRARIES}  # Python and pybind11 come first
    pybind11::module
    chaste_pychaste  # Library built from non-wrapper code in src/cpp/
)

target_include_directories(
    _pychaste_all
    PRIVATE
    ${PYTHON3_INCLUDE_DIRS}
    ${Chaste_pychaste_INCLUDE_DIRS}  # Non-wrapper code in src/cpp/
    ${PYCHASTE_WRAPPER_INCLUDE_DIRS}
    ${PETSC4PY_INCLUDES}
)

target_compile_options(_pychaste_all PRIVATE -Wno-unused-local-typedefs)

add_dependencies(pychaste _pychaste_all)

##################################################
# Python wrapper generation
##################################################

add_custom_target(pychaste_wrappers
    COMMAND ${chaste_python3_venv}/cppwg ${CMAKE_SOURCE_DIR}
            -w ${PYCHASTE_WRAPPER_DIR}
            -p ${CMAKE_SOURCE_DIR}/pychaste/dynamic/config.yaml
            -i "$<JOIN:$<TARGET_PROPERTY:chaste_pychaste,INCLUDE_DIRECTORIES>,;>"
            -l ${CMAKE_CURRENT_BINARY_DIR}/cppwg.log
            --std c++17
    COMMAND_EXPAND_LISTS
)
