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

################################
####  Generate Python wrappers
################################

# Get required includes
get_target_property(PYCHASTE_INCLUDE_DIRS chaste_pychaste INCLUDE_DIRECTORIES)

# Get includes for VTK 9+
if(TARGET VTK::CommonCore)
    get_target_property(_include_dirs VTK::CommonCore INTERFACE_INCLUDE_DIRECTORIES)
    list(APPEND PYCHASTE_INCLUDE_DIRS ${_include_dirs})
endif()

# Add typecasters to includes
header_dirs("${CMAKE_CURRENT_SOURCE_DIR}/dynamic" _include_dirs)
list(APPEND PYCHASTE_INCLUDE_DIRS ${_include_dirs})

# Create the wrappers
execute_process(
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/wrappers
    COMMAND ${chaste_python3_venv}/cppwg ${CMAKE_SOURCE_DIR}
        -w ${CMAKE_CURRENT_BINARY_DIR}/wrappers
        -p ${CMAKE_SOURCE_DIR}/pychaste/dynamic/config.yaml
        -i ${PYCHASTE_INCLUDE_DIRS}
        -l ${CMAKE_CURRENT_BINARY_DIR}/cppwg.log
        --std c++17
)

# Add wrappers to includes
header_dirs("${CMAKE_CURRENT_BINARY_DIR}/wrappers" _include_dirs)
list(APPEND PYCHASTE_INCLUDE_DIRS ${_include_dirs})

################################
####  Build Python module
################################
# Creates a `_pychaste_all` shared library from the Python wrappers.
# The library name `_pychaste_all` is the same as the module name in the main 
# module wrapper source file `wrappers/all/_pychaste_all.main.cppwg.cpp`.
# When the shared library has been built, it can be used in Python code 
# as `from chaste._pychaste_all import *`.

file(
    GLOB_RECURSE
    PYCHASTE_WRAPPER_SOURCES
    ${CMAKE_CURRENT_BINARY_DIR}/wrappers/all/*.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/wrappers/all/*.hpp
)

add_library(_pychaste_all SHARED ${PYCHASTE_WRAPPER_SOURCES})

set_target_properties(
    _pychaste_all
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/package/chaste
        PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${CMAKE_SHARED_LIBRARY_SUFFIX}"
)

target_link_libraries(
    _pychaste_all
    PRIVATE
    ${Python3_LIBRARIES}  # Python and pybind11 come first
    pybind11::module
    Chaste_COMMON_DEPS
    chaste_pychaste  # Library built from non-wrapper code in src/cpp/
)

target_include_directories(
    _pychaste_all
    PRIVATE
    ${PYTHON3_INCLUDE_DIRS}
    ${PYCHASTE_INCLUDE_DIRS}
    ${PETSC4PY_INCLUDES}
)

target_compile_options(_pychaste_all PRIVATE -Wno-unused-local-typedefs)

add_dependencies(pychaste _pychaste_all)

################################
####  Copy Python package
################################

# Get list of files in the Python package directory `pychaste/src/py/`
set(package_file_paths "")
file(
    GLOB_RECURSE
    package_file_paths
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/src/py
    ${CMAKE_CURRENT_SOURCE_DIR}/src/py/*
)

# Exclude compiled files
list(FILTER package_file_paths EXCLUDE REGEX ".*\\.(dll|dylib|pyc|so)$")

# Copy files to the build directory
foreach(file_path ${package_file_paths})
    if(file_path MATCHES ".*\\.(cfg|in|js|py|toml)$")
        # Copy file and track changes
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/src/py/${file_path}
            ${CMAKE_CURRENT_BINARY_DIR}/package/${file_path}
            COPYONLY
        )
    else()
        # Copy file, but don't track changes
        file(
            COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/py/${file_path}
            DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/package/
        )
    endif()
endforeach()
