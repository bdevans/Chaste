/*

Copyright (c) 2005-2025, University of Oxford.
All rights reserved.

University of Oxford means the Chancellor, Masters and Scholars of the
University of Oxford, having an administrative office at Wellington
Square, Oxford OX1 2JD, UK.

This file is part of Chaste.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 * Neither the name of the University of Oxford nor the names of its
   contributors may be used to endorse or promote products derived from this
   software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "Hdf5DataReader.hpp"
#include "Exception.hpp"
#include "OutputFileHandler.hpp"
#include "PetscTools.hpp"

#include <cassert>
#include <algorithm>

Hdf5DataReader::Hdf5DataReader(const std::string& rDirectory,
                               const std::string& rBaseName,
                               bool makeAbsolute,
                               std::string datasetName)
    : AbstractHdf5Access(rDirectory, rBaseName, datasetName, makeAbsolute),
      mNumberTimesteps(1),
      mClosed(false)
{
    CommonConstructor();
}

Hdf5DataReader::Hdf5DataReader(const FileFinder& rDirectory,
                               const std::string& rBaseName,
                               std::string datasetName)
    : AbstractHdf5Access(rDirectory, rBaseName, datasetName),
      mNumberTimesteps(1),
      mClosed(false)
{
    CommonConstructor();
}

void Hdf5DataReader::CommonConstructor()
{
    if (!mDirectory.IsDir() || !mDirectory.Exists())
    {
        EXCEPTION("Directory does not exist: " + mDirectory.GetAbsolutePath());
    }

    std::string file_name = mDirectory.GetAbsolutePath() + mBaseName + ".h5";
    FileFinder h5_file(file_name, RelativeTo::Absolute);

    if (!h5_file.Exists())
    {
        EXCEPTION("Hdf5DataReader could not open " + file_name + " , as it does not exist.");
    }

    // Open the file and the main dataset
    mFileId = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    if (mFileId <= 0)
    {
        EXCEPTION("Hdf5DataReader could not open " << file_name <<
                  " , H5Fopen error code = " << mFileId);
    }

    mVariablesDatasetId = H5Dopen(mFileId, mDatasetName.c_str(), H5P_DEFAULT);
    SetMainDatasetRawChunkCache();

    if (mVariablesDatasetId <= 0)
    {
        H5Fclose(mFileId);
        EXCEPTION("Hdf5DataReader opened " << file_name << " but could not find the dataset '" <<
                  mDatasetName.c_str() << "', H5Dopen error code = " << mVariablesDatasetId);
    }

    hid_t variables_dataspace = H5Dget_space(mVariablesDatasetId);
    mVariablesDatasetRank = H5Sget_simple_extent_ndims(variables_dataspace);

    // Get the dataset/dataspace dimensions
    hsize_t dataset_max_sizes[AbstractHdf5Access::DATASET_DIMS];
    H5Sget_simple_extent_dims(variables_dataspace, mDatasetDims, dataset_max_sizes);

    for (unsigned i=1; i<AbstractHdf5Access::DATASET_DIMS; i++)  // Zero is excluded since it may be unlimited
    {
        assert(mDatasetDims[i] == dataset_max_sizes[i]);
    }

    // Check if an unlimited dimension has been defined
    if (dataset_max_sizes[0] == H5S_UNLIMITED)
    {
        // In terms of an Unlimited dimension dataset:
        // * Files pre - r16738 (inc. Release 3.1 and earlier) use simply "Time" for "Data"'s unlimited variable.
        // * Files generated by r16738 - r18257 used "<DatasetName>_Time" for "<DatasetName>"'s unlimited variable,
        //   - These are not to be used and there is no backwards compatibility for them, since they weren't in a release.
        // * Files post r18257 (inc. Release 3.2 onwards) use "<DatasetName>_Unlimited" for "<DatasetName>"'s
        //   unlimited variable,
        //   - a new attribute "Name" has been added to the Unlimited Dataset to allow it to assign
        //     any name to the unlimited variable. Which can then be easily read by this class.
        //   - if this dataset is missing we look for simply "Time" to remain backwards compatible with Releases <= 3.1
        SetUnlimitedDatasetId();

        hid_t timestep_dataspace = H5Dget_space(mUnlimitedDatasetId);

        // Get the dataset/dataspace dimensions
        H5Sget_simple_extent_dims(timestep_dataspace, &mNumberTimesteps, nullptr);
    }

    // Get the attribute where the name of the variables are stored
    hid_t attribute_id = H5Aopen_name(mVariablesDatasetId, "Variable Details");

    // Get attribute datatype, dataspace, rank, and dimensions
    hid_t attribute_type  = H5Aget_type(attribute_id);
    hid_t attribute_space = H5Aget_space(attribute_id);

    hsize_t attr_dataspace_dim;
    H5Sget_simple_extent_dims(attribute_space, &attr_dataspace_dim, nullptr);

    unsigned num_columns = H5Sget_simple_extent_npoints(attribute_space);
    char* string_array = (char *)malloc(sizeof(char)*MAX_STRING_SIZE*(int)num_columns);
    H5Aread(attribute_id, attribute_type, string_array);

    // Loop over column names and store them.
    for (unsigned index=0; index < num_columns; index++)
    {
        // Read the string from the raw vector
        std::string column_name_unit(&string_array[MAX_STRING_SIZE*index]);

        // Find beginning of unit definition.
        size_t name_length = column_name_unit.find('(');
        size_t unit_length = column_name_unit.find(')') - name_length - 1;

        std::string column_name = column_name_unit.substr(0, name_length);
        std::string column_unit = column_name_unit.substr(name_length+1, unit_length);

        mVariableNames.push_back(column_name);
        mVariableToColumnIndex[column_name] = index;
        mVariableToUnit[column_name] = column_unit;
    }

    // Release all the identifiers
    H5Tclose(attribute_type);
    H5Sclose(attribute_space);
    H5Aclose(attribute_id);

    // Free allocated memory
    free(string_array);

    // Find out if it's incomplete data
    H5E_BEGIN_TRY //Supress HDF5 error if the IsDataComplete name isn't there
    {
        attribute_id = H5Aopen_name(mVariablesDatasetId, "IsDataComplete");
    }
    H5E_END_TRY;
    if (attribute_id < 0)
    {
        // This is in the old format (before we added the IsDataComplete attribute).
        // Just quit (leaving a nasty hdf5 error).
        return;
    }

    attribute_type  = H5Aget_type(attribute_id);
    attribute_space = H5Aget_space(attribute_id);
    unsigned is_data_complete;
    H5Aread(attribute_id, H5T_NATIVE_UINT, &is_data_complete);

    // Release all the identifiers
    H5Tclose(attribute_type);
    H5Sclose(attribute_space);
    H5Aclose(attribute_id);
    mIsDataComplete = (is_data_complete == 1) ? true : false;

    if (is_data_complete)
    {
        return;
    }

    // Incomplete data
    // Read the vector thing
    attribute_id = H5Aopen_name(mVariablesDatasetId, "NodeMap");
    attribute_type  = H5Aget_type(attribute_id);
    attribute_space = H5Aget_space(attribute_id);

    // Get the dataset/dataspace dimensions
    unsigned num_node_indices = H5Sget_simple_extent_npoints(attribute_space);

    // Read data from hyperslab in the file into the hyperslab in memory
    mIncompleteNodeIndices.clear();
    mIncompleteNodeIndices.resize(num_node_indices);
    H5Aread(attribute_id, H5T_NATIVE_UINT, &mIncompleteNodeIndices[0]);

    H5Tclose(attribute_type);
    H5Sclose(attribute_space);
    H5Aclose(attribute_id);
}

std::vector<double> Hdf5DataReader::GetVariableOverTime(const std::string& rVariableName,
                                                        unsigned nodeIndex)
{
    if (!mIsUnlimitedDimensionSet)
    {
        EXCEPTION("The dataset '" << mDatasetName << "' does not contain time dependent data");
    }

    unsigned actual_node_index = nodeIndex;
    if (!mIsDataComplete)
    {
        unsigned node_index = 0;
        for (node_index=0; node_index<mIncompleteNodeIndices.size(); node_index++)
        {
            if (mIncompleteNodeIndices[node_index]==nodeIndex)
            {
                actual_node_index = node_index;
                break;
            }
        }
        if (node_index == mIncompleteNodeIndices.size())
        {
            EXCEPTION("The incomplete dataset '" << mDatasetName << "' does not contain info of node " << nodeIndex);
        }
    }
    if (actual_node_index >= mDatasetDims[1])
    {
        EXCEPTION("The dataset '" << mDatasetName << "' doesn't contain info of node " << actual_node_index);
    }

    std::map<std::string, unsigned>::iterator col_iter = mVariableToColumnIndex.find(rVariableName);
    if (col_iter == mVariableToColumnIndex.end())
    {
        EXCEPTION("The dataset '" << mDatasetName << "' doesn't contain data for variable " << rVariableName);
    }
    unsigned column_index = (*col_iter).second;

    // Define hyperslab in the dataset.
    hsize_t offset[3] = {0, actual_node_index, column_index};
    hsize_t count[3]  = {mDatasetDims[0], 1, 1};
    hid_t variables_dataspace = H5Dget_space(mVariablesDatasetId);
    H5Sselect_hyperslab(variables_dataspace, H5S_SELECT_SET, offset, nullptr, count, nullptr);

    // Define a simple memory dataspace
    hid_t memspace = H5Screate_simple(1, &mDatasetDims[0] ,nullptr);

    // Data buffer to return
    std::vector<double> ret(mDatasetDims[0]);

    // Read data from hyperslab in the file into the hyperslab in memory
    H5Dread(mVariablesDatasetId, H5T_NATIVE_DOUBLE, memspace, variables_dataspace, H5P_DEFAULT, &ret[0]);

    H5Sclose(variables_dataspace);
    H5Sclose(memspace);

    return ret;
}

std::vector<std::vector<double> > Hdf5DataReader::GetVariableOverTimeOverMultipleNodes(const std::string& rVariableName,
                                                                                       unsigned lowerIndex,
                                                                                       unsigned upperIndex)
{
    if (!mIsUnlimitedDimensionSet)
    {
        EXCEPTION("The dataset '" << mDatasetName << "' does not contain time dependent data");
    }

    if (!mIsDataComplete)
    {
        EXCEPTION("GetVariableOverTimeOverMultipleNodes() cannot be called using incomplete data sets (those for which data was only written for certain nodes)");
    }

    if (upperIndex > mDatasetDims[1])
    {
       EXCEPTION("The dataset '" << mDatasetName << "' doesn't contain info for node " << upperIndex-1);
    }

    std::map<std::string, unsigned>::iterator col_iter = mVariableToColumnIndex.find(rVariableName);
    if (col_iter == mVariableToColumnIndex.end())
    {
        EXCEPTION("The dataset '" << mDatasetName << "' doesn't contain data for variable " << rVariableName);
    }
    unsigned column_index = (*col_iter).second;

    // Define hyperslab in the dataset.
    hsize_t offset[3] = {0, lowerIndex, column_index};
    hsize_t count[3]  = {mDatasetDims[0], upperIndex-lowerIndex, 1};
    hid_t variables_dataspace = H5Dget_space(mVariablesDatasetId);
    H5Sselect_hyperslab(variables_dataspace, H5S_SELECT_SET, offset, nullptr, count, nullptr);

    // Define a simple memory dataspace
    hsize_t data_dimensions[2];
    data_dimensions[0] = mDatasetDims[0];
    data_dimensions[1] = upperIndex-lowerIndex;
    hid_t memspace = H5Screate_simple(2, data_dimensions, nullptr);

    double* data_read = new double[mDatasetDims[0]*(upperIndex-lowerIndex)];

    // Read data from hyperslab in the file into the hyperslab in memory
    H5Dread(mVariablesDatasetId, H5T_NATIVE_DOUBLE, memspace, variables_dataspace, H5P_DEFAULT, data_read);

    H5Sclose(variables_dataspace);
    H5Sclose(memspace);

    // Data buffer to return
    unsigned num_nodes_read = upperIndex-lowerIndex;
    unsigned num_timesteps = mDatasetDims[0];

    std::vector<std::vector<double> > ret(num_nodes_read);

    for (unsigned node_num=0; node_num<num_nodes_read; node_num++)
    {
        ret[node_num].resize(num_timesteps);
        for (unsigned time_num=0; time_num<num_timesteps; time_num++)
        {
            ret[node_num][time_num] = data_read[num_nodes_read*time_num + node_num];
        }
    }

    delete[] data_read;

    return ret;
}

void Hdf5DataReader::GetVariableOverNodes(Vec data,
                                          const std::string& rVariableName,
                                          unsigned timestep)
{
    if (!mIsDataComplete)
    {
        EXCEPTION("You can only get a vector for complete data");
    }
    if (!mIsUnlimitedDimensionSet && timestep!=0)
    {
        EXCEPTION("The dataset '" << mDatasetName << "' does not contain time dependent data");
    }

    std::map<std::string, unsigned>::iterator col_iter = mVariableToColumnIndex.find(rVariableName);
    if (col_iter == mVariableToColumnIndex.end())
    {
        EXCEPTION("The dataset '" << mDatasetName << "' does not contain data for variable " << rVariableName);
    }
    unsigned column_index = (*col_iter).second;

    // Check for valid timestep
    if (timestep >= mNumberTimesteps)
    {
        EXCEPTION("The dataset '" << mDatasetName << "' does not contain data for timestep number " << timestep);
    }

    int lo, hi, size;
    VecGetSize(data, &size);
    if ((unsigned)size != mDatasetDims[1])
    {
        EXCEPTION("Could not read data because Vec is the wrong size");
    }
    // Get range owned by each processor
    VecGetOwnershipRange(data, &lo, &hi);

    if (hi > lo) // i.e. we own some...
    {
        // Define a dataset in memory for this process
        hsize_t v_size[1] = {(unsigned)(hi-lo)};
        hid_t memspace = H5Screate_simple(1, v_size, nullptr);

        // Select hyperslab in the file.
        hsize_t offset[3] = {timestep, (unsigned)(lo), column_index};
        hsize_t count[3]  = {1, (unsigned)(hi-lo), 1};
        hid_t hyperslab_space = H5Dget_space(mVariablesDatasetId);
        H5Sselect_hyperslab(hyperslab_space, H5S_SELECT_SET, offset, nullptr, count, nullptr);

        double* p_petsc_vector;
        VecGetArray(data, &p_petsc_vector);

        herr_t err = H5Dread(mVariablesDatasetId, H5T_NATIVE_DOUBLE, memspace, hyperslab_space, H5P_DEFAULT, p_petsc_vector);
        UNUSED_OPT(err);
        assert(err==0);

        VecRestoreArray(data, &p_petsc_vector);

        H5Sclose(hyperslab_space);
        H5Sclose(memspace);
    }
}

std::vector<double> Hdf5DataReader::GetUnlimitedDimensionValues()
{
    // Data buffer to return
    std::vector<double> ret(mNumberTimesteps);

    if (!mIsUnlimitedDimensionSet)
    {
        // Fake it
        assert(mNumberTimesteps==1);
        ret[0] = 0.0;
        return ret;
    }

    // Read data from hyperslab in the file into memory
    H5Dread(mUnlimitedDatasetId, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &ret[0]);

    return ret;
}

void Hdf5DataReader::Close()
{
    if (!mClosed)
    {
        H5Dclose(mVariablesDatasetId);
        if (mIsUnlimitedDimensionSet)
        {
            H5Dclose(mUnlimitedDatasetId);
        }
        H5Fclose(mFileId);
        mClosed = true;
    }
}

Hdf5DataReader::~Hdf5DataReader()
{
    Close();
}

unsigned Hdf5DataReader::GetNumberOfRows()
{
    return mDatasetDims[1];
}

std::vector<std::string> Hdf5DataReader::GetVariableNames()
{
    return mVariableNames;
}

std::string Hdf5DataReader::GetUnit(const std::string& rVariableName)
{
    return mVariableToUnit[rVariableName];
}


