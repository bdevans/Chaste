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

#include "VtkSceneModifier.hpp"

template<unsigned DIM>
VtkSceneModifier<DIM>::VtkSceneModifier()
    : AbstractCellBasedSimulationModifier<DIM>(),
      mpScene(),
      mUpdateFrequency(1)
{
}

template<unsigned DIM>
VtkSceneModifier<DIM>::~VtkSceneModifier()
{
}

// 1-D is not supported
template<>
void VtkSceneModifier<1>::UpdateAtEndOfTimeStep(AbstractCellPopulation<1,1>& rCellPopulation)
{
    UpdateCellData(rCellPopulation);
}

template<unsigned DIM>
void VtkSceneModifier<DIM>::UpdateAtEndOfTimeStep(AbstractCellPopulation<DIM,DIM>& rCellPopulation)
{
    UpdateCellData(rCellPopulation);

    if(DIM>1)
    {
        if(mpScene and SimulationTime::Instance()->GetTimeStepsElapsed()%mUpdateFrequency==0)
        {
            mpScene->ResetRenderer(SimulationTime::Instance()->GetTimeStepsElapsed());
        }
    }
}

template<unsigned DIM>
boost::shared_ptr<VtkScene<DIM> > VtkSceneModifier<DIM>::GetVtkScene()
{
    return mpScene;
}

template<unsigned DIM>
void VtkSceneModifier<DIM>::SetVtkScene(boost::shared_ptr<VtkScene<DIM> > pScene)
{
    mpScene = pScene;
}

// 1-D is not supported
template<>
void VtkSceneModifier<1>::SetupSolve(AbstractCellPopulation<1,1>& rCellPopulation, std::string outputDirectory)
{
    /*
     * We must update CellData in SetupSolve(), otherwise it will not have been
     * fully initialised by the time we enter the main time loop.
     */
    UpdateCellData(rCellPopulation);
}

template<unsigned DIM>
void VtkSceneModifier<DIM>::SetupSolve(AbstractCellPopulation<DIM,DIM>& rCellPopulation, std::string outputDirectory)
{
    /*
     * We must update CellData in SetupSolve(), otherwise it will not have been
     * fully initialised by the time we enter the main time loop.
     */
    UpdateCellData(rCellPopulation);
    if(mpScene and SimulationTime::Instance()->GetTimeStepsElapsed()%mUpdateFrequency==0)
    {
        mpScene->ResetRenderer(SimulationTime::Instance()->GetTimeStepsElapsed());
    }
}

template<unsigned DIM>
void VtkSceneModifier<DIM>::UpdateCellData(AbstractCellPopulation<DIM,DIM>& rCellPopulation)
{
    // Make sure the cell population is updated
    rCellPopulation.Update();
}

template<unsigned DIM>
void VtkSceneModifier<DIM>::OutputSimulationModifierParameters(out_stream& rParamsFile)
{
    // No parameters to output, so just call method on direct parent class
    AbstractCellBasedSimulationModifier<DIM>::OutputSimulationModifierParameters(rParamsFile);
}

template<unsigned DIM>
void VtkSceneModifier<DIM>::SetUpdateFrequency(unsigned frequency)
{
    mUpdateFrequency = frequency;
}

// Explicit instantiation
template class VtkSceneModifier<2>;
template class VtkSceneModifier<3>;

// Serialization for Boost >= 1.36
#include "SerializationExportWrapperForCpp.hpp"
EXPORT_TEMPLATE_CLASS_SAME_DIMS(VtkSceneModifier)
