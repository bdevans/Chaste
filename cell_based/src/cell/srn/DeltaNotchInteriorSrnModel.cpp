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

#include "DeltaNotchInteriorSrnModel.hpp"
#include "DeltaNotchEdgeSrnModel.hpp"

DeltaNotchInteriorSrnModel::DeltaNotchInteriorSrnModel(boost::shared_ptr<AbstractCellCycleModelOdeSolver> pOdeSolver)
        : AbstractOdeSrnModel(2, pOdeSolver)
{
    if (mpOdeSolver == boost::shared_ptr<AbstractCellCycleModelOdeSolver>())
    {
#ifdef CHASTE_CVODE
        mpOdeSolver = CellCycleModelOdeSolver<DeltaNotchInteriorSrnModel, CvodeAdaptor>::Instance();
        mpOdeSolver->Initialise();
        mpOdeSolver->SetMaxSteps(10000);
#else
        mpOdeSolver = CellCycleModelOdeSolver<DeltaNotchInteriorSrnModel, RungeKutta4IvpOdeSolver>::Instance();
        mpOdeSolver->Initialise();
        SetDt(0.001);
#endif //CHASTE_CVODE
    }
    assert(mpOdeSolver->IsSetUp());
}

DeltaNotchInteriorSrnModel::DeltaNotchInteriorSrnModel(const DeltaNotchInteriorSrnModel& rModel)
        : AbstractOdeSrnModel(rModel)
{
    /*
     * Set each member variable of the new SRN model that inherits
     * its value from the parent.
     *
     * Note 1: some of the new SRN model's member variables
     * will already have been correctly initialized in its constructor.
     *
     * Note 2: one or more of the new SRN model's member variables
     * may be set/overwritten as soon as InitialiseDaughterCell() is called on
     * the new SRN model.
     *
     * Note 3: Only set the variables defined in this class. Variables defined
     * in parent classes will be defined there.
     */

    assert(rModel.GetOdeSystem());
    AbstractOdeSystem* p_parent_system(rModel.GetOdeSystem());
    SetOdeSystem(new DeltaNotchInteriorOdeSystem(p_parent_system->rGetStateVariables()));
    for (unsigned i=0; i < p_parent_system->GetNumberOfParameters(); ++i)
    {
        mpOdeSystem->SetParameter(i, p_parent_system->GetParameter(i));
    }
}

AbstractSrnModel* DeltaNotchInteriorSrnModel::CreateSrnModel()
{
    return new DeltaNotchInteriorSrnModel(*this);
}

void DeltaNotchInteriorSrnModel::ResetForDivision()
{
    assert(mpOdeSystem != nullptr);
    assert(mpCell != nullptr);
    ScaleSrnVariables(0.5);
}

void DeltaNotchInteriorSrnModel::SimulateToCurrentTime()
{
    // Custom behaviour
    UpdateDeltaNotch();

    // Run the ODE simulation as needed
    AbstractOdeSrnModel::SimulateToCurrentTime();
}

void DeltaNotchInteriorSrnModel::Initialise()
{
    AbstractOdeSrnModel::Initialise(new DeltaNotchInteriorOdeSystem);
}

void DeltaNotchInteriorSrnModel::UpdateDeltaNotch()
{
    assert(mpOdeSystem != nullptr);
    assert(mpCell != nullptr);

    const double total_edge_delta = mpCell->GetCellData()->GetItem("total neighbour edge delta");
    mpOdeSystem->SetParameter("total neighbour edge delta", total_edge_delta);

    const double total_edge_notch = mpCell->GetCellData()->GetItem("total edge notch");
    mpOdeSystem->SetParameter("total edge notch", total_edge_notch);
}

double DeltaNotchInteriorSrnModel::GetNotch()
{
    assert(mpOdeSystem != nullptr);
    double notch = mpOdeSystem->rGetStateVariables()[0];
    return notch;
}

void DeltaNotchInteriorSrnModel::SetNotch(double value)
{
    assert(mpOdeSystem != nullptr);
    mpOdeSystem->rGetStateVariables()[0] = value;
}

double DeltaNotchInteriorSrnModel::GetDelta()
{
    assert(mpOdeSystem != nullptr);
    double delta = mpOdeSystem->rGetStateVariables()[1];
    return delta;
}

void DeltaNotchInteriorSrnModel::SetDelta(double value)
{
    assert(mpOdeSystem != nullptr);
    mpOdeSystem->rGetStateVariables()[1] = value;
}

double DeltaNotchInteriorSrnModel::GetTotalEdgeDelta()
{
    assert(mpOdeSystem != nullptr);
    double total_edge_delta = mpOdeSystem->GetParameter("total neighbour edge delta");
    return total_edge_delta;
}

double DeltaNotchInteriorSrnModel::GetTotalEdgeNotch()
{
    assert(mpOdeSystem != nullptr);
    double total_edge_notch = mpOdeSystem->GetParameter("total edge notch");
    return total_edge_notch;
}

void DeltaNotchInteriorSrnModel::OutputSrnModelParameters(out_stream& rParamsFile)
{
    // No new parameters to output, so just call method on direct parent class
    AbstractOdeSrnModel::OutputSrnModelParameters(rParamsFile);
}

void DeltaNotchInteriorSrnModel::AddShrunkEdgeToInterior(AbstractSrnModel* pShrunkEdgeSrn)
{
    // Half of junctional Delta/Notch is endocytosed back to the cytoplasm
    auto p_shrunk_srn = static_cast<DeltaNotchEdgeSrnModel*>(pShrunkEdgeSrn);
    const double edge_notch = p_shrunk_srn->GetNotch();
    const double edge_delta = p_shrunk_srn->GetDelta();
    const double this_notch = GetNotch();
    const double this_delta = GetDelta();

    const double fraction = 0.5;
    SetDelta(this_delta+fraction*edge_delta);
    SetNotch(this_notch+fraction*edge_notch);
}



// Declare identifier for the serializer
#include "SerializationExportWrapperForCpp.hpp"
CHASTE_CLASS_EXPORT(DeltaNotchInteriorSrnModel)
#include "CellCycleModelOdeSolverExportWrapper.hpp"
EXPORT_CELL_CYCLE_MODEL_ODE_SOLVER(DeltaNotchInteriorSrnModel)
