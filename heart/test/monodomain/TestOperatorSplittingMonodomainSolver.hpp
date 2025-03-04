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

#ifndef TESTOPERATORSPLITTINGMONODOMAINSOLVER_HPP_
#define TESTOPERATORSPLITTINGMONODOMAINSOLVER_HPP_


#include <cxxtest/TestSuite.h>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <vector>
#include "MonodomainProblem.hpp"
#include "ZeroStimulusCellFactory.hpp"
#include "AbstractCardiacCellFactory.hpp"
#include "LuoRudy1991BackwardEulerOpt.hpp"
#include "PlaneStimulusCellFactory.hpp"
#include "TetrahedralMesh.hpp"
#include "PetscTools.hpp"
#include "PetscSetupAndFinalize.hpp"
#include "PropagationPropertiesCalculator.hpp"
#include "MonodomainSolver.hpp"
#include "TenTusscher2006Epi.hpp"
#include "Mahajan2008.hpp"


/* HOW_TO_TAG Cardiac/Solver
 * Run using operator-splitting
 */

// stimulate a block of cells (an interval in 1d, a block in a corner in 2d)
template<unsigned DIM>
class BlockCellFactory : public AbstractCardiacCellFactory<DIM>
{
private:
    boost::shared_ptr<SimpleStimulus> mpStimulus;

public:
    BlockCellFactory()
        : AbstractCardiacCellFactory<DIM>(),
          mpStimulus(new SimpleStimulus(-1000000.0, 0.5))
    {
        assert(DIM<3);
    }

    AbstractCardiacCell* CreateCardiacCellForTissueNode(Node<DIM>* pNode)
    {
        double x = pNode->rGetLocation()[0];

        if (fabs(x)<0.02+1e-6)
        {
            return new CellLuoRudy1991FromCellMLBackwardEulerOpt(this->mpSolver, this->mpStimulus);
        }
        else
        {
            return new CellLuoRudy1991FromCellMLBackwardEulerOpt(this->mpSolver, this->mpZeroStimulus);
        }
    }
};


class TestOperatorSplittingMonodomainSolver : public CxxTest::TestSuite
{
public:

    // The operator splitting and normal methods should agree closely with very small dt and h, but this takes
    // too long to run in the continuous build (see instead TestOperatorSplittingMonodomainSolverLong)
    //
    // Here we run on a fine (as opposed to v fine) mesh and with a normal dt, and check that the solutions
    // are near.
    void TestComparisonOnNormalMeshes()
    {
        ReplicatableVector final_voltage_normal;
        ReplicatableVector final_voltage_operator_splitting;

        HeartConfig::Instance()->SetSimulationDuration(4.0); //ms
        HeartConfig::Instance()->SetOutputFilenamePrefix("results");
        HeartConfig::Instance()->SetOdePdeAndPrintingTimeSteps(0.005, 0.01, 0.1);
        double h = 0.01;

        // Normal
        {
            TetrahedralMesh<1,1> mesh;
            mesh.ConstructRegularSlabMesh(h, 1.0);
            HeartConfig::Instance()->SetOutputDirectory("MonodomainCompareWithOperatorSplitting_normal");
            BlockCellFactory<1> cell_factory;

            MonodomainProblem<1> monodomain_problem( &cell_factory );
            monodomain_problem.SetMesh(&mesh);
            monodomain_problem.Initialise();
            monodomain_problem.Solve();

            final_voltage_normal.ReplicatePetscVector(monodomain_problem.GetSolution());
        }

        // Operator splitting
        {
            TetrahedralMesh<1,1> mesh;
            mesh.ConstructRegularSlabMesh(h, 1.0);
            HeartConfig::Instance()->SetOutputDirectory("MonodomainCompareWithOperatorSplitting_splitting");
            BlockCellFactory<1> cell_factory;

            HeartConfig::Instance()->SetUseReactionDiffusionOperatorSplitting();

            MonodomainProblem<1> monodomain_problem( &cell_factory );
            monodomain_problem.SetMesh(&mesh);
            monodomain_problem.Initialise();
            monodomain_problem.Solve();

            final_voltage_operator_splitting.ReplicatePetscVector(monodomain_problem.GetSolution());
        }

        // hardcoded value to check nothing has changed

        TS_ASSERT_DELTA(final_voltage_operator_splitting[30], 5.0567, 1e-3);

        bool some_node_depolarised = false;
        assert(final_voltage_normal.GetSize()==final_voltage_operator_splitting.GetSize());
        for (unsigned j=0; j<final_voltage_normal.GetSize(); j++)
        {
            // this tolerance means the wavefronts are not on top of each other, but not too far
            // separated (as otherwise max difference between the voltages across space would be
            // greater than 80).
            double tol = 25;

            TS_ASSERT_DELTA(final_voltage_normal[j], final_voltage_operator_splitting[j], tol);

            if (final_voltage_normal[j]>-80)
            {
                // shouldn't be exactly equal, as long as away from resting potential
                TS_ASSERT_DIFFERS(final_voltage_normal[j], final_voltage_operator_splitting[j]);
            }

            if (final_voltage_normal[j]>0.0)
            {
                some_node_depolarised = true;
            }
        }
        UNUSED_OPT(some_node_depolarised);
        assert(some_node_depolarised);
    }
};

#endif /* TESTOPERATORSPLITTINGMONODOMAINSOLVER_HPP_ */
