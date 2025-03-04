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


#ifndef _TESTMONODOMAINSLABLONG_HPP_
#define _TESTMONODOMAINSLABLONG_HPP_

#include <cxxtest/TestSuite.h>
#include "MonodomainProblem.hpp"
#include "petscvec.h"
#include <vector>

#include "PetscSetupAndFinalize.hpp"
#include "AbstractCardiacCellFactory.hpp"
#include "TetrahedralMesh.hpp"
#include "LuoRudy1991.hpp"
#include "SimpleStimulus.hpp"

class CornerStimulusCellFactory : public AbstractCardiacCellFactory<3>
{
private:
    boost::shared_ptr<SimpleStimulus> mpStimulus;
public:
    CornerStimulusCellFactory()
        : AbstractCardiacCellFactory<3>(),
          mpStimulus(new SimpleStimulus(-600.0*1000, 0.5))
    {
    }

    AbstractCardiacCell* CreateCardiacCellForTissueNode(Node<3>* pNode)
    {
        return new CellLuoRudy1991FromCellML(mpSolver, mpZeroStimulus);
    }

    void FinaliseCellCreation(std::vector<AbstractCardiacCellInterface* >* pCellsDistributed, unsigned lo, unsigned hi)
    {
        unsigned stimulated_cells[] = { 0, 1, 11, 121 };

        for (unsigned i=0; i<4; i++)
        {
            if ((stimulated_cells[i]>=lo) && (stimulated_cells[i]<hi))
            {
                (*pCellsDistributed)[ stimulated_cells[i] - lo ]->SetStimulusFunction(mpStimulus);
            }
        }
    }
};


class TestMonodomainSlabLong : public CxxTest::TestSuite
{
public:
    void TestMonodomainSlabLongWithCornerNodesStimulated( void )
    {
        HeartConfig::Instance()->SetSimulationDuration(200); //ms
        HeartConfig::Instance()->SetMeshFileName("mesh/test/data/3D_0_to_1mm_6000_elements");
        HeartConfig::Instance()->SetOutputDirectory("MonoDg03dSlabLong");
        HeartConfig::Instance()->SetOutputFilenamePrefix("MonodomainLR91_3dSlabLong");

        CornerStimulusCellFactory cell_factory;

        MonodomainProblem<3> monodomain_problem( &cell_factory );

        monodomain_problem.Initialise();
        monodomain_problem.Solve();
    }
};

#endif //_TESTMONODOMAINSLABLONG_HPP_
