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

/*
 *
 *  Chaste tutorial - this page gets automatically changed to a wiki page
 *  DO NOT remove the comments below, and if the code has to be changed in
 *  order to run, please check the comments are still accurate
 *
 *
 */

#ifndef TESTANOTHERCARDIACELECTROMECHANICSTUTORIAL_HPP_
#define TESTANOTHERCARDIACELECTROMECHANICSTUTORIAL_HPP_

/*
 * ## Cardiac Electro-mechanical Problems (cont.)
 *
 * It is worth running this test suite with `build=GccOpt_ndebug`
 *
 * The same includes as the previous tutorial */
#include <cxxtest/TestSuite.h>
#include "PlaneStimulusCellFactory.hpp"
#include "PetscSetupAndFinalize.hpp"
#include "CardiacElectroMechProbRegularGeom.hpp"
#include "CardiacElectroMechanicsProblem.hpp"
#include "LuoRudy1991.hpp"
#include "NonlinearElasticityTools.hpp"
#include "NobleVargheseKohlNoble1998WithSac.hpp"
#include "CompressibleMooneyRivlinMaterialLaw.hpp"
#include "NobleVargheseKohlNoble1998WithSac.hpp"
#include "Hdf5ToMeshalyzerConverter.hpp"
#include "ZeroStimulusCellFactory.hpp"
#include "FileFinder.hpp"

/* A cell factory used in one of the tests */
class PointStimulus2dCellFactory : public AbstractCardiacCellFactory<2>
{
private:
    boost::shared_ptr<SimpleStimulus> mpStimulus;

public:
    PointStimulus2dCellFactory()
        : AbstractCardiacCellFactory<2>(),
          mpStimulus(new SimpleStimulus(-5e5, 0.5))
    {
    }

    AbstractCardiacCell* CreateCardiacCellForTissueNode(Node<2>* pNode)
    {
        double x = pNode->rGetLocation()[0];
        double y = pNode->rGetLocation()[1];
        if (fabs(x)<0.02+1e-6 && y<-0.4+1e-6) // stimulating small region
        {
            return new CellLuoRudy1991FromCellML(mpSolver, mpStimulus);
        }
        else
        {
            return new CellLuoRudy1991FromCellML(mpSolver, mpZeroStimulus);
        }
    }
};

class TestAnotherCardiacElectroMechanicsTutorial : public CxxTest::TestSuite
{
public:
    /*
     * HOW_TO_TAG Cardiac/Electro-mechanics
     * Run electro-mechanics with mechano-electric feedback
     */

    /* ### Mechano-electric feedback, and alternative boundary conditions
     *
     * Let us now run a simulation with mechano-electric feedback (MEF), and with different boundary conditions.
     */
    void TestWithMef()
    {
        /* If we want to use MEF, where the stretch (in the fibre-direction) couples back to the cell
         * model and is used in stretch-activated channels (SACs), we can't just let Chaste convert
         * from cellml to C++ code as usual (see electro-physiology tutorials on how cell model files
         * are autogenerated from CellML during compilation), since these files don't use stretch and don't
         * have SACs. We have to use `chaste_codegen` to create a cell model class for us, rename and save it, and
         * manually add the SAC current.
         *
         * There is one example of this already in the code-base, which we will use it the following
         * simulation. It is the Noble 98 model, with a SAC added that depends linearly on stretches (>1).
         * It is found in the file [NobleVargheseKohlNoble1998WithSac.hpp](https://github.com/Chaste/Chaste/blob/develop/heart/src/odes/ionicmodels/NobleVargheseKohlNoble1998WithSac.hpp), and defines a class called
         * `CML_noble_varghese_kohl_noble_1998_basic_with_sac`.
         *
         * To add a SAC current to (or otherwise alter) your favourite cell model, you have to
         * auto-generate the non-SAC C++ code at the command line, following the guide [Code Generation From CellML](../../user-guides/code-generation-from-cellml/#chaste_codegen-command-line-arguments).
         *
         * Copy and rename the resultant `.hpp` and `.cpp` files (which can be found in the same folder as the
         * input cellml). For example, rename everything to `LuoRudy1991WithSac`. Then alter the class
         * to overload the method `AbstractCardiacCell::SetStretch(double stretch)` to store the stretch,
         * and then implement the SAC in the `GetIIonic()` method. [NobleVargheseKohlNoble1998WithSac.cpp](https://github.com/Chaste/Chaste/blob/develop/heart/src/odes/ionicmodels/NobleVargheseKohlNoble1998WithSac.cpp)
         * provides an example of the changes that need to be made.
         *
         * Let us create a cell factory returning these Noble98 SAC cells, but with no stimulus - the
         * SAC switching on will lead be to activation.
         */
        ZeroStimulusCellFactory<CML_noble_varghese_kohl_noble_1998_basic_with_sac, 2> cell_factory;

        /* Construct two meshes are before, in 2D */
        TetrahedralMesh<2,2> electrics_mesh;
        electrics_mesh.ConstructRegularSlabMesh(0.01/*stepsize*/, 0.1/*length*/, 0.1/*width*/, 0.1/*depth*/);

        QuadraticMesh<2> mechanics_mesh;
        mechanics_mesh.ConstructRegularSlabMesh(0.02, 0.1, 0.1, 0.1 /*as above with a different stepsize*/);

        /* Collect the fixed nodes. This time we directly specify the new locations. We say the
         * nodes on $X=0$ are to be fixed, setting the deformed $x=0$, but leaving $y$ to be free
         * (sliding boundary conditions). This functionality is described in more detail in the
         * solid mechanics tutorials.
         */
        std::vector<unsigned> fixed_nodes;
        std::vector<c_vector<double,2> > fixed_node_locations;

        fixed_nodes.push_back(0);
        fixed_node_locations.push_back(zero_vector<double>(2));

        for (unsigned i=1; i<mechanics_mesh.GetNumNodes(); i++)
        {
            double X = mechanics_mesh.GetNode(i)->rGetLocation()[0];
            if (fabs(X) < 1e-6) // ie, if X==0
            {
                c_vector<double,2> new_position;
                new_position(0) = 0.0;
                new_position(1) = ElectroMechanicsProblemDefinition<2>::FREE;

                fixed_nodes.push_back(i);
                fixed_node_locations.push_back(new_position);
            }
        }

        /* Now specify tractions on the top and bottom surfaces. For full descriptions of how
         * to apply tractions see the solid mechanics tutorials. Here, we collect the boundary
         * elements on the bottom and top surfaces, and apply inward tractions - this will have the
         * effect of stretching the tissue in the $X$-direction.
         */
        std::vector<BoundaryElement<1,2>*> boundary_elems;
        std::vector<c_vector<double,2> > tractions;

        c_vector<double,2> traction;

        for (TetrahedralMesh<2,2>::BoundaryElementIterator iter = mechanics_mesh.GetBoundaryElementIteratorBegin();
             iter != mechanics_mesh.GetBoundaryElementIteratorEnd();
             ++iter)
        {
            if (fabs((*iter)->CalculateCentroid()[1] - 0.0) < 1e-6) // if Y=0
            {
                BoundaryElement<1,2>* p_element = *iter;
                boundary_elems.push_back(p_element);

                traction(0) =  0.0; // kPa, since the contraction model and material law use kPa for stiffnesses
                traction(1) =  1.0; // kPa, since the contraction model and material law use kPa for stiffnesses
                tractions.push_back(traction);
            }
            if (fabs((*iter)->CalculateCentroid()[1] - 0.1) < 1e-6) // if Y=0.1
            {
                BoundaryElement<1,2>* p_element = *iter;
                boundary_elems.push_back(p_element);

                traction(0) =  0.0;
                traction(1) = -1.0;
                tractions.push_back(traction);
            }
        }

        /* Now set up the problem. We will use a compressible approach. */
        ElectroMechanicsProblemDefinition<2> problem_defn(mechanics_mesh);
        problem_defn.SetContractionModel(KERCHOFFS2003,0.01/*contraction model ODE timestep*/);
        problem_defn.SetUseDefaultCardiacMaterialLaw(INCOMPRESSIBLE);
        problem_defn.SetMechanicsSolveTimestep(1.0);
        /* Set the fixed node and traction info. */
        problem_defn.SetFixedNodes(fixed_nodes, fixed_node_locations);
        problem_defn.SetTractionBoundaryConditions(boundary_elems, tractions);

        /* Now say that the deformation should affect the electro-physiology */
        problem_defn.SetDeformationAffectsElectrophysiology(false /*deformation affects conductivity*/, true /*deformation affects cell models*/);

        /* Set the end time, create the problem, and solve */
        HeartConfig::Instance()->SetSimulationDuration(50.0);

        CardiacElectroMechanicsProblem<2,1> problem(INCOMPRESSIBLE,
                                                    MONODOMAIN,
                                                    &electrics_mesh,
                                                    &mechanics_mesh,
                                                    &cell_factory,
                                                    &problem_defn,
                                                    "TestCardiacElectroMechanicsWithMef");
        problem.Solve();

        /* Nothing exciting happens in the simulation as it is currently written. To get some interesting occurring,
         * alter the SAC conductance in the cell model from 0.035 to 0.35 (micro-Siemens).
         * (look for the line `const double g_sac = 0.035` in [`NobleVargheseKohlNoble1998WithSac.hpp`](https://github.com/Chaste/Chaste/blob/develop/heart/src/odes/ionicmodels/NobleVargheseKohlNoble1998WithSac.hpp)).
         *
         * Rerun and visualise as usual, using Cmgui. By visualising the voltage on the deforming mesh, you can see that the
         * voltage gradually increases due to the SAC, since the tissue is stretched, until the threshold is reached
         * and activation occurs.
         *
         * For MEF simulations, we may want to visualise the electrical results on the electrics mesh using
         * Meshalyzer, for example to more easily visualise action potentials. This isn't (and currently
         * can't be) created by `CardiacElectroMechanicsProblem`. We can use a converter as follows
         * to post-process:
         */
        FileFinder test_output_folder("TestCardiacElectroMechanicsWithMef/electrics", RelativeTo::ChasteTestOutput);
        Hdf5ToMeshalyzerConverter<2,2> converter(test_output_folder, "voltage",
                                                 &electrics_mesh, false,
                                                 HeartConfig::Instance()->GetVisualizerOutputPrecision());

        /* Some other notes. If you want to apply time-dependent traction boundary conditions, this is possible by
         * specifying the traction in functional form - see solid mechanics tutorials. Similarly, more natural
         * 'pressure acting on the deformed body' boundary conditions are possible - see below tutorial.
         *
         * **Robustness:** Sometimes the nonlinear solver doesn't converge, and will give an error. This can be due to either
         * a non-physical (or not very physical) situation, or just because the current guess is quite far
         * from the solution and the solver can't find the solution (this can occur in nonlinear elasticity
         * problems if the loading is large, for example). Current work in progress is on making the solver
         * more robust, and also on parallelising the solver. One option when a solve fails is to decrease the
         * mechanics timestep.
         */

        /* Ignore the following, it is just to check nothing has changed. */
        Hdf5DataReader reader("TestCardiacElectroMechanicsWithMef/electrics", "voltage");
        unsigned num_timesteps = reader.GetUnlimitedDimensionValues().size();
        Vec voltage = PetscTools::CreateVec(electrics_mesh.GetNumNodes());
        reader.GetVariableOverNodes(voltage, "V", num_timesteps-1);
        ReplicatableVector voltage_repl(voltage);
        for (unsigned i=0; i<voltage_repl.GetSize(); i++)
        {
            TS_ASSERT_DELTA(voltage_repl[i], -81.9080, 1e-3);
        }
        PetscTools::Destroy(voltage);
    }

    /*
     * HOW_TO_TAG Cardiac/Electro-mechanics
     * Run electro-mechanics with inflation pressures
     */

    /* ### Internal pressures
     *
     * Next, we run a simulation on a 2d annulus, with an internal pressure applied.
     */
    void TestAnnulusWithInternalPressure()
    {
        /* The following should require little explanation now */
        TetrahedralMesh<2,2> electrics_mesh;
        QuadraticMesh<2> mechanics_mesh;

        // could (should?) use finer electrics mesh, but keeping electrics simulation time down
        TrianglesMeshReader<2,2> reader1("mesh/test/data/annuli/circular_annulus_960_elements");
        electrics_mesh.ConstructFromMeshReader(reader1);

        TrianglesMeshReader<2,2> reader2("mesh/test/data/annuli/circular_annulus_960_elements_quad",2 /*quadratic elements*/);
        mechanics_mesh.ConstructFromMeshReader(reader2);

        PointStimulus2dCellFactory cell_factory;

        std::vector<unsigned> fixed_nodes;
        std::vector<c_vector<double,2> > fixed_node_locations;
        for (unsigned i=0; i<mechanics_mesh.GetNumNodes(); i++)
        {
            double x = mechanics_mesh.GetNode(i)->rGetLocation()[0];
            double y = mechanics_mesh.GetNode(i)->rGetLocation()[1];

            if (fabs(x)<1e-6 && fabs(y+0.5)<1e-6)  // fixed point (0.0,-0.5) at bottom of mesh
            {
                fixed_nodes.push_back(i);
                c_vector<double,2> new_position;
                new_position(0) = x;
                new_position(1) = y;
                fixed_node_locations.push_back(new_position);
            }
            if (fabs(x)<1e-6 && fabs(y-0.5)<1e-6)  // constrained point (0.0,0.5) at top of mesh
            {
                fixed_nodes.push_back(i);
                c_vector<double,2> new_position;
                new_position(0) = x;
                new_position(1) = ElectroMechanicsProblemDefinition<2>::FREE;
                fixed_node_locations.push_back(new_position);
            }
        }

        /* Increase this end time to see more contraction */
        HeartConfig::Instance()->SetSimulationDuration(30.0);

        ElectroMechanicsProblemDefinition<2> problem_defn(mechanics_mesh);

        problem_defn.SetContractionModel(KERCHOFFS2003,0.1);
        problem_defn.SetUseDefaultCardiacMaterialLaw(COMPRESSIBLE);
        //problem_defn.SetZeroDisplacementNodes(fixed_nodes);
        problem_defn.SetFixedNodes(fixed_nodes, fixed_node_locations);
        problem_defn.SetMechanicsSolveTimestep(1.0);

        FileFinder finder("heart/test/data/fibre_tests/circular_annulus_960_elements.ortho",RelativeTo::ChasteSourceRoot);
        problem_defn.SetVariableFibreSheetDirectionsFile(finder, false);

        /* The elasticity solvers have two nonlinear solvers implemented, one hand-coded and one which uses PETSc's SNES
         * solver. The latter is not the default but can be more robust (and will probably be the default in later
         * versions). This is how it can be used. (This option can also be called if the compiled binary is run from
         * the command line (see [Building the Cardiac Executable](../../dev-guides/building-executable-apps)) and run it using the option "-mech_use_snes").
         */
        problem_defn.SetSolveUsingSnes();

        /* Now let us collect all the boundary elements on the inner (endocardial) surface. The following
         * uses knowledge about the geometry - the inner surface is $r=0.3$, the outer is $r=0.5$. */
        std::vector<BoundaryElement<1,2>*> boundary_elems;
        for (TetrahedralMesh<2,2>::BoundaryElementIterator iter
               = mechanics_mesh.GetBoundaryElementIteratorBegin();
             iter != mechanics_mesh.GetBoundaryElementIteratorEnd();
             ++iter)
        {
            ChastePoint<2> centroid = (*iter)->CalculateCentroid();
            double r = sqrt( centroid[0]*centroid[0] + centroid[1]*centroid[1] );

            if (r < 0.4)
            {
                BoundaryElement<1,2>* p_element = *iter;
                boundary_elems.push_back(p_element);
            }
        }

        /* This is how to set the pressure to be applied to these boundary elements. The negative sign implies
         * inward pressure.
         */
        problem_defn.SetApplyNormalPressureOnDeformedSurface(boundary_elems, -1.0 /*1 KPa is about 8mmHg*/);
        /* The solver computes the equilibrium solution (given the pressure loading) before the first timestep.
         * As there is a big deformation from the undeformed state to this loaded state, the nonlinear solver may
         * not converge. The following increments the loading (solves with $p=-1/3$, then $p=-2/3$, then $p=-1$), which
         * allows convergence to occur.
         */
        problem_defn.SetNumIncrementsForInitialDeformation(3);

        CardiacElectroMechanicsProblem<2,1> problem(COMPRESSIBLE,
                                                    MONODOMAIN,
                                                    &electrics_mesh,
                                                    &mechanics_mesh,
                                                    &cell_factory,
                                                    &problem_defn,
                                                    "TestAnnulusWithInternalPressure");

        /* If we want stresses and strains output, we can do the following. The deformation gradients and 2nd PK stresses
         * for each element will be written at the requested times.  */
        problem.SetOutputDeformationGradientsAndStress(10.0 /*how often (in ms) to write - should be a multiple of mechanics timestep*/);


        /* Since this test involves a large deformation at t=0, several Newton iterations are required. To see how the nonlinear
         * solve is progressing, you can run from the binary from the command line with the command line argument `-mech_verbose`.
         */

        problem.Solve();
    }
};
/*
* Visualise using cmgui, and note the different shapes at $t=-1$ (undeformed) and $t=0$ (loaded)
*
* Note: if you want to have a time-dependent pressure, you can replace the second parameter (the pressure)
* in `SetApplyNormalPressureOnDeformedSurface()` with a function pointer (the name of a function) which returns
* the pressure as a function of time.
*
 * **More examples:** For a 3d ellipsoid geometry test, see [`TestCardiacElectroMechanicsOnEllipsoid.hpp`](https://github.com/Chaste/Chaste/blob/develop/heart/test/mechanics/TestCardiacElectroMechanicsOnEllipsoid.hpp).
 */
#endif /* TESTANOTHERCARDIACELECTROMECHANICSTUTORIAL_HPP_ */
