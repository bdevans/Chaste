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

#ifndef TOROIDALHONEYCOMBMESHGENERATOR_HPP_
#define TOROIDALHONEYCOMBMESHGENERATOR_HPP_

#include <cmath>
#include <vector>

#include <boost/shared_ptr.hpp>
#include "HoneycombMeshGenerator.hpp"
#include "Toroidal2dMesh.hpp"

/**
 * Honeycomb mesh generator that creates a 2D "Toroidal" mesh (one in which
 * periodicity is imposed on the left and right boundaries) for use in cell-centre
 * simulations.
 *
 * NOTE: the user should delete the mesh after use to manage memory.
 */
class ToroidalHoneycombMeshGenerator : public HoneycombMeshGenerator
{
public:

    /**
     * Default constructor.
     *
     * @param numNodesAlongWidth  The number of cells you want along the bottom of the domain
     * @param numNodesAlongDepth  The number of cells you want sides of the domain
     * @param widthScaleFactor  The scale factor for the width of the cells (defaults to 1.0)
     * @param depthScaleFactor  The scale factor for the depth of the cells (defaults to 1.0)
     */
    ToroidalHoneycombMeshGenerator(unsigned numNodesAlongWidth, unsigned numNodesAlongDepth, double widthScaleFactor=1.0, double depthScaleFactor=1.0);

    /**
     * @return a 2D honeycomb mesh
     */
    boost::shared_ptr<MutableMesh<2,2> > GetMesh();

    /**
     * @return a 2D honeycomb mesh with periodic left/right and top/bottom boundaries
     */
    boost::shared_ptr<Toroidal2dMesh> GetToroidalMesh();
};

#endif /*TOROIDALHONEYCOMBMESHGENERATOR_HPP_*/
