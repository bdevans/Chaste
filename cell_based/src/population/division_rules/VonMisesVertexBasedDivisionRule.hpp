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

#ifndef VONMISESVERTEXBASEDDIVISIONRULE_HPP_
#define VONMISESVERTEXBASEDDIVISIONRULE_HPP_

#include "ChasteSerialization.hpp"
#include <boost/serialization/base_object.hpp>
#include "AbstractVertexBasedDivisionRule.hpp"
#include "VertexBasedCellPopulation.hpp"

// Forward declaration prevents circular include chain
template<unsigned SPACE_DIM> class VertexBasedCellPopulation;
template<unsigned SPACE_DIM> class AbstractVertexBasedDivisionRule;

/**
 * A class to generate a division vector of unit length that points in a direction
 * randomly sampled from a von Mises distribution parameterised by mu (mean parameter)
 * and kappa (concentration parameter).
 */
template <unsigned SPACE_DIM>
class VonMisesVertexBasedDivisionRule : public AbstractVertexBasedDivisionRule<SPACE_DIM>
{
private:

    /** Mean parameter. Initialised to 0 in the constructor. */
    double mMeanParameter;

    /** Concentration parameter. Must be positive. Initialised to 1 in the constructor.  */
    double mConcentrationParameter;

    friend class boost::serialization::access;
    /**
     * Serialize the object and its member variables.
     *
     * @param archive the archive
     * @param version the current version of this class
     */
    template<class Archive>
    void serialize(Archive & archive, const unsigned int version)
    {
        archive & boost::serialization::base_object<AbstractVertexBasedDivisionRule<SPACE_DIM> >(*this);
        archive & mMeanParameter;
        archive & mConcentrationParameter;
    }

public:
    /**
     * Default constructor.
     */
    VonMisesVertexBasedDivisionRule();

    /**
     * Empty destructor.
     */
    virtual ~VonMisesVertexBasedDivisionRule();

    /**
     * @return mMeanParameter
     */
    double GetMeanParameter();

    /**
     * @return mConcentrationParameter
     */
    double GetConcentrationParameter();

    /**
     * Set mMeanParameter.
     *
     * @param meanParameter the new value of mMeanParameter
     */
    void SetMeanParameter(double meanParameter);

    /**
     * Set mConcentrationParameter.
     *
     * @param concentrationParameter the new value of mConcentrationParameter
     */
    void SetConcentrationParameter(double concentrationParameter);

    /**
     * Overridden CalculateCellDivisionVector() method.
     *
     * Return a unit vector that points in a direction randomly sampled from a von Mises distribution, i.e the arguments are redundant for this division rule.
     *
     * @param pParentCell  The cell to divide
     * @param rCellPopulation  The vertex-based cell population
     * @return the division vector.
     */
    virtual c_vector<double, SPACE_DIM> CalculateCellDivisionVector(CellPtr pParentCell,
        VertexBasedCellPopulation<SPACE_DIM>& rCellPopulation);
};

#include "SerializationExportWrapper.hpp"
EXPORT_TEMPLATE_CLASS_SAME_DIMS(VonMisesVertexBasedDivisionRule)

#endif // VONMISESVERTEXBASEDDIVISIONRULE_HPP_
