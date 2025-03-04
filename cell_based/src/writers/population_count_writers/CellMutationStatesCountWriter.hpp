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

#ifndef CELLMUTATIONSTATESCOUNTWRITER_HPP_
#define CELLMUTATIONSTATESCOUNTWRITER_HPP_

#include "AbstractCellPopulationCountWriter.hpp"
#include "ChasteSerialization.hpp"
#include <boost/serialization/base_object.hpp>

/**
 * A class written using the visitor pattern for writing cell mutations states from a cell population to file.
 *
 * The output file is called cellmutationstates.dat by default.
 */
template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
class CellMutationStatesCountWriter : public AbstractCellPopulationCountWriter<ELEMENT_DIM, SPACE_DIM>
{
private:
    /** Needed for serialization. */
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
        archive & boost::serialization::base_object<AbstractCellPopulationCountWriter<ELEMENT_DIM, SPACE_DIM> >(*this);
    }

public:

    /**
     * Default constructor.
     */
    CellMutationStatesCountWriter();

    /**
     * Overridden WriteHeader() method.
     *
     * Write the header to file.
     *
     * @param pCellPopulation a pointer to the population to be written.
     */
    virtual void WriteHeader(AbstractCellPopulation<ELEMENT_DIM, SPACE_DIM>* pCellPopulation);

    /**
     * A general method for writing to any population.
     *
     * @param pCellPopulation the population to write
     */
    void VisitAnyPopulation(AbstractCellPopulation<SPACE_DIM, SPACE_DIM>* pCellPopulation);

    /**
     * Visit the population and write the number of cells in the population that have each mutation state.
     *
     * Outputs a line of tab-separated values of the form:
     * [num mutation state 0] [num mutation state 1] [num mutation state 2] ...
     *
     * where [num mutation state 0] denotes the number of cells in the population that have the mutation state
     * with index 0 in the registry of cell properties, and so on. These counts are computed through the cell
     * population method GetCellMutationStateCount(). The ordering of mutation states is usually specified
     * by the cell population method SetDefaultCellMutationStateAndProliferativeTypeOrdering().
     *
     * This line is appended to the output written by AbstractCellBasedWriter, which is a single
     * value [present simulation time], followed by a tab.
     *
     * @param pCellPopulation a pointer to the MeshBasedCellPopulation to visit.
     */
    virtual void Visit(MeshBasedCellPopulation<ELEMENT_DIM, SPACE_DIM>* pCellPopulation);

    /**
     * Visit the population and write the number of cells in the population that have each mutation state.
     *
     * Outputs a line of tab-separated values of the form:
     * [num mutation state 0] [num mutation state 1] [num mutation state 2] ...
     *
     * where [num mutation state 0] denotes the number of cells in the population that have the mutation state
     * with index 0 in the registry of cell properties, and so on. These counts are computed through the cell
     * population method GetCellMutationStateCount(). The ordering of mutation states is usually specified
     * by the cell population method SetDefaultCellMutationStateAndProliferativeTypeOrdering().
     *
     * This line is appended to the output written by AbstractCellBasedWriter, which is a single
     * value [present simulation time], followed by a tab.
     *
     * @param pCellPopulation a pointer to the CaBasedCellPopulation to visit.
     */
    virtual void Visit(CaBasedCellPopulation<SPACE_DIM>* pCellPopulation);

    /**
     * Visit the population and write the number of cells in the population that have each mutation state.
     *
     * Outputs a line of tab-separated values of the form:
     * [num mutation state 0] [num mutation state 1] [num mutation state 2] ...
     *
     * where [num mutation state 0] denotes the number of cells in the population that have the mutation state
     * with index 0 in the registry of cell properties, and so on. These counts are computed through the cell
     * population method GetCellMutationStateCount(). The ordering of mutation states is usually specified
     * by the cell population method SetDefaultCellMutationStateAndProliferativeTypeOrdering().
     *
     * This line is appended to the output written by AbstractCellBasedWriter, which is a single
     * value [present simulation time], followed by a tab.
     *
     * @param pCellPopulation a pointer to the NodeBasedCellPopulation to visit.
     */
    virtual void Visit(NodeBasedCellPopulation<SPACE_DIM>* pCellPopulation);

    /**
     * Visit the population and write the number of cells in the population that have each mutation state.
     *
     * Outputs a line of tab-separated values of the form:
     * [num mutation state 0] [num mutation state 1] [num mutation state 2] ...
     *
     * where [num mutation state 0] denotes the number of cells in the population that have the mutation state
     * with index 0 in the registry of cell properties, and so on. These counts are computed through the cell
     * population method GetCellMutationStateCount(). The ordering of mutation states is usually specified
     * by the cell population method SetDefaultCellMutationStateAndProliferativeTypeOrdering().
     *
     * This line is appended to the output written by AbstractCellBasedWriter, which is a single
     * value [present simulation time], followed by a tab.
     *
     * @param pCellPopulation a pointer to the PottsBasedCellPopulation to visit.
     */
    virtual void Visit(PottsBasedCellPopulation<SPACE_DIM>* pCellPopulation);

    /**
     * Visit the population and write the number of cells in the population that have each mutation state.
     *
     * Outputs a line of tab-separated values of the form:
     * [num mutation state 0] [num mutation state 1] [num mutation state 2] ...
     *
     * where [num mutation state 0] denotes the number of cells in the population that have the mutation state
     * with index 0 in the registry of cell properties, and so on. These counts are computed through the cell
     * population method GetCellMutationStateCount(). The ordering of mutation states is usually specified
     * by the cell population method SetDefaultCellMutationStateAndProliferativeTypeOrdering().
     *
     * This line is appended to the output written by AbstractCellBasedWriter, which is a single
     * value [present simulation time], followed by a tab.
     *
     * @param pCellPopulation a pointer to the VertexBasedCellPopulation to visit.
     */
    virtual void Visit(VertexBasedCellPopulation<SPACE_DIM>* pCellPopulation);

    /**
     * Visit the population and write the data.
     *
     * Just passes through to VisitAnyPopulation
     *
     * @param pCellPopulation a pointer to the ImmersedBoundaryBasedCellPopulation to visit.
     */
    virtual void Visit(ImmersedBoundaryCellPopulation<SPACE_DIM>* pCellPopulation);
};

#include "SerializationExportWrapper.hpp"
// Declare identifier for the serializer
EXPORT_TEMPLATE_CLASS_ALL_DIMS(CellMutationStatesCountWriter)

#endif /*CELLMUTATIONSTATESCOUNTWRITER_HPP_*/
