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

#ifndef ABSTRACTCARDIACCELL_HPP_
#define ABSTRACTCARDIACCELL_HPP_

#include "ChasteSerialization.hpp"
#include "ChasteSerializationVersion.hpp"
#include "ClassIsAbstract.hpp"
#include <boost/serialization/base_object.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/serialization/shared_ptr.hpp>


// This is only needed to prevent compilation errors on PETSc 2.2/Boost 1.33.1 combo
#include "UblasVectorInclude.hpp"

#include "AbstractCardiacCellInterface.hpp"
#include "AbstractOdeSystem.hpp"
#include "AbstractIvpOdeSolver.hpp"
#include "AbstractStimulusFunction.hpp"

#include <vector>

typedef enum _CellModelState
{
    STATE_UNSET = 0,
    FAST_VARS_ONLY,
    ALL_VARS
} CellModelState;

/**
 * This is the base class for ode-based cardiac cell models.
 *
 * It is essentially a cardiac-specific wrapper for ODE systems
 * providing an interface which can interact with the stimulus
 * classes and the voltage in a mono/bidomain simulation.
 *
 * Concrete classes can be autogenerated from CellML files
 * by chaste_codegen, and will automatically inherit from this class.
 */
class AbstractCardiacCell : public AbstractCardiacCellInterface, public AbstractOdeSystem
{
private:
    /** Needed for serialization. */
    friend class boost::serialization::access;
    /**
     * Archive the member variables.
     *
     * @param archive
     * @param version
     */
    template<class Archive>
    void serialize(Archive & archive, const unsigned int version)
    {
        // This calls serialize on the base class.
        archive & boost::serialization::base_object<AbstractOdeSystem>(*this);

        if (version > 0)
        {
            archive & boost::serialization::base_object<AbstractCardiacCellInterface>(*this);
        }
        archive & mDt;

        // For version 2 and above these move into AbstractCardiacCellInterface
        // (AbstractCardiacCellInterface serialization moved to 1 at the same time as this moved to 2).
        if (version <= 1)
        {
            archive & this->mSetVoltageDerivativeToZero;
            if (version > 0)
            {
                // Note that when loading a version 0 archive, this will be initialised to
                // false by our constructor.  So we should get a consistent (wrong) answer
                // with previous versions of Chaste when in tissue.
                archive & this->mIsUsedInTissue;
                archive & this->mHasDefaultStimulusFromCellML;
            }
        }

        if (version == 0)
        {
            CheckForArchiveFix();
        }

        // Paranoia check
        assert(this->mParameters.size() == this->rGetParameterNames().size());
    }

    /**
     * The Luo-Rudy 1991 model saved in previous Chaste versions had a different ordering of state variables.
     *
     * It also didn't save mParameters.
     *
     * Gary: Any changes to other cell models won't work anyway, and as the comment above says,
     * these cells won't work in tissue. We also don't seem to have been checking for optimised cell models.
     * So I think we should retire this conversion with a NEVER_REACHED.
     *
     * If we're loading that model, we could permute the state vector.
     * This can't (easily) be done in the LR91 code itself, since that is auto-generated!
     */
    void CheckForArchiveFix();

protected:
    /** The timestep to use when simulating this cell.  Set from the HeartConfig object. */
    double mDt;

public:
    /** Create a new cardiac cell. The state variables of the cell will be
     *  set to AbstractOdeSystemInformation::GetInitialConditions(). Note that
     *  calls to SetDefaultInitialConditions() on a particular instance of this class
     *  will not modify its state variables. You can modify them directly with
     *  rGetStateVariables().
     *
     * @param pOdeSolver  the ODE solver to use when simulating this cell
     * @param numberOfStateVariables  the size of the ODE system modelling this cell
     * @param voltageIndex  the index of the transmembrane potential within the vector of state variables
     * @param pIntracellularStimulus  the intracellular stimulus current
     */
    AbstractCardiacCell(boost::shared_ptr<AbstractIvpOdeSolver> pOdeSolver,
                        unsigned numberOfStateVariables,
                        unsigned voltageIndex,
                        boost::shared_ptr<AbstractStimulusFunction> pIntracellularStimulus);

    /** Virtual destructor */
    virtual ~AbstractCardiacCell();

    /**
     * Initialise the cell:
     *  - set our state variables to the initial conditions,
     *  - resize model parameters vector.
     *
     * @note Must be called by subclass constructors.
     */
    void Init();

    /**
     * Set the timestep to use for simulating this cell.
     *
     * @param dt  the timestep
     */
    void SetTimestep(double dt);

    /**
     * Simulate this cell's behaviour between the time interval [tStart, tEnd],
     * with timestemp #mDt, updating the internal state variable values.
     *
     * @param tStart  beginning of the time interval to simulate
     * @param tEnd  end of the time interval to simulate
     */
    virtual void SolveAndUpdateState(double tStart, double tEnd);

    /**
     * Simulates this cell's behaviour between the time interval [tStart, tEnd],
     * with timestep #mDt, and return state variable values.
     *
     * @param tStart  beginning of the time interval to simulate
     * @param tEnd  end of the time interval to simulate
     * @param tSamp  sampling interval for returned results (defaults to #mDt)
     * @return solution object
     */
    virtual OdeSolution Compute(double tStart, double tEnd, double tSamp=0.0);

    /**
     * Simulates this cell's behaviour between the time interval [tStart, tEnd],
     * with timestep #mDt, but does not update the voltage.
     *
     * @param tStart  beginning of the time interval to simulate
     * @param tEnd  end of the time interval to simulate
     */
    virtual void ComputeExceptVoltage(double tStart, double tEnd);

    /** Set the transmembrane potential
     * @param voltage  new value
     */
    void SetVoltage(double voltage);

    /**
     * @return the current value of the transmembrane potential, as given
     * in our state variable vector.
     */
    double GetVoltage();


    /**
     * This just returns the number of state variables in the cell model.
     *
     * It is here because we seem to need to specify explicitly
     * which method in the parent classes we intend to implement
     * to take care of the pure definition in AbstractCardiacCellInterface
     *
     * @return the number of state variables
     */
    unsigned GetNumberOfStateVariables() const;

    /**
     * This just returns the number of parameters in the cell model.
     *
     * It is here because we seem to need to specify explicitly
     * which method in the parent classes we intend to implement
     * to take care of the pure definition in AbstractCardiacCellInterface
     *
     * @return the number of parameters
     */
    unsigned GetNumberOfParameters() const;

    /**
     * This just returns the state variables in the cell model.
     *
     * It is here (despite being inherited) because we seem to need to specify explicitly
     * which method in the parent classes we intend to implement
     * to take care of the pure definition in AbstractCardiacCellInterface.
     *
     * @return the state variables
     */
    std::vector<double> GetStdVecStateVariables();

    /**
     * Just calls AbstractOdeSystem::rGetStateVariableNames().
     *
     * It is here (despite being inherited) because we seem to need to specify explicitly
     * which method in the parent classes we intend to implement
     * to take care of the pure definition in AbstractCardiacCellInterface.
     *
     * @return the state variable names in the cell's ODE system.
     */
    const std::vector<std::string>& rGetStateVariableNames() const;


    /**
     * This just sets the state variables in the cell model.
     *
     * It is here (despite being inherited) because we seem to need to specify explicitly
     * which method in the parent classes we intend to implement
     * to take care of the pure definition in AbstractCardiacCellInterface.
     *
     * @param rVariables  the state variables (to take a copy of).
     */
    void SetStateVariables(const std::vector<double>& rVariables);

    /**
     * This just calls the method AbstractOdeSystem::SetStateVariable
     *
     * It is here (despite being inherited) because we seem to need to specify explicitly
     * which method in the parent classes we intend to implement
     * to take care of the pure definition in AbstractCardiacCellInterface.
     *
     * @param index index of the state variable to be set
     * @param newValue new value of the state variable
     */
    void SetStateVariable(unsigned index, double newValue);

    /**
     * This just calls the method AbstractOdeSystem::SetStateVariable
     *
     * It is here (despite being inherited) because we seem to need to specify explicitly
     * which method in the parent classes we intend to implement
     * to take care of the pure definition in AbstractCardiacCellInterface.
     *
     * @param rName name of the state variable to be set
     * @param newValue new value of the state variable
     */
    void SetStateVariable(const std::string& rName, double newValue);

    /**
     * This just calls the method AbstractOdeSystem::GetAnyVariable
     *
     * It is here (despite being inherited) because we seem to need to specify explicitly
     * which method in the parent classes we intend to implement
     * to take care of the pure definition in AbstractCardiacCellInterface.
     *
     * @param rName variable name
     * @param time the time at which you want it
     * @return value of the variable at that time
     */
    double GetAnyVariable(const std::string& rName, double time=0.0);

    /**
     * This just calls AbstractOdeSystem::GetParameter
     *
     * It is here (despite being inherited) because we seem to need to specify explicitly
     * which method in the parent classes we intend to implement
     * to take care of the pure definition in AbstractCardiacCellInterface.
     *
     * @param rParameterName  the name of a parameter to get the value of,
     * @return  the parameter's value.
     */
    double GetParameter(const std::string& rParameterName);

    /**
     * This just calls AbstractOdeSystem::GetParameter
     *
     * It is here (despite being inherited) because we seem to need to specify explicitly
     * which method in the parent classes we intend to implement
     * to take care of the pure definition in AbstractCardiacCellInterface.
     *
     * @param parameterIndex  the index of a parameter to get the value of,
     * @return  the parameter's value.
     */
    double GetParameter(unsigned parameterIndex);

    /**
     * This just calls AbstractOdeSystem::SetParameter
     *
     * It is here (despite being inherited) because we seem to need to specify explicitly
     * which method in the parent classes we intend to implement
     * to take care of the pure definition in AbstractCardiacCellInterface.
     *
     * @param rParameterName  the parameter name to set the value of,
     * @param value  value to set it to.
     */
    void SetParameter(const std::string& rParameterName, double value);

    ////////////////////////////////////////////////////////////////////////
    //  METHODS NEEDED BY FAST CARDIAC CELLS
    ////////////////////////////////////////////////////////////////////////

    /**
     * This should be implemented by fast/slow cardiac cell subclasses, and
     *  \li set the state
     *  \li initialise the cell
     *  \li \b SET #mNumberOfStateVariables \b CORRECTLY
     *      (as this would not have been known in the constructor.
     *
     * \note  This \e must be implemented by fast/slow cardiac cell subclasses.
     *
     * @param state  whether this cell is in fast or slow mode.
     */
    virtual void SetState(CellModelState state);

    /**
     * Set the slow variables. Should only be valid in fast mode.
     *
     * \note  This \e must be implemented by fast/slow cardiac cell subclasses.
     *
     * @param rSlowValues  values for the slow variables
     */
    virtual void SetSlowValues(const std::vector<double> &rSlowValues);

    /**
     * Returns the current values of the slow variables (via rSlowValues). Should only be valid in slow mode.
     *
     * \note  This \e must be implemented by fast/slow cardiac cell subclasses.
     *
     * @param rSlowValues  will be filled in with the values of the slow variables (returned).
     */
    virtual void GetSlowValues(std::vector<double>& rSlowValues);

    /** @return whether this cell is a fast or slow version.
     *
     * \note  This \e must be implemented by fast/slow cardiac cell subclasses.
     */
    virtual bool IsFastOnly();

    /**
     * In a multiscale simulation a cut-down cell model can be run:
     *  - fast values are calculated according to the CellML definition
     *  - slow values are interpolated on synchronisation time-steps.
     * There's a chance that linear interpolation/extrapolation may push
     * some gating variable out of the range [0, 1].  This method alters
     * any values which are out-of-range.
     *
     * \note  This \e must be implemented by fast/slow cardiac cell subclasses.
     *
     * @param rSlowValues A vector of the slow values for a particular cell after they have been interpolated from nearby coarse cells
     */
    virtual void AdjustOutOfRangeSlowValues(std::vector<double>& rSlowValues);

    /**
     * @return the number of slow variables for the cell model
     * (irrespective of whether in fast or slow mode).
     *
     * \note  This \e must be implemented by fast/slow cardiac cell subclasses.
     */
    virtual unsigned GetNumSlowValues();
};

CLASS_IS_ABSTRACT(AbstractCardiacCell)
BOOST_CLASS_VERSION(AbstractCardiacCell, 2)

#endif /*ABSTRACTCARDIACCELL_HPP_*/
