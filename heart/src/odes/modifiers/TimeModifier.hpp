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

#ifndef TIME_MODIFIER_HPP_
#define TIME_MODIFIER_HPP_

#include "AbstractModifier.hpp"
#include <cmath>

// Serialization headers
#include "ChasteSerialization.hpp"
#include <boost/serialization/base_object.hpp>

/**
 * This is just an example class to show how you might specify a custom
 * modifier to change a parameter through time. In this case it implements
 * a sin(time)*default_parameter factor modifier.
 */
class TimeModifier : public AbstractModifier
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
        archive & boost::serialization::base_object<AbstractModifier>(*this);
        // No member variables at present.
    }

public:
    /**
     * Constructor
     */
    TimeModifier()
    {
    }

    /**
     * Default destructor.
     */
    virtual ~TimeModifier()
    {
    }

    /**
     * Perform the modification.
     *
     * @param param  the current value of the quantity which is being modified
     * @param time  the current simulation time
     * @return the new value for the quantity which is being modified
     */
    virtual double Calc(double param, double time);
};

#include "SerializationExportWrapper.hpp"
CHASTE_CLASS_EXPORT(TimeModifier)

#endif  //TIME_MODIFIER_HPP_
