#  Copyright (c) 2017-25, Oscar Dowson and RVSDDP.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v. 2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

# ========================== The Expectation Operator ======================== #

"""
    Expectation()

Take the expectation under the nominal distribution. This is the only risk
measure the paper uses.
"""
struct Expectation <: AbstractRiskMeasure end

function adjust_probability(
    ::Expectation,
    risk_adjusted_probability::Vector{Float64},
    original_probability::Vector{Float64},
    ::Vector,
    ::Vector{Float64},
    ::Bool,
)
    risk_adjusted_probability .= original_probability
    return 0.0
end

