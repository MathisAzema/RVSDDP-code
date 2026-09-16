#  Copyright (c) 2017-25, Oscar Dowson and RVSDDP.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v. 2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

# ========================== The Expectation Operator ======================== #

"""
    Expectation()

The Expectation risk measure.

This risk measure is identical to taking the expectation with respect to the
nominal distribution.

## Example

```jldoctest
julia> risk_adjusted_probability = zeros(4);

julia> RVSDDP.adjust_probability(
           RVSDDP.Expectation(),
           risk_adjusted_probability,
           [0.1, 0.2, 0.3, 0.4],  # nominal_probability,
           RVSDDP.Noise.([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]),  # noise_supports,
           [5.0, 4.0, 6.0, 2.0],  # cost_realizations,
           true,                  # is_minimization
       )
0.0

julia> risk_adjusted_probability
4-element Vector{Float64}:
 0.1
 0.2
 0.3
 0.4
```
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

