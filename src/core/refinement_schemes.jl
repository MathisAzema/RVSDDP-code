#  Copyright (c) 2017-25, Oscar Dowson and RVSDDP.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Which of the trial states visited by the forward pass receive a cut on the way
# back. This is the "Fw/Bw scheme" axis of the computational experiments, the
# second choice a method is made of, alongside the shift rule of `shifts.jl`.

# Each scheme below is followed by the short name its results directory is built
# from (see `refine_label` in interfaces.jl). A scheme that defines no label
# falls back to its function name.
refine_label(scheme::Function) = string(scheme)

"""
    refine_all(scenario_length::Int, period::Int)

Refine at every state visited by the forward pass, which is scheme `A` of the
computational experiments: a forward pass of length `L(k)` yields `L(k)` cuts.

This is the scheme the convergence analysis covers, provided `L(k) → ∞`.
"""
refine_all(scenario_length::Int, period::Int) = 0:(scenario_length-1)

refine_label(::typeof(refine_all)) = ""

"""
    refine_periodic(scenario_length::Int, period::Int)

Refine one block of `period` consecutive levels, drawn uniformly among the
blocks the forward pass contains. This is scheme `B`, following the periodic
approach of Shapiro and Ding: whatever the length of the forward pass, an
iteration produces exactly `period` cuts, one per phase.

Note that a fixed forward-pass length does not meet the `L(k) → ∞` condition of
the convergence theorem, so this scheme is an empirical baseline.
"""
function refine_periodic(scenario_length::Int, period::Int)
    block = rand(1:round(Int, scenario_length / period))
    return ((block-1)*period):(block*period-1)
end

refine_label(::typeof(refine_periodic)) = "periodic"

"""
    method_label(shift_function::Function, refine_scheme::Function)

The name the pair of choices is known by in the results directories, built from
[`RVSDDP.shift_label`](@ref) and [`RVSDDP.refine_label`](@ref).

A scheme with an empty label contributes nothing, so the four methods compared
in the experiments come out as they are named in the paper:

```julia
method_label(random_shift, refine_all)      # "RVSDDP"
method_label(no_shift,     refine_all)      # "cyclic_sddp"
method_label(random_shift, refine_periodic) # "periodic_RVSDDP"
method_label(no_shift,     refine_periodic) # "periodic_cyclic_sddp"
```
"""
function method_label(shift_function::Function, refine_scheme::Function)
    parts = [refine_label(refine_scheme), shift_label(shift_function)]
    return join(filter(!isempty, parts), "_")
end
