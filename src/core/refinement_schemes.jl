#  Copyright (c) 2017-25, Oscar Dowson and RVSDDP.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Which of the trial states visited by the forward pass receive a cut on the way
# back. This is the "Fw/Bw scheme" axis of the computational experiments, the
# second choice a method is made of, alongside the shift rule of `shifts.jl`.
#
# Levels follow the numbering of the paper: a forward pass of length `L` solves
# `L` subproblems and visits the trial states `x^1 = x0, x^2, ..., x^{L+1}`, and
# level `i` is the state `x^i`. As in Algorithm 1, only levels `1` to `L` can
# receive a cut: the last state `x^{L+1}` never does. Level `1` is the initial
# state, which `backward_pass` refines last, through `_refine_at_initial_point`.

# Each scheme below is followed by the short name its results directory is built
# from (see `refine_label` in interfaces.jl). A scheme that defines no label
# falls back to its function name.
refine_label(scheme::Function) = string(scheme)

"""
    refine_all(scenario_length::Int, period::Int)

Refine at every level `x^1 = x0, ..., x^L` of the forward pass, which is scheme
`A` of the computational experiments: a forward pass of length `L(k)` yields
`L(k)` cuts, and with `L(k) = T k` every phase receives exactly `k` of them.

This is the scheme the convergence analysis covers, provided `L(k) → ∞`. It
refines at `x0` at every iteration, which is what puts `x0` in the stable set.
"""
refine_all(scenario_length::Int, period::Int) = 1:scenario_length

refine_label(::typeof(refine_all)) = ""

"""
    refine_periodic(scenario_length::Int, period::Int)

Refine one block of `period` consecutive levels, drawn uniformly among the
blocks `(b-1) period + 1 : b period` the forward pass contains. This is scheme
`B`, following the periodic approach of Shapiro and Ding: whatever the length of
the forward pass, an iteration produces exactly `period` cuts, one per phase.
The initial state `x0` is refined only when the first block is drawn.

If the forward pass is shorter than one period it holds no complete block, and
every level is refined.

Note that a fixed forward-pass length does not meet the `L(k) → ∞` condition of
the convergence theorem, so this scheme is an empirical baseline.
"""
function refine_periodic(scenario_length::Int, period::Int)
    n_blocks = div(scenario_length, period)
    if n_blocks == 0
        return 1:scenario_length
    end
    block = rand(1:n_blocks)
    return ((block-1)*period+1):(block*period)
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
