#  Copyright (c) 2017-25, Oscar Dowson and RVSDDP.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v. 2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

# The abstract types and the functions that make up each extension point of the
# algorithm, with no implementation attached. Everything in `plugins/` is an
# implementation of an interface declared here, as are the shift rules of
# `core/shifts.jl` and the forward pass of `core/forward_passes.jl`.
#
# Keeping the declarations together means a new component can be written
# against this file alone.

# ================================ risk_measures ============================= #

"""
    AbstractRiskMeasure

The abstract type for the risk measure interface.

You need to define the following methods:
 - [`RVSDDP.adjust_probability`](@ref)
"""
abstract type AbstractRiskMeasure end

"""
    adjust_probability(
        measure::Expectation
        risk_adjusted_probability::Vector{Float64},
        original_probability::Vector{Float64},
        noise_support::Vector{Noise{T}},
        objective_realizations::Vector{Float64},
        is_minimization::Bool,
    ) where {T}
"""
function adjust_probability end

# ============================== sampling_schemes ============================ #

"""
    AbstractSamplingScheme

The abstract type for the sampling-scheme interface.

You need to define the following methods:
 - [`RVSDDP.sample_scenario`](@ref)
"""
abstract type AbstractSamplingScheme end

"""
    sample_scenario(graph::PolicyGraph{T}, ::AbstractSamplingScheme) where {T}

Sample a scenario from the policy graph `graph` based on the sampling scheme.

Returns the scenario as a list of tuples (type `Vector{Tuple{T, <:Any}}`) where
the first component of each tuple is the index of the node, and the second
component is the stagewise-independent noise term observed in that node.
"""
function sample_scenario end

# =============================== stopping_rules ============================= #

"""
    AbstractStoppingRule

The abstract type for the stopping-rule interface.

You need to define the following methods:
 - [`RVSDDP.stopping_rule_status`](@ref)
 - [`RVSDDP.convergence_test`](@ref)
"""
abstract type AbstractStoppingRule end

"""
    stopping_rule_status(::AbstractStoppingRule)::Symbol

Return a symbol describing the stopping rule.
"""
function stopping_rule_status end

"""
    convergence_test(
        model::PolicyGraph,
        log::Vector{Log},
        ::AbstractStoppingRule,
    )::Bool

Return a `Bool` indicating if the algorithm should terminate the training.
"""
function convergence_test(
    graph::PolicyGraph,
    log::Vector{Log},
    stopping_rules::Vector{AbstractStoppingRule},
)
    for stopping_rule in stopping_rules
        if convergence_test(graph, log, stopping_rule)
            return true, stopping_rule_status(stopping_rule)
        end
    end
    return false, :not_solved
end

# ============================== backward_samplers =========================== #

"""
    AbstractBackwardSamplingScheme

The abstract type for backward sampling scheme interface.

You need to define the following methods:
 - [`RVSDDP.sample_backward_noise_terms`](@ref)
"""
abstract type AbstractBackwardSamplingScheme end

"""
    sample_backward_noise_terms(
        backward_sampling_scheme::AbstractBackwardSamplingScheme,
        node::Node{T},
    )::Vector{Noise}

Returns a `Vector{Noise}` of noises sampled from `node.noise_terms` using
`backward_sampling_scheme`.
"""
function sample_backward_noise_terms end

"""
    sample_backward_noise_terms_with_state(
        sampler::AbstractBackwardSamplingScheme,
        node::Node,
        state::Dict{Symbol,Float64},
    )::Vector{Noise}

Returns a `Vector{Noise}` of noises sampled conditionally on the `state` using
`sampler`.
"""
function sample_backward_noise_terms_with_state(
    sampler::AbstractBackwardSamplingScheme,
    node::Node,
    ::Dict{Symbol,Float64},
)
    return sample_backward_noise_terms(sampler, node)
end

# =========================== duality_handlers =========================== #

"""
    AbstractDualityHandler

The abstract type for the duality handler interface.
"""
abstract type AbstractDualityHandler end

"""
    get_dual_solution(
        node::Node,
        duality_handler::AbstractDualityHandler,
    )::Tuple{Float64,Dict{Symbol,Float64}}

Returns a `Float64` for the objective of the dual solution, and a
`Dict{Symbol,Float64}` where the keys are the names of the state variables and
the values are the dual variables associated with the fishing constraint at
`node`.
"""
function get_dual_solution end

"""
    prepare_backward_pass(
        node::Node,
        handler::AbstractDualityHandler,
        options::Options,
    )

Performs any setup needed by the duality handler prior to the backward pass.

Returns a function that, when called with no arguments, undoes the setup.
"""
function prepare_backward_pass(::Node, ::AbstractDualityHandler, ::Any)
    return () -> nothing
end

# ================================== shifts ================================== #

"""
    shift_function(
        model::PolicyGraph{T},
        node::Node{T},
        batch_items::Vector{BackwardPassItems{T,Noise}},
        trial_states::Vector{Dict{Symbol,Float64}},
    )::Tuple{Float64,Int}

The interface for a shift-selection rule, passed as
`train(; shift_function = ...)`.

Called once per backward step, after the children of `node` have been solved for
every trajectory of the batch. `batch_items[j]` holds those solutions for
trajectory `j`, and `trial_states[j]` the trial state they were solved at.

Returns the selected shift `Δ` and the cut count at which it takes effect. The
rule is responsible for applying `Δ` itself, through [`RVSDDP.apply_shift!`](@ref).

The sequence of shifts it produces must be *nonnegative* (`Δ >= 0`) and
*nonanticipative* (`Δ` may only depend on information available at that step);
the convergence analysis rests on those two conditions.

See [`RVSDDP.no_shift`](@ref) and
[`RVSDDP.random_shift`](@ref) for the two rules shipped here.
"""
function shift_function end

"""
    shift_label(rule)::String

The short name a shift rule is known by in the results directories, defined in
`shifts.jl` alongside the rule itself.

This is deliberately *not* the function's own name. Keeping the two apart lets a
directory be named after the method it implements --- `RVSDDP`, `cyclic_sddp`
--- rather than after the mechanism, and lets a rule be renamed without
orphaning the results already written to disk. The run scripts build their paths
with it:

```julia
folder = "results_toy/\$(RVSDDP.shift_label(shift_function))_parallel_\$(parallel)"
```

Defaults to the function's name, so a new rule works without defining one.
"""
function shift_label end

# =========================== refinement schemes ============================= #

"""
    refine_scheme(scenario_length::Int, period::Int)

The interface for a refinement scheme, passed as `train(; refine_scheme = ...)`.

Called once at the start of each backward pass. Returns the levels of the
forward pass that receive a cut, as an ordered collection of integers; level `0`
denotes the initial state `x0`, and level `i >= 1` the `i`-th trial state.

The first `period` levels it returns are also the ones whose Bellman residual is
recorded, so that each phase contributes exactly one residual per iteration.

See [`RVSDDP.refine_all`](@ref) and [`RVSDDP.refine_periodic`](@ref) for the two
schemes shipped here.
"""
function refine_scheme end

"""
    refine_label(scheme)::String

The short name a refinement scheme is known by in the results directories,
defined in `refinement_schemes.jl` alongside the scheme itself. Combined with
[`RVSDDP.shift_label`](@ref) by [`RVSDDP.method_label`](@ref).

An empty label means the scheme adds nothing to the method name, which is how
scheme `A` stays plain `RVSDDP` while scheme `B` becomes `periodic_RVSDDP`.

Defaults to the function's name, so a new scheme works without defining one.
"""
function refine_label end

"""
    method_label(shift_function, refine_scheme)::String

The directory name a (shift rule, refinement scheme) pair is stored under. The
run scripts build their paths with it:

```julia
folder = "results_toy/\$(RVSDDP.method_label(shift_function, refine_scheme))_parallel_\$(parallel)"
```
"""
function method_label end

# ============================= parallel schemes ============================= #

# ============================= forward pass ============================= #

"""
    AbstractForwardPass

Abstract type for different forward passes.
"""
abstract type AbstractForwardPass end

"""
    forward_pass(model::PolicyGraph, options::Options, ::AbstractForwardPass)

Return a forward pass as a named tuple with the following fields:

    (
        ;scenario_path,
        sampled_states,
        cumulative_value,
    )

See [`DefaultForwardPass`](@ref) for details.
"""
function forward_pass end
