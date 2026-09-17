#  Copyright (c) 2017-25, Oscar Dowson and RVSDDP.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Out-of-sample simulation of a trained policy: replay the policy along freshly
# sampled scenarios and record the requested variables. This is how the
# estimated policy costs reported in the experiments are produced.

# Internal: run `number_replications` independent simulations of the policy.
function _simulate(
    model::PolicyGraph,
    number_replications::Int,
    variables::Vector{Symbol};
    kwargs...,
)
    _initialize_solver(model; throw_error = false)
    return map(_ -> _simulate(model, variables; kwargs...), 1:number_replications)
end

# Internal function: helper to conduct a single simulation. Users should use the
# documented, user-facing function RVSDDP.simulate instead.
function _simulate(
    model::PolicyGraph{T},
    variables::Vector{Symbol};
    infinite::Bool=true,
    sampling_scheme::AbstractSamplingScheme,
    custom_recorders::Dict{Symbol,Function},
    duality_handler::Union{Nothing,AbstractDualityHandler},
    skip_undefined_variables::Bool,
    incoming_state::Dict{Symbol,Float64},
) where {T}
    # Sample a scenario path.
    scenario_path = sample_scenario(model, sampling_scheme)

    # Storage for the simulation results.
    simulation = Dict{Symbol,Any}[]
    # A cumulator for the stage-objectives.
    cumulative_value = 0.0
    length_scenario_path = length(scenario_path)
    for (depth, (node_index, noise)) in enumerate(scenario_path)
        node = model[node_index]
        lock(node.lock)
        try
            # Solve the subproblem.
            subproblem_results = solve_subproblem(
                model,
                node,
                incoming_state,
                noise;
                duality_handler = duality_handler,
            )
            # Add the stage-objective
            cumulative_value += subproblem_results.stage_objective
            # Record useful variables from the solve.
            store = Dict{Symbol,Any}(
                :node_index => node_index,
                :noise_term => noise,
                :stage_objective => subproblem_results.stage_objective,
                :bellman_term =>
                    subproblem_results.objective -
                    subproblem_results.stage_objective,
                :outgoing_state => subproblem_results.state,
            )
            if depth == length_scenario_path && infinite
                store[:cost_end_of_horizon] = compute_cost_end_of_horizon(model, length_scenario_path+1, subproblem_results.state)
            end
            # Loop through the primal variable values that the user wants.
            for variable in variables
                if haskey(node.subproblem.obj_dict, variable)
                    # Note: we broadcast the call to value for variables which are
                    # containers (like Array, Containers.DenseAxisArray, etc). If
                    # the variable is a scalar (e.g. just a plain VariableRef), the
                    # broadcast preseves the scalar shape.
                    # TODO: what if the variable container is a dictionary? They
                    # should be using Containers.SparseAxisArray, but this might not
                    # always be the case...
                    store[variable] = JuMP.value.(node.subproblem[variable])
                elseif skip_undefined_variables
                    store[variable] = NaN
                else
                    error(
                        "No variable named $(variable) exists in the subproblem.",
                        " If you want to simulate the value of a variable, make ",
                        "sure it is defined in _all_ subproblems, or pass ",
                        "`skip_undefined_variables=true` to `simulate`.",
                    )
                end
            end
            # Loop through any custom recorders that the user provided.
            for (sym, recorder) in custom_recorders
                store[sym] = recorder(node.subproblem)
            end
            # Add the store to our list.
            push!(simulation, store)
            # Set outgoing state as the incoming state for the next node.
            incoming_state = copy(subproblem_results.state)
        finally
            unlock(node.lock)
        end
    end
    return simulation
end

function _initial_state(model::PolicyGraph)
    return Dict(String(k) => v for (k, v) in model.initial_root_state)
end

"""
    simulate(
        model::PolicyGraph,
        number_replications::Int = 1,
        variables::Vector{Symbol} = Symbol[];
        sampling_scheme::AbstractSamplingScheme =
            InSampleMonteCarlo(),
        custom_recorders = Dict{Symbol, Function}(),
        duality_handler::Union{Nothing,AbstractDualityHandler} = nothing,
        skip_undefined_variables::Bool = false,
        incoming_state::Dict{String,Float64} = _initial_state(model),
     )::Vector{Vector{Dict{Symbol,Any}}}

Perform a simulation of the policy model with `number_replications` replications.

## Return data structure

Returns a vector with one element for each replication. Each element is a vector
with one-element for each node in the scenario that was sampled. Each element in
that vector is a dictionary containing information about the subproblem that was
solved.

In that dictionary there are four special keys:

 - `:node_index`, which records the index of the sampled node in the policy model

 - `:noise_term`, which records the noise observed at the node

 - `:stage_objective`, which records the stage-objective of the subproblem

 - `:bellman_term`, which records the cost/value-to-go of the node.

The sum of `:stage_objective + :bellman_term` will equal the objective value of
the solved subproblem.

In addition to the special keys, the dictionary will contain the result of
`key => JuMP.value(subproblem[key])` for each `key` in `variables`. This is
useful to obtain the primal value of the state and control variables.

## Positonal arguments

 - `model`: the model to simulate

 - `number_replications::Int = 1`: the number of simulation replications to
   conduct, that is, the length of the simulation vector that is returned by
   this function. If omitted, this defaults to `1`.`

 - `variables::Vector{Symbol} = Symbol[]`: a list of the variable names to
   record the value of in each stage.

## Keyword arguments

 - `sampling_scheme`: the sampling scheme used when simulating.

 - `custom_recorders`: see `Custom recorders` section below.

 - `duality_handler`: the [`RVSDDP.AbstractDualityHandler`](@ref) used to compute
   dual variables. If you do not require dual variables (or if they are not
   available), pass `duality_handler = nothing`.

 - `skip_undefined_variables`: If you attempt to simulate the value of a
   variable that is only defined in some of the stage problems, an error will be
   thrown. To over-ride this (and return a `NaN` instead), pass
   `skip_undefined_variables = true`.

 - `initial_state`: Use `incoming_state` to pass an initial value of the state
   variable, if it differs from that at the root node. Each key should be the
   string name of the state variable.

## Custom recorders

For more complicated data, the `custom_recorders` keyword argument can be used.

For example, to record the dual of a constraint named `my_constraint`, pass the
following:

```julia
simulation_results = RVSDDP.simulate(model, 2;
    custom_recorders = Dict{Symbol, Function}(
        :constraint_dual => sp -> JuMP.dual(sp[:my_constraint])
    )
)
```
The value of the dual in the first stage of the second replication can be
accessed as:
```julia
simulation_results[2][1][:constraint_dual]
```
"""
function simulate(
    model::PolicyGraph,
    number_replications::Int = 1,
    variables::Vector{Symbol} = Symbol[];
    infinite::Bool=true,
    sampling_scheme::AbstractSamplingScheme = InSampleMonteCarlo(),
    custom_recorders = Dict{Symbol,Function}(),
    duality_handler::Union{Nothing,AbstractDualityHandler} = nothing,
    skip_undefined_variables::Bool = false,
    incoming_state::Dict{String,Float64} = _initial_state(model),
)
    return _simulate(
        model,
        number_replications,
        variables;
        infinite=infinite,
        sampling_scheme = sampling_scheme,
        custom_recorders = custom_recorders,
        duality_handler = duality_handler,
        skip_undefined_variables = skip_undefined_variables,
        incoming_state = Dict(Symbol(k) => v for (k, v) in incoming_state),
    )
end
