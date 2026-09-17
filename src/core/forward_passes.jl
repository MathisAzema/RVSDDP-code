#  Copyright (c) 2017-25, Oscar Dowson and RVSDDP.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
    DefaultForwardPass()

The default forward pass.

The `options.parallel` trajectories of a batch are simulated concurrently, one
Julia task each (see `_parallel_foreach`). Trajectory `i` solves replica `i` of
every node, i.e. its own JuMP model, so no two trajectories touch the same
subproblem. Start Julia with `julia -t N` to give those tasks `N` cores; with
one thread the batch falls back to a plain loop.

Sampling is deliberately *not* part of the concurrent section: the whole batch
is sampled up-front on the calling task. Simulating a trajectory consumes no
randomness, so this leaves the sequence of `rand` calls, and therefore the
trajectories a given seed produces, identical to the sequential implementation.
"""
struct DefaultForwardPass <: AbstractForwardPass end

function forward_pass(
    model::PolicyGraph{T},
    options::Options,
    ::DefaultForwardPass,
) where {T}
    # Sample the whole batch first, on this task, so that the random stream is
    # unaffected by how the simulations below are scheduled.
    scenarios = Vector{Vector{Tuple{T,Any}}}(undef, options.parallel)
    for i in 1:options.parallel
        @_timeit_threadsafe model.timer_output "sample_scenario" begin
            scenarios[i] = sample_scenario(
                model,
                options.sampling_scheme;
                iteration = length(options.log) + 1,
            )
        end
    end
    forward_trajectory = Vector{Trajectory{T}}(undef, options.parallel)
    _parallel_foreach(options.parallel) do i
        forward_trajectory[i] =
            _forward_trajectory(model, options, scenarios[i], i)
    end
    return forward_trajectory
end

# Internal: simulate one trajectory of the batch along `scenario_path`, solving
# the `replica`-th copy of every node so that the trajectories of a batch can be
# simulated concurrently.
function _forward_trajectory(
    model::PolicyGraph{T},
    options::Options,
    scenario_path::Vector{Tuple{T,Any}},
    replica::Int,
) where {T}
    # Storage for the list of outgoing states that we visit on the forward pass.
    sampled_states = Dict{Symbol,Float64}[]
    # Our initial incoming state.
    incoming_state_value = copy(options.initial_state)
    # A cumulator for the stage-objectives.
    cumulative_value = 0.0
    # Iterate down the scenario.
    for (node_index, noise) in scenario_path
        node = _node(model, node_index, replica)
        lock(node.lock)
        try
            # Solve the subproblem, note that `duality_handler = nothing`.
            @_timeit_threadsafe model.timer_output "solve_subproblem" begin
                subproblem_results = solve_subproblem(
                    model,
                    node,
                    incoming_state_value,
                    noise;
                    duality_handler = nothing,
                )
            end
            # Cumulate the stage_objective.
            cumulative_value += subproblem_results.stage_objective
            # Set the outgoing state value as the incoming state value for the next
            # node.
            incoming_state_value = copy(subproblem_results.state)
            # Add the outgoing state variable to the list of states we have sampled
            # on this forward pass.
            push!(sampled_states, incoming_state_value)
        finally
            unlock(node.lock)
        end
    end
    return Trajectory{T}(
        scenario_path,
        sampled_states,
        cumulative_value,
    )
end
