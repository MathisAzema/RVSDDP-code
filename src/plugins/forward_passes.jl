#  Copyright (c) 2017-25, Oscar Dowson and RVSDDP.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
    DefaultForwardPass(; include_last_node::Bool = true)

The default forward pass.

If `include_last_node = false` and the sample terminated due to a cycle, then
the last node (which forms the cycle) is omitted. This can be useful option to
set when training, but it comes at the cost of not knowing which node formed the
cycle (if there are multiple possibilities).

The `options.parallel` trajectories of the batch are sampled using
`Threads.@threads`, so they run concurrently across the Julia threads available
to the process (start Julia with `julia -t N` or set `JULIA_NUM_THREADS=N` to
get more than one). Access to each node's shared `subproblem` (and to the
per-node `starting_states` list) is serialized through `node.lock`, so this is
safe even though most nodes are visited by several trajectories.
"""
struct DefaultForwardPass <: AbstractForwardPass
    include_last_node::Bool
    function DefaultForwardPass(; include_last_node::Bool = true)
        return new(include_last_node)
    end
end

# mutable struct Trajectory{T}
#     scenario_path::Vector{Tuple{T,NoiseType}},
#     sampled_states::Vector{Dict{Symbol,Float64}},
#     objective_states::Vector{Tuple{}},
#     belief_states::Vector{Tuple{Int,Dict{T,Float64}}}
#     cumulative_value::Float64
# end
# scenario_path::Vector{Tuple{Int64, Float64}}
# sampled_states::Vector{Dict{Symbol, Float64}}
# objective_states::Vector{Tuple{}}
# belief_states::Vector{Tuple{Int64, Dict{Int64, Float64}}}

function forward_pass(
    model::PolicyGraph{T},
    options::Options,
    pass::DefaultForwardPass,
) where {T}

    forward_trajectory = Vector{Trajectory{T}}(undef, options.parallel)
    Threads.@threads for i in 1:options.parallel
        # First up, sample a scenario. Note that if a cycle is detected, this will
        # return the cycle node as well.
        @_timeit_threadsafe model.timer_output "sample_scenario" begin
            scenario_path, terminated_due_to_cycle =
                sample_scenario(model, options.sampling_scheme)
        end
        final_node = scenario_path[end]
        if terminated_due_to_cycle && !pass.include_last_node
            pop!(scenario_path)
        end
        # Storage for the list of outgoing states that we visit on the forward pass.
        sampled_states = Dict{Symbol,Float64}[]
        # Storage for the belief states: partition index and the belief dictionary.
        belief_states = Tuple{Int,Dict{T,Float64}}[]
        # Our initial incoming state.
        incoming_state_value = copy(options.initial_state)
        # A cumulator for the stage-objectives.
        cumulative_value = 0.0
        # Objective state interpolation.
        objective_state_vector, N =
            initialize_objective_state(model[scenario_path[1][1]])
        objective_states = NTuple{N,Float64}[]
        # Iterate down the scenario.
        for (depth, (node_index, noise)) in enumerate(scenario_path)
            node = model[node_index]
            lock(node.lock)
            try
                # Objective state interpolation.
                objective_state_vector = update_objective_state(
                    node.objective_state,
                    objective_state_vector,
                    noise,
                )
                if objective_state_vector !== nothing
                    push!(objective_states, objective_state_vector)
                end
                # ===== Begin: starting state for infinite horizon =====
                starting_states = options.starting_states[node_index]
                if length(starting_states) > 0
                    # There is at least one other possible starting state. If our
                    # incoming state is more than δ away from the other states, add it
                    # as a possible starting state.
                    if distance(starting_states, incoming_state_value) >
                    options.cycle_discretization_delta
                        push!(starting_states, incoming_state_value)
                    end
                    # TODO(odow):
                    # - A better way of randomly sampling a starting state.
                    # - Is is bad that we splice! here instead of just sampling? For
                    #   convergence it is probably bad, since our list of possible
                    #   starting states keeps changing, but from a computational
                    #   perspective, we don't want to keep a list of discretized points
                    #   in the state-space δ distance apart...
                    incoming_state_value =
                        splice!(starting_states, rand(1:length(starting_states)))
                end
                # ===== End: starting state for infinite horizon =====
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
        if terminated_due_to_cycle
            # We terminated due to a cycle. Here is the list of possible
            # starting states for that node. Lock the node because
            # `starting_states` is shared across the concurrently-running
            # trajectories of this batch.
            final_node_object = model[final_node[1]]
            lock(final_node_object.lock)
            try
                starting_states = options.starting_states[final_node[1]]
                # We also need the incoming state variable to the final node, which
                # is the outgoing state value of the second to last node:
                incoming_state_value = if pass.include_last_node
                    sampled_states[end-1]
                else
                    sampled_states[end]
                end
                # If this incoming state value is more than δ away from another
                # state, add it to the list.
                if distance(starting_states, incoming_state_value) >
                options.cycle_discretization_delta
                    push!(starting_states, incoming_state_value)
                end
            finally
                unlock(final_node_object.lock)
            end
        end
        forward_trajectory[i] = Trajectory{T}(
            scenario_path,
            sampled_states,
            objective_states,
            belief_states,
            cumulative_value,
        )    
    end
    # ===== End: drop off starting state if terminated due to cycle =====
    return forward_trajectory
    # return (
    #     scenario_path = scenario_path,
    #     sampled_states = sampled_states,
    #     objective_states = objective_states,
    #     belief_states = belief_states,
    #     cumulative_value = cumulative_value,
    # )
end

