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
struct DefaultForwardPass <: AbstractForwardPass
    include_last_node::Bool
    function DefaultForwardPass(; include_last_node::Bool = true)
        return new(include_last_node)
    end
end

function forward_pass(
    model::PolicyGraph{T},
    options::Options,
    pass::DefaultForwardPass,
) where {T}
    # Sample the whole batch first, on this task, so that the random stream is
    # unaffected by how the simulations below are scheduled.
    scenarios = Vector{Tuple{Vector{Tuple{T,Any}},Bool}}(undef, options.parallel)
    for i in 1:options.parallel
        @_timeit_threadsafe model.timer_output "sample_scenario" begin
            scenarios[i] = sample_scenario(model, options.sampling_scheme)
        end
    end
    forward_trajectory = Vector{Trajectory{T}}(undef, options.parallel)
    _parallel_foreach(options.parallel) do i
        scenario_path, terminated_due_to_cycle = scenarios[i]
        forward_trajectory[i] = _forward_trajectory(
            model,
            options,
            pass,
            scenario_path,
            terminated_due_to_cycle,
            i,
        )
    end
    return forward_trajectory
end

# Internal: simulate one trajectory of the batch along `scenario_path`, solving
# the `replica`-th copy of every node so that the trajectories of a batch can be
# simulated concurrently.
function _forward_trajectory(
    model::PolicyGraph{T},
    options::Options,
    pass::DefaultForwardPass,
    scenario_path::Vector{Tuple{T,Any}},
    terminated_due_to_cycle::Bool,
    replica::Int,
) where {T}
    final_node = scenario_path[end]
    if terminated_due_to_cycle && !pass.include_last_node
        pop!(scenario_path)
    end
    # Storage for the list of outgoing states that we visit on the forward pass.
    sampled_states = Dict{Symbol,Float64}[]
    # Our initial incoming state.
    incoming_state_value = copy(options.initial_state)
    # A cumulator for the stage-objectives.
    cumulative_value = 0.0
    # Iterate down the scenario.
    for (depth, (node_index, noise)) in enumerate(scenario_path)
        node = _node(model, node_index, replica)
        lock(node.lock)
        try
            # ===== Begin: starting state for infinite horizon =====
            starting_states = options.starting_states[node_index]
            if length(starting_states) > 0
                # There is at least one other possible starting state. If our
                # incoming state is more than δ away from the other states, add it
                # as a possible starting state.
                #
                # Unlike the subproblem, this list is shared by the whole batch,
                # so it is guarded by the *master* node's lock rather than by
                # `node.lock` (which, for `replica > 1`, is the replica's own).
                master_node = model[node_index]
                lock(master_node.lock)
                try
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
                finally
                    unlock(master_node.lock)
                end
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
        # starting states for that node. Lock the *master* node because
        # `starting_states` is shared across the trajectories of the batch,
        # which each hold their own replica's lock.
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
    # ===== End: drop off starting state if terminated due to cycle =====
    return Trajectory{T}(
        scenario_path,
        sampled_states,
        cumulative_value,
    )
end
