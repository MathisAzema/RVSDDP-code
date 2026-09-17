#  Copyright (c) 2017-25, Oscar Dowson and RVSDDP.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

# The backward pass of RV-SDDP: given the trial states visited by the forward
# pass, walk the scenario path backwards and refine the value function at each
# of them.
#
# One backward step consists of solving the children of the current node at
# every trial state of the batch (`solve_all_children`, accumulating into a
# `BackwardPassItems`), computing a single shift from the whole batch, and
# turning each trajectory's solutions into a cut. `_refine_at_initial_point`
# does the same thing at the initial state, which the forward pass only visits
# once and which `backward_pass` therefore handles separately.

struct BackwardPassItems{T,U}
    "Given a (node, noise) tuple, index the element in the array."
    cached_solutions::Dict{Tuple{T,Any},Int}
    duals::Vector{Dict{Symbol,Float64}}
    supports::Vector{U}
    nodes::Vector{T}
    probability::Vector{Float64}
    objectives::Vector{Float64}
    function BackwardPassItems(T, U)
        return new{T,U}(
            Dict{Tuple{T,Any},Int}(),
            Dict{Symbol,Float64}[],
            U[],
            T[],
            Float64[],
            Float64[],
        )
    end
end

function solve_one_children(
    model::PolicyGraph{T},
    node::Node{T},
    items::BackwardPassItems,
    incoming_state::Dict{Symbol,Float64},
    backward_sampling_scheme::AbstractBackwardSamplingScheme,
    duality_handler::Union{Nothing,AbstractDualityHandler},
    options,
) where {T}
    lock(node.lock)
    try
        @_timeit_threadsafe model.timer_output "prepare_backward_pass" begin
            restore_duality = prepare_backward_pass(
                node,
                options.duality_handler,
                options,
            )
        end
        for noise in sample_backward_noise_terms_with_state(
            backward_sampling_scheme,
            node,
            incoming_state,
        )
            if haskey(items.cached_solutions, (node.index, noise.term))
                sol_index = items.cached_solutions[(node.index, noise.term)]
                push!(items.duals, items.duals[sol_index])
                push!(items.supports, items.supports[sol_index])
                push!(items.nodes, node.index)
                push!(items.probability, items.probability[sol_index])
                push!(items.objectives, items.objectives[sol_index])
            else
                @_timeit_threadsafe model.timer_output "solve_subproblem" begin
                    subproblem_results = solve_subproblem(
                        model,
                        node,
                        incoming_state,
                        noise.term;
                        duality_handler = duality_handler,
                    )
                end
                push!(items.duals, subproblem_results.duals)
                push!(items.supports, noise)
                push!(items.nodes, node.index)
                push!(
                    items.probability,
                    noise.probability,
                )
                push!(items.objectives, subproblem_results.objective)
                items.cached_solutions[(node.index, noise.term)] =
                    length(items.duals)
            end
        end
        @_timeit_threadsafe model.timer_output "prepare_backward_pass" begin
            restore_duality()
        end
    finally
        unlock(node.lock)
    end
    return
end

"""
    solve_all_children(model, node, items, outgoing_state, ...; replica)

Solve every child of `node` at `outgoing_state` and accumulate the results in
`items`.

`replica` selects which copy of each child subproblem to solve, so that the
`options.parallel` calls made at a given stage of the backward pass can run
concurrently without sharing a JuMP model. See `_node`.
"""
function solve_all_children(
    model::PolicyGraph{T},
    node::Node{T},
    items::BackwardPassItems,
    outgoing_state::Dict{Symbol,Float64},
    backward_sampling_scheme::AbstractBackwardSamplingScheme,
    duality_handler::Union{Nothing,AbstractDualityHandler},
    options;
    replica::Int = 1,
) where {T}
    for child in node.children
        child_node = _node(model, child.term, replica)
        solve_one_children(
            model,
            child_node,
            items,
            outgoing_state,
            backward_sampling_scheme,
            duality_handler,
            options,
        )
    end
    return
end

# Internal function: generate the cut at the initial state x0, which
# `backward_pass` does once per iteration whatever the refinement scheme.
#
# Unlike the levels of the forward pass, this one is not batched: every
# trajectory of a `parallel > 1` batch starts from the same x0, so solving the
# children once is enough -- a batch would only produce `parallel` identical
# cuts. The shift is therefore computed from a batch of one.
function _refine_at_initial_point(
    model::PolicyGraph{T},
    options::Options,
) where {T}
    if options.infinite
        node_index = length(model.nodes)
        node =  model[node_index]
        next_node = model[node.children[1].term]
        items = BackwardPassItems(T, Noise)
        outgoing_state = model.initial_root_state

        solve_all_children(
            model,
            node,
            items,
            outgoing_state,
            options.backward_sampling_scheme,
            options.duality_handler,
            options,
        )
        shift=options.shift_function(model, next_node, [items], [outgoing_state])

        record_bellman_residual!(next_node, outgoing_state, items.probability, items.objectives)

        new_cuts = refine_bellman_function(
            model,
            node,
            node.bellman_function,
            options.risk_measure,
            outgoing_state,
            items.duals,
            items.supports,
            items.probability,
            items.objectives,
            shift,
            length(options.log)+1,
            time()-options.start_time,
        )

    else
        node_index = 1
        node =  model[node_index]
        items = BackwardPassItems(T, Noise)
        incoming_state = model.initial_root_state

        solve_one_children(
            model,
            node,
            items,
            incoming_state,
            options.backward_sampling_scheme,
            options.duality_handler,
            options,
        )
        shift=options.shift_function(model, node, [items], [incoming_state])

        πᵏ = Dict(key => 0.0 for key in keys(incoming_state))
        θᵏ = 0.0
        objective_realizations = items.objectives
        risk_adjusted_probability = items.probability
        dual_variables = items.duals
        for i in 1:length(objective_realizations)
            p = risk_adjusted_probability[i]
            θᵏ += p * objective_realizations[i]
            for (key, dual) in dual_variables[i]
                πᵏ[key] += p * dual
            end
        end

        for (key, x) in incoming_state
            θᵏ -= πᵏ[key] * x
        end

        iteration = length(options.log)+1

        cut = CandidateCut(iteration, time() - options.start_time, θᵏ, πᵏ, incoming_state)

        _update_value_function(node, cut, shift, nothing)
        record_bellman_residual!(node, incoming_state, items.probability, items.objectives)
        return []
    end
    return new_cuts
end

function backward_pass(
    model::PolicyGraph{T},
    options::Options,
    trajectory::Vector{Trajectory{T}},
) where {T}
    scenario_length = length(trajectory[1].scenario_path)
    period= length(model.nodes)

    index_to_refine = options.refine_scheme(scenario_length, period)
    # One Bellman residual per phase per iteration. The refinement at x0 below
    # always records the one for the first phase, so the loop covers the other
    # `period - 1` phases, i.e. the first `period - 1` refined levels.
    # (`index_to_refine` is empty when the forward pass is too short to hold a
    # level, e.g. `refine_all` on a pass of length 1; the loop below then does
    # nothing and only the refinement at x0 runs.)
    first_level = isempty(index_to_refine) ? 1 : minimum(index_to_refine)
    last_residual_level = first_level + period - 2
    cuts = Dict{T,Vector{Any}}(index => Any[] for index in keys(model.nodes))
    for index in scenario_length:-1:1
        node_index, _ = trajectory[1].scenario_path[index]
        node =  model[node_index]
        if length(node.children) == 0
            continue
        end
        if index in index_to_refine
            items_traj = [BackwardPassItems(T, Noise) for _ in trajectory]
            outgoing_states = [traj.sampled_states[index] for traj in trajectory]
            # Solve the children for every trajectory of the batch in parallel:
            # trajectory `j` solves replica `j` of each child and fills its own
            # `items_traj[j]`, so the tasks share nothing. They are all joined
            # before the common shift below is computed.
            _parallel_foreach(length(trajectory)) do index_traj
                solve_all_children(
                    model,
                    node,
                    items_traj[index_traj],
                    outgoing_states[index_traj],
                    options.backward_sampling_scheme,
                    options.duality_handler,
                    options;
                    replica = index_traj,
                )
            end

            # From here on we are back on a single task: the shift is computed
            # once from the whole batch, and the cuts it produces mutate the
            # master model (and, through `_add_cut_constraint_to_model`, every
            # replica), so they have to be added one at a time.
            next_node = model[node.children[1].term]
            shift=options.shift_function(model, next_node, items_traj, outgoing_states)
            if index <= last_residual_level
                outgoing_state = outgoing_states[1]
                items = items_traj[1]
                record_bellman_residual!(next_node, outgoing_state, items.probability, items.objectives)
            end
            for (index_traj, traj) in enumerate(trajectory)
                outgoing_state = outgoing_states[index_traj]
                items = items_traj[index_traj]
                new_cuts = refine_bellman_function(
                    model,
                    node,
                    node.bellman_function,
                    options.risk_measure,
                    outgoing_state,
                    items.duals,
                    items.supports,
                    items.probability,
                    items.objectives,
                    shift,
                    length(options.log)+1,
                    time()-options.start_time,
                )

                push!(cuts[node_index], new_cuts)
            end
        end
    end
    # The cut at the initial state is generated at every iteration, whatever the
    # refinement scheme: it is what puts x0 in the stable set, which the
    # convergence analysis needs. It is deliberately not batched, see
    # `_refine_at_initial_point`.
    new_cuts_0 = _refine_at_initial_point(model, options)
    push!(cuts[length(model.nodes)], new_cuts_0)
    push!(model.approx_value, (time()-options.start_time, compute_approx_value(model)))
    return cuts
end
