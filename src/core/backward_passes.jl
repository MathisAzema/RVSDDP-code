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

# Internal: whether the shift rule's candidates can be evaluated at the same time
# as the batch. That takes one more task per candidate, so it is only done when
# there is a core for every task and a replica of `next_node` for every
# candidate; otherwise the rule evaluates them itself, sequentially, after the
# batch (where it can skip those that cannot lower the shift).
function _concurrent_candidates(next_node::Node, n_trajectories::Int, n_candidates::Int)
    return n_candidates > 0 &&
           Threads.nthreads() >= n_trajectories + n_candidates &&
           length(next_node.replicas) >= n_trajectories + n_candidates - 1
end

"""
    _solve_batch(model, node, trial_states, level, options)

Solve the children of `node` at every trial state of the batch and, when a core
is free for it, evaluate `T(V)` at the candidates of the shift rule at the same
time. Returns the `BackwardPassItems` of each trajectory and the
`ShiftCandidate`s, ready to be handed to the shift rule. `level` is the level
being refined, under which the timings of the step are recorded (see
[`RVSDDP.parallel_timings`](@ref)).

Every task works on its own subproblems: trajectory `j` solves replica `j` of
each child, and candidate `c` replica `n + c` of `node`'s child, `n` being the
batch size. Once they are all joined, all that is left for the rule is a
minimum.
"""
function _solve_batch(
    model::PolicyGraph{T},
    node::Node{T},
    trial_states::Vector{Dict{Symbol,Float64}},
    level::Int,
    options::Options,
) where {T}
    next_node = model[node.children[1].term]
    n_trajectories = length(trial_states)
    # Drawn before the solves, on this task: the solves consume no randomness,
    # so the random stream is the same as when the rule drew them afterwards.
    candidates = [
        ShiftCandidate(state, nothing) for
        state in shift_candidates(options.shift_function, next_node)
    ]
    n_concurrent =
        _concurrent_candidates(next_node, n_trajectories, length(candidates)) ?
        length(candidates) : 0
    items = [BackwardPassItems(T, Noise) for _ in 1:n_trajectories]
    timings = _parallel_foreach_timed(n_trajectories + n_concurrent) do task
        if task <= n_trajectories
            solve_all_children(
                model,
                node,
                items[task],
                trial_states[task],
                options.backward_sampling_scheme,
                options.duality_handler,
                options;
                replica = task,
            )
        else
            candidate = candidates[task-n_trajectories]
            candidate.TV = compute_TV(
                _node(model, next_node.index, task),
                candidate.state,
            )
        end
    end
    _record_parallel_timings!(model, options, "backward", level, timings, n_trajectories)
    return items, candidates
end

# Internal function: generate the cut at the initial state x0, i.e. refine
# level 1, which `backward_pass` does whenever the refinement scheme selects it.
#
# Unlike the other levels of the forward pass, this one is not batched: every
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
        outgoing_state = model.initial_root_state

        batch_items, candidates = _solve_batch(model, node, [outgoing_state], 1, options)
        items = batch_items[1]
        # Record the residual before the shift rule moves the cuts of
        # `next_node`, so that it is the residual of the current iterate.
        record_bellman_residual!(next_node, outgoing_state, items.probability, items.objectives)
        shift=options.shift_function(model, next_node, batch_items, [outgoing_state], candidates)

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
        record_bellman_residual!(node, incoming_state, items.probability, items.objectives)
        # Finite horizon: the candidates are evaluated by the rule itself.
        candidates = [
            ShiftCandidate(state, nothing) for
            state in shift_candidates(options.shift_function, node)
        ]
        shift=options.shift_function(model, node, [items], [incoming_state], candidates)

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

    # The levels that receive a cut, numbered as in the paper: level `i` is the
    # trial state `x^i`, so level 1 is x0 and level `i >= 2` is the state
    # `sampled_states[i-1]` the forward pass reached after its `(i-1)`-th solve.
    # See `refinement_schemes.jl`.
    levels_to_refine = options.refine_scheme(scenario_length, period)
    # One Bellman residual per phase per iteration: those of the first `period`
    # refined levels, which cover each phase once.
    first_level = isempty(levels_to_refine) ? 1 : minimum(levels_to_refine)
    last_residual_level = first_level + period - 1
    cuts = Dict{T,Vector{Any}}(index => Any[] for index in keys(model.nodes))
    for level in scenario_length:-1:2
        if !(level in levels_to_refine)
            continue
        end
        # The state `x^level` is the outgoing state of the `(level-1)`-th node
        # of the path, and its cut goes into that node's cost-to-go.
        index = level - 1
        node_index, _ = trajectory[1].scenario_path[index]
        node =  model[node_index]
        if length(node.children) == 0
            continue
        end
        outgoing_states = [traj.sampled_states[index] for traj in trajectory]
        # Solve the children for every trajectory of the batch, and evaluate the
        # shift rule's candidates, in parallel. See `_solve_batch`.
        items_traj, candidates = _solve_batch(model, node, outgoing_states, level, options)

        # From here on we are back on a single task: the shift is computed
        # once from the whole batch, and the cuts it produces mutate the
        # master model (and, through `_add_cut_constraint_to_model`, every
        # replica), so they have to be added one at a time.
        next_node = model[node.children[1].term]
        # Record the residual before the shift rule moves the cuts of
        # `next_node`, so that it is the residual of the current iterate.
        if level <= last_residual_level
            record_bellman_residual!(next_node, outgoing_states[1], items_traj[1].probability, items_traj[1].objectives)
        end
        shift=options.shift_function(model, next_node, items_traj, outgoing_states, candidates)
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
    # Level 1, the initial state, comes last, as in the backward pass of
    # Algorithm 1. `refine_all` refines it at every iteration, which is what puts
    # x0 in the stable set of the convergence analysis; `refine_periodic` only
    # when it draws the first block. It is deliberately not batched, see
    # `_refine_at_initial_point`.
    if 1 in levels_to_refine
        new_cuts_0 = _refine_at_initial_point(model, options)
        push!(cuts[length(model.nodes)], new_cuts_0)
    end
    push!(model.approx_value, (time()-options.start_time, compute_approx_value(model)))
    return cuts
end
