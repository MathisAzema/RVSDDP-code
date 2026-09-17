#  Copyright (c) 2017-25, Oscar Dowson and RVSDDP.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v. 2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

# Shift-selection rules --- the "RV" of RV-SDDP.
#
# A Bellman-greedy policy only sees *differences* of the value function, so a
# downstream value function may be sought up to an additive constant. RV-SDDP
# exploits this: after each backward pass it picks a shift Δ >= 0 and lowers the
# cuts by it, which keeps the approximation focused on the policy-relevant shape
# of V instead of first learning its absolute level (of order 1/(1-β), hence
# very large when β is close to one).
#
# A shift rule is any function
#
#     rule(model, node, batch_items, trial_states) -> (Δ, cut_index)
#
# passed to `train(; shift_function = rule)`. It is called once per backward
# step, after the children of `node` have been solved for every trajectory of
# the batch, and it is responsible for applying its own shift through
# `apply_shift!`. The returned `cut_index` is the cut count at which the shift
# takes effect, and is recorded in the shift history of the cut being created.
#
# A rule must keep the shift sequence *nonnegative* and *nonanticipative*: Δ >= 0
# always, and Δ may only depend on information available at that step. Those are
# the two conditions the convergence analysis rests on.

# ---------------------------------------------------------------------------
# The Bellman residual
# ---------------------------------------------------------------------------

"""
    bellman_residual(node, state, child_probabilities, child_objectives)

The Bellman residual `T(V)(state) - V(state)` at one state.

`T(V)(state)` is not re-solved: it is recovered as the expectation of
`child_objectives` under `child_probabilities`, both of which the backward pass
has just computed by solving the children of `node` at `state`.

This residual is the quantity every shift rule minimises, and also what
[`RVSDDP.record_bellman_residual!`](@ref) stores for diagnostics.
"""
function bellman_residual(
    node::Node,
    state::Dict{Symbol,Float64},
    child_probabilities::Vector{Float64},
    child_objectives::Vector{Float64},
)
    expected_cost_to_go = 0.0
    for i in 1:length(child_objectives)
        expected_cost_to_go += child_probabilities[i] * child_objectives[i]
    end
    return expected_cost_to_go - compute_V(node.value_function, state)
end

"""
    record_bellman_residual!(node, state, child_probabilities, child_objectives)

Append the Bellman residual at `state` to `node.delta`.

This is bookkeeping only --- nothing in the algorithm reads `node.delta` back.
It is the sequence the experiments plot next to the selected shifts.
"""
function record_bellman_residual!(
    node::Node{T},
    state::Dict{Symbol,Float64},
    child_probabilities::Vector{Float64},
    child_objectives::Vector{Float64},
) where {T}
    push!(
        node.delta,
        bellman_residual(node, state, child_probabilities, child_objectives),
    )
    return
end

# ---------------------------------------------------------------------------
# Applying a shift
# ---------------------------------------------------------------------------

"""
    next_cut_index(node)

The index the next cut of `node` will be given.

Shifts are stamped with this index, which is what lets a cut tell later which
shifts predate it --- see `AttachedCut.shift` and the cut replay in `_add_cuts`.
"""
next_cut_index(node::Node) = length(node.value_function.cut_V) + 1

"""
    apply_shift!(model, node, shift)

Bring the effective shift of every cut of `node` down to `shift`, wherever the
cut currently carries a larger one, and stamp the new level into its shift
history.

The direction is easy to get backwards. A cut is stored at `intercept - shift`,
so *reducing* its shift moves the cut itself back *up*, towards the unshifted
position it was generated at. A cut's effective shift is therefore the smallest
one selected since it was created, which is why a rule has to revisit the whole
cut collection and not just the newest cut.
"""
function apply_shift!(
    model::PolicyGraph{T},
    node::Node{T},
    shift::Float64,
) where {T}
    effective_from = next_cut_index(node)
    raised = AttachedCut[]
    for cut in node.value_function.cut_V
        if shift < cut.shift[end][1]
            push!(cut.shift, (shift, effective_from))
            set_normalized_rhs(cut.constraint_V, cut.intercept - shift)
            if cut.constraint_subproblem !== nothing
                set_normalized_rhs(cut.constraint_subproblem, cut.intercept - shift)
            end
            push!(raised, cut)
        end
    end
    # Mirror the same move onto the replicas' copies of those cuts. Task `r`
    # only ever touches replica `r`, so the refreshes run concurrently. This
    # matters because a shift can touch every cut generated so far.
    if !isempty(raised)
        n_replicas = maximum(length(cut.constraint_replicas) for cut in raised)
        _parallel_foreach(n_replicas) do r
            for cut in raised
                if r <= length(cut.constraint_replicas)
                    set_normalized_rhs(
                        cut.constraint_replicas[r],
                        cut.intercept - shift,
                    )
                end
            end
        end
    end
    return
end

# ---------------------------------------------------------------------------
# The rules
# ---------------------------------------------------------------------------

# Each rule below is followed by the short name its results directory is built
# from (see `shift_label` in headers.jl). A rule that defines no label falls back
# to its function name.
shift_label(rule::Function) = string(rule)

#Mathis attention il faudrait un shift pour chaque enfant
"""
    no_shift(model, node, batch_items, trial_states)

The baseline rule: never shift. RV-SDDP then degenerates into the plain cyclic
SDDP scheme, whose cuts keep their unshifted position and whose value function
therefore still provides a converging lower bound.
"""
function no_shift(
    model::PolicyGraph{T},
    node::Node{T},
    batch_items::Vector{BackwardPassItems{T,Noise}},
    trial_states::Vector{Dict{Symbol,Float64}},
) where {T}
    return (0.0, next_cut_index(node))
end

shift_label(::typeof(no_shift)) = "cyclic_sddp"

# Draw one state uniformly in the node's state box, keyed by the node's own state
# variables.
function _random_state_in_box(node::Node)
    state = Dict{Symbol,Float64}()
    for key in keys(node.states)
        lower = node.state_lower_bounds[key]
        upper = node.state_upper_bounds[key]
        state[key] = rand() * (upper - lower) + lower
    end
    return state
end

"""
    random_shift(model, node, batch_items, trial_states)

The random-shift rule used in the experiments: the shift is the smallest Bellman
residual over the trial states of the batch and one extra state drawn uniformly
at random in the state box.

Keeping the trial states as candidates is what makes a newly generated cut
active where it was generated; the random state is what makes the rule explore
the whole state space, which is what the almost-sure convergence argument needs.

The random state is only paid for when it can actually win. Its residual is
first bounded below using the cut envelope alone (`compute_approx_TV`, no
solve), and the exact Bellman value (`compute_TV`, one LP per noise) is computed
only if that bound still beats the best trial-state residual.
"""
function random_shift(
    model::PolicyGraph{T},
    node::Node{T},
    batch_items::Vector{BackwardPassItems{T,Noise}},
    trial_states::Vector{Dict{Symbol,Float64}},
) where {T}
    # Candidates 1..n: the trial states the batch has just cut at.
    trial_residuals = zeros(length(batch_items))
    for (i, items) in enumerate(batch_items)
        trial_residuals[i] = bellman_residual(
            node,
            trial_states[i],
            items.probability,
            items.objectives,
        )
    end
    shift = minimum(trial_residuals)

    # Candidate n+1: a uniform draw in the state box, screened before solving.
    candidate = _random_state_in_box(node)
    V_candidate = compute_V(node.value_function, candidate)
    residual_bound = compute_approx_TV(node.value_function, candidate) - V_candidate
    if residual_bound < shift
        shift = min(shift, compute_TV(node, candidate) - V_candidate)
    end

    apply_shift!(model, node, shift)
    return (shift, next_cut_index(node))
end

shift_label(::typeof(random_shift)) = "RVSDDP"
