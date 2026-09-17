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
#     rule(model, node, items_traj, outgoing_states) -> (Δ, cut_index)
#
# passed to `train(; shift_function = rule)`. It is called once per backward
# step, after the children of `node` have been solved for every trajectory of
# the batch, and it is responsible for applying its own shift through
# `update_shift`. The returned `cut_index` is the cut count at which the shift
# takes effect, and is recorded in the shift history of the cut being created.
#
# A rule must keep the shift sequence *nonnegative* and *nonanticipative*: Δ >= 0
# always, and Δ may only depend on information already available at that step.
# Those are the two conditions the convergence analysis rests on.

# The Bellman residual T(V)(x) - V(x) at one trial state, where T(V)(x) is
# recovered from the children's objectives already computed by the backward
# pass. This is the quantity the shift rules minimise, and also the per-iteration
# residual δ recorded in `node.delta`.
function bellman_residual(
    node::Node,
    state::Dict{Symbol,Float64},
    probability::Vector{Float64},
    objectives::Vector{Float64},
)
    TVx = 0.0
    for i in 1:length(objectives)
        TVx += probability[i] * objectives[i]
    end
    return TVx - compute_V(node.value_function, state)
end

# Record the Bellman residual at this step for diagnostics; `node.delta` is what
# the experiments plot as the shift sequence.
function _update_delta(
    node::Node{T},
    incoming_state::Dict{Symbol,Float64},
    risk_adjusted_probability::Vector{Float64},
    objective_realizations::Vector{Float64},
) where {T}
    push!(
        node.delta,
        bellman_residual(
            node,
            incoming_state,
            risk_adjusted_probability,
            objective_realizations,
        ),
    )
    return
end

"""
    update_shift(model, node, shift_k)

Lower every cut of `node` to the shift level `shift_k`, and record that level in
each cut's shift history.

A cut is only ever moved *down*, so its effective shift is the smallest one
selected since it was created: picking a small `shift_k` raises all the earlier
cuts back towards their unshifted position, while later cuts are untouched. This
is why a rule shifts the whole cut collection and not just the newest cut.
"""
function update_shift(
    model::PolicyGraph{T},
    node::Node{T},
    shift_k::Float64,
) where {T}
    iter = length(node.value_function.cut_V)+1
    shifted = Cut2[]
    for cut in node.value_function.cut_V
        if shift_k < cut.shift[end][1]
            push!(cut.shift, (shift_k, iter))
            set_normalized_rhs(cut.constraint_V, cut.intercept-shift_k)
            if cut.constraint_subproblem !== nothing
                set_normalized_rhs(cut.constraint_subproblem, cut.intercept-shift_k)
            end
            push!(shifted, cut)
        end
    end
    # Bring the replicas' copies of those cuts down to the same level. Task `r`
    # only ever touches replica `r`, so the refreshes run concurrently. This
    # matters because a shift can touch every cut generated so far.
    if !isempty(shifted)
        n_replicas = maximum(length(cut.constraint_replicas) for cut in shifted)
        _parallel_foreach(n_replicas) do r
            for cut in shifted
                if r <= length(cut.constraint_replicas)
                    set_normalized_rhs(
                        cut.constraint_replicas[r],
                        cut.intercept-shift_k,
                    )
                end
            end
        end
    end
    return
end

#Mathis attention il faudrait un shift pour chaque enfant
"""
    no_shift(model, node, items_traj, outgoing_states)

The baseline rule: never shift. RV-SDDP then degenerates into the plain cyclic
SDDP scheme, whose cuts keep their unshifted position and whose value function
therefore still provides a converging lower bound.
"""
function no_shift(
    model::PolicyGraph{T},
    node::Node{T},
    items_traj::Vector{BackwardPassItems{T, Noise}},
    outgoing_states::Vector{Dict{Symbol, Float64}},
) where {T}
    return (0.0, length(node.value_function.cut_V)+1)
end

"""
    shift_update_random_forward(model, node, items_traj, outgoing_states)

The random-shift rule used in the experiments: take the smallest Bellman
residual over the trial states of the batch and one extra state drawn uniformly
at random in the state box.

Including the trial states is what makes the newly generated cut active where it
is generated; the random state is what makes the rule explore the whole state
space, which is what the almost-sure convergence argument needs. The random
state is only paid for when it can actually win: its residual is first bounded
below using the cut envelope alone (`compute_approx_TV`, cheap), and the exact
Bellman value (`compute_TV`, one LP per noise) is computed only if that bound is
still better than the best trial-state residual.
"""
function shift_update_random_forward(
    model::PolicyGraph{T},
    node::Node{T},
    items_traj::Vector{BackwardPassItems{T, Noise}},
    outgoing_states::Vector{Dict{Symbol, Float64}},
) where {T}
    # Best residual over the trial states of the batch.
    res_traj = zeros(length(items_traj))
    for (i, items) in enumerate(items_traj)
        res_traj[i] = bellman_residual(
            node,
            outgoing_states[i],
            items.probability,
            items.objectives,
        )
    end
    shift, _ = findmin(res_traj)

    # One extra candidate, drawn uniformly in the state box.
    sol=Dict{Symbol,Float64}()
    for (i, _) in outgoing_states[1]
        lb=node.state_lower_bounds[i]
        ub=node.state_upper_bounds[i]
        sol[i]=rand()*(ub-lb)+lb
    end
    Vrand=compute_V(node.value_function, sol)
    TVrandapprox = compute_approx_TV(node.value_function, sol)
    shift_rand=Inf
    if TVrandapprox-Vrand<= shift-1e-4
        TVrand=compute_TV(node, sol)
        shift_rand=TVrand-Vrand
    end
    shift = min(shift, shift_rand)
    update_shift(model, node, shift)

    return (shift, length(node.value_function.cut_V)+1)
end
