#  Copyright (c) 2017-25, Oscar Dowson and RVSDDP.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Everything that surrounds a node's `ValueFunction`: how it is built, how it
# is evaluated, how many of its cuts are active, and how a run saved to CSV is
# replayed into a fresh model.

# `CSV.read(path, DataFrame)` below needs the unqualified name.
using DataFrames

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

"""
    record_state_bounds!(node::Node)

Record the box `[lower_bound, upper_bound]` of each outgoing state variable of
`node`.

That box is the only thing the algorithm needs to know about the geometry of the
state space: `random_shift` draws the random shift candidate `ζ`
uniformly inside it, and `add_state_variables_to_value_function` uses it to bound
the value function's own copy of the state.
"""
function record_state_bounds!(node::Node)
    for (key, state) in node.states
        node.state_lower_bounds[key] = JuMP.lower_bound(state.out)
        node.state_upper_bounds[key] = JuMP.upper_bound(state.out)
    end
    return
end

function initialize_value_function(sense::Symbol, optimizer = nothing)
    model = optimizer === nothing ? JuMP.Model() : JuMP.Model(optimizer)
    set_silent(model)
    theta = @variable(model, V)
    if sense == :Min
        @objective(model, Min, theta)
    else
        @objective(model, Max, theta)
    end

    #TV
    model_TV = optimizer === nothing ? JuMP.Model() : JuMP.Model(optimizer)
    set_silent(model_TV)
    theta_TV = @variable(model_TV, TV)
    if sense == :Min
        @objective(model_TV, Min, theta_TV)
    else
        @objective(model_TV, Max, theta_TV)
    end

    return ValueFunction(
        model,
        AttachedCut[],
        theta,
        Dict{Symbol,JuMP.VariableRef}(),
        model_TV,
        theta_TV,
        Dict{Symbol,JuMP.VariableRef}(),
        Dict{Symbol,JuMP.Float64}(),
    )
end

# Give the value function its own copy of the state variables, boxed by the
# bounds `record_state_bounds!` read off the subproblem. Must run after it.
function add_state_variables_to_value_function(node::Node)
    value_function = node.value_function

    y = @variable(value_function.model, [1:length(node.states)])
    state_variables = [(key, var.in) for (key, var) in node.states]
    for (src, dest) in zip(state_variables, y)
        name = JuMP.name(src[2]) #MATHIS: checker si name exist
        JuMP.set_name(dest, name)
        value_function.states[src[1]] = dest
        lb = node.state_lower_bounds[src[1]]
        ub = node.state_upper_bounds[src[1]]
        value_function.heuristic_state[src[1]] = (ub + lb) / 2
        @constraint(value_function.model, dest >= lb)
        @constraint(value_function.model, dest <= ub)
    end

    #TV
    y_TV = @variable(value_function.model_TV, [1:length(node.states)])
    state_variables = [(key, var.in) for (key, var) in node.states]
    for (src, dest) in zip(state_variables, y_TV)
        name = JuMP.name(src[2]) #MATHIS: checker si name exist
        JuMP.set_name(dest, name)
        value_function.states_TV[src[1]] = dest
        lb = node.state_lower_bounds[src[1]]
        ub = node.state_upper_bounds[src[1]]
        @constraint(value_function.model_TV, dest >= lb)
        @constraint(value_function.model_TV, dest <= ub)
    end
    return
end

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

# V(x): the cut envelope at its current, shifted position.
function compute_V(vf::ValueFunction, incoming_state::Dict{Symbol,Float64})
    return maximum([
        cut.intercept - cut.shift[end][1] +
        sum(cut.coefficients[i] * x for (i, x) in incoming_state)
        for cut in vf.cut_V
    ])
end

# The same envelope with the shifts undone, i.e. the cheap lower estimate of
# T(V)(x) that `random_shift` screens candidates with before
# paying for a real Bellman evaluation (Proposition 10).
function compute_approx_TV(vf::ValueFunction, incoming_state::Dict{Symbol,Float64})
    return maximum([
        cut.intercept + sum(cut.coefficients[i] * x for (i, x) in incoming_state)
        for cut in vf.cut_V
    ])
end

# T(V)(x): one exact Bellman evaluation, solving the subproblem once per noise.
function compute_TV(node::Node, incoming_state::Dict{Symbol,Float64})
    TVx = 0.0
    model = node.subproblem
    set_incoming_state(node, incoming_state)
    for noise in node.noise_terms
        parameterize(node, noise.term)
        JuMP.optimize!(model)
        TVx += noise.probability * JuMP.objective_value(model)
    end
    return TVx
end

# ---------------------------------------------------------------------------
# Active cuts
# ---------------------------------------------------------------------------

function is_active(
    node::Node,
    intercept::Float64,
    coef::Dict{Symbol,Float64},
    tol::Float64,
)
    vf = node.value_function
    model = node.value_function.model
    @objective(model, Max, intercept - vf.theta + sum(a * vf.states[i] for (i, a) in coef))
    JuMP.optimize!(model)
    if JuMP.objective_value(model) >= -tol
        return 1
    else
        return 0
    end
end

function count_active_cuts(node::Node, tol::Float64)
    active_cuts = 0
    for (k, cutb) in enumerate(node.value_function.cut_V)
        interceptb = cutb.intercept
        coefficientb = cutb.coefficients
        active_cuts += is_active(node, interceptb - cutb.shift[end][1], coefficientb, tol)
    end
    return active_cuts
end

function count_all_active_cuts(model::PolicyGraph{T}, tol::Float64) where {T}
    res = [0.0 for (index, node) in model.nodes]
    for (index, node) in model.nodes
        res[index] = count_active_cuts(node, tol)
    end
    return res
end

# ---------------------------------------------------------------------------
# Replaying a saved run
# ---------------------------------------------------------------------------

function reconstruct_cuts(df)
    cuts = []
    for row in eachrow(df)
        cut = (
            iteration = row.iteration,
            time = row.time,
            node = row.node,
            intercept = row.intercept,
            coefficients = JSON.parse(row.coefficients),
            shift = JSON.parse(row.shift),
            state = JSON.parse(row.state),
        )
        push!(cuts, cut)
    end
    return cuts
end

"""
    _add_cuts(model, folder, keep; reference_iteration)

Replay into `model` the cuts of the run saved under `folder` that satisfy
`keep(cut)`, rebuilding each one in the three places that hold it: the value
function's own model, its unshifted `model_TV` twin, and the *previous* node's
subproblem (where the cut bounds that node's cost-to-go).

A cut is replayed at the shift it had reached *at that point of the saved run*,
not at its final one: `limit[n]` counts how many cuts node `n` had received by
then, and any shift `(value, cut_count)` recorded beyond that count had not
happened yet and is dropped.

`reference_iteration` overrides the iteration used to truncate
`approx_values.csv` and `deltas.csv`; by default that is the largest iteration
among the kept cuts.

The two `_add_cuts_*` entry points below differ only in that argument.
"""
function _add_cuts(
    model::PolicyGraph,
    folder::String,
    keep::Function;
    reference_iteration::Union{Int,Nothing} = nothing,
)
    if !isfile("$(folder)/cuts.csv")
        println("Fichier $folder non trouvé")
        return
    end
    cuts = reconstruct_cuts(CSV.read("$(folder)/cuts.csv", DataFrame))

    T = length(model.nodes)
    limit = Dict()
    iteration_max = 0
    for node_index in keys(model.nodes)
        lim = 0
        for cut in cuts
            if keep(cut) && cut.node == node_index
                lim += 1
                iteration_max = max(iteration_max, cut.iteration)
            end
        end
        limit[node_index] = lim
    end

    for cut in cuts
        if cut.iteration < 1 || !keep(cut)
            continue
        end
        node_index = cut.node
        node = model[node_index]
        vf = node.value_function
        index = node.index == 1 ? T : node.index - 1
        V = model[index].bellman_function.global_theta
        intercept = cut.intercept
        coefficient = Dict(Symbol(i) => val for (i, val) in cut.coefficients)
        state = Dict(Symbol(i) => val for (i, val) in cut.state)
        n_shift = 1
        while n_shift <= length(cut.shift) && cut.shift[n_shift][2] <= limit[node_index]
            n_shift += 1
        end
        shift = [(s[1], s[2]) for s in cut.shift[1:n_shift-1]]
        cV = @constraint(vf.model, vf.theta - sum(coefficient[i] * x for (i, x) in vf.states) >= intercept - shift[end][1])
        @constraint(vf.model_TV, vf.theta_TV - sum(coefficient[i] * x for (i, x) in vf.states_TV) >= intercept)
        cS = @constraint(model[index].subproblem, V.theta - sum(coefficient[i] * x for (i, x) in V.states) >= intercept - shift[end][1])
        push!(vf.cut_V, AttachedCut(cut.iteration, cut.time, intercept, coefficient, shift, cV, cS, state))
    end

    ref = reference_iteration === nothing ? iteration_max : reference_iteration
    df_approx_value = CSV.read("$(folder)/approx_values.csv", DataFrame)
    model.approx_value =
        [(row.time, row.approx_value) for row in eachrow(df_approx_value) if row.iteration <= ref]

    df_delta = CSV.read("$(folder)/deltas.csv", DataFrame)
    for (_, node) in model.nodes
        node.delta =
            [row.delta for row in eachrow(df_delta) if row.node == node.index && row.iteration <= ref]
    end
    return
end

# Replay the saved run up to a given iteration count.
function _add_cuts_iter(model::PolicyGraph, iteration::Int64, folder::String)
    return _add_cuts(
        model,
        folder,
        cut -> cut.iteration <= iteration;
        reference_iteration = iteration,
    )
end

# Replay the saved run up to a given wall-clock time.
function _add_cuts_time(model::PolicyGraph, time::Int64, folder::String)
    return _add_cuts(model, folder, cut -> cut.time <= time)
end

# Copy the cuts of one in-memory model into another, dropping the shift from the
# copy (`model_copy` starts them at their unshifted position).
function add_cuts_to_model(
    model_copy::PolicyGraph,
    model_to_copy::PolicyGraph,
    iteration::Int64,
)
    T = length(model_copy.nodes)
    for node_index in keys(model_copy.nodes)
        node = model_copy[node_index]
        vf = node.value_function
        index = node.index == 1 ? T : node.index - 1
        V = model_copy[index].bellman_function.global_theta
        for cut in model_to_copy[node_index].value_function.cut_V[2:end]
            if cut.iteration <= iteration
                intercept = cut.intercept
                coefficient = cut.coefficients
                shift = cut.shift[end]
                cV = @constraint(vf.model, vf.theta - sum(coefficient[i] * x for (i, x) in vf.states) >= intercept)
                @constraint(vf.model_TV, vf.theta_TV - sum(coefficient[i] * x for (i, x) in vf.states_TV) >= intercept + shift[1])
                cS = @constraint(model_copy[index].subproblem, V.theta - sum(coefficient[i] * x for (i, x) in V.states) >= intercept)
                push!(vf.cut_V, AttachedCut(cut.iteration, cut.time, intercept, coefficient, [shift], cV, cS, cut.state))
            end
        end
    end
    return
end
