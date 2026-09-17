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

"""
    active_cut_indices(node::Node, tol::Float64)

Positions, inside `node.value_function.cut_V`, of the cuts that still attain the
value function somewhere on the state box (up to `tol`). The others are
dominated everywhere on that box, so dropping them leaves `V` unchanged.

That position is also the identity under which a cut is recorded on disk:
`cut_V[k]` is the `k`-th cut the node received, and `_add_cuts` rebuilds a saved
run in that same order, so `k` picks the same cut out of any replay made under
the same filter.
"""
function active_cut_indices(node::Node, tol::Float64)
    indices = Int[]
    for (k, cut) in enumerate(node.value_function.cut_V)
        if is_active(node, cut.intercept - cut.shift[end][1], cut.coefficients, tol) == 1
            push!(indices, k)
        end
    end
    return indices
end

"""
    all_active_cut_indices(model, tol)

`active_cut_indices` for every node, as a `Dict(node index => Vector{Int})`.
"""
function all_active_cut_indices(model::PolicyGraph{T}, tol::Float64) where {T}
    return Dict(
        index => active_cut_indices(node, tol) for (index, node) in model.nodes
    )
end

count_active_cuts(node::Node, tol::Float64) = length(active_cut_indices(node, tol))

function count_all_active_cuts(model::PolicyGraph{T}, tol::Float64) where {T}
    res = [0.0 for (index, node) in model.nodes]
    for (index, node) in model.nodes
        res[index] = count_active_cuts(node, tol)
    end
    return res
end

# ---------------------------------------------------------------------------
# Recording which cuts are active
# ---------------------------------------------------------------------------

"""
    active_cuts_path(folder)

Where the identity of the active cuts of the run saved under `folder` is kept.

One row per active cut: `limit` is the filter value the model was replayed at
(a wall-clock time for `_add_cuts_time`, an iteration count for
`_add_cuts_iter`), `node` the node the cut belongs to, and `cut_index` its
position among that node's cuts.
"""
active_cuts_path(folder::String) = joinpath(folder, "active_cut_indices.csv")

"""
    save_active_cut_indices(folder, indices_by_limit)

Record the active cuts found at each filter value. `indices_by_limit` maps a
limit to the `Dict(node => Vector{Int})` that `all_active_cut_indices` returns
for the model replayed at that limit.
"""
function save_active_cut_indices(folder::String, indices_by_limit)
    limits, nodes, cut_indices = Int[], Int[], Int[]
    for (limit, by_node) in sort(collect(indices_by_limit); by = first)
        for node_index in sort(collect(keys(by_node)))
            for k in by_node[node_index]
                push!(limits, limit)
                push!(nodes, node_index)
                push!(cut_indices, k)
            end
        end
    end
    CSV.write(
        active_cuts_path(folder),
        DataFrame(limit = limits, node = nodes, cut_index = cut_indices),
    )
    return
end

"""
    load_active_cut_indices(folder, limit)

The active cuts recorded for `limit`, as the `Dict(node => Set{Int})` that
`_add_cuts` takes as its `select`, or `nothing` when `folder` holds no record
for that limit.
"""
function load_active_cut_indices(folder::String, limit)
    path = active_cuts_path(folder)
    if !isfile(path)
        return nothing
    end
    df = CSV.read(path, DataFrame)
    select = Dict{Int,Set{Int}}()
    for row in eachrow(df)
        if row.limit == limit
            push!(get!(select, row.node, Set{Int}()), row.cut_index)
        end
    end
    return isempty(select) ? nothing : select
end

# The `select` of a replay asked to keep only the active cuts. Missing records
# are not fatal: the caller falls back on the full replay, which gives the same
# policy, only with every dominated cut carried along.
function _active_selection(folder::String, limit)
    select = load_active_cut_indices(folder, limit)
    if select === nothing
        @warn "No active cuts recorded for limit $(limit) in $(folder); " *
              "replaying every cut. Run the active-cuts job first."
    end
    return select
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

`select`, when given, maps a node to the positions --- among *its kept cuts* ---
of the only cuts to actually attach, as recorded by `save_active_cut_indices`.
Cuts left out still count towards those positions and towards `limit`, so the
shift each attached cut is replayed at is the one it would have had in the full
replay.

The two `_add_cuts_*` entry points below differ only in the first argument.
"""
function _add_cuts(
    model::PolicyGraph,
    folder::String,
    keep::Function;
    reference_iteration::Union{Int,Nothing} = nothing,
    select::Union{Nothing,Dict{Int,Set{Int}}} = nothing,
)
    if !isfile("$(folder)/cuts.csv")
        # Returning here would leave `model` without a single cut, and the
        # caller would go on to simulate and write results for an empty policy.
        # Under `pmap` the message alone would be lost on a worker's stdout.
        error("Nothing to replay: $(folder)/cuts.csv does not exist.")
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

    # Position of the cut among the kept cuts of its node, i.e. the index it
    # will have in `cut_V` of a full replay --- the identity `select` uses.
    rank = Dict(node_index => 0 for node_index in keys(model.nodes))

    for cut in cuts
        if !keep(cut)
            continue
        end
        node_index = cut.node
        rank[node_index] += 1
        # Cut 0 is the lower bound, which a freshly built model already carries.
        if cut.iteration < 1
            continue
        end
        if select !== nothing &&
           !(rank[node_index] in get(select, node_index, Set{Int}()))
            continue
        end
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

# Replay the saved run up to a given iteration count. With `active_only`, only
# the cuts recorded as active at that same iteration count are attached.
function _add_cuts_iter(
    model::PolicyGraph,
    iteration::Int64,
    folder::String;
    active_only::Bool = false,
)
    return _add_cuts(
        model,
        folder,
        cut -> cut.iteration <= iteration;
        reference_iteration = iteration,
        select = active_only ? _active_selection(folder, iteration) : nothing,
    )
end

# Replay the saved run up to a given wall-clock time. With `active_only`, only
# the cuts recorded as active at that same time are attached.
function _add_cuts_time(
    model::PolicyGraph,
    time::Int64,
    folder::String;
    active_only::Bool = false,
)
    return _add_cuts(
        model,
        folder,
        cut -> cut.time <= time;
        select = active_only ? _active_selection(folder, time) : nothing,
    )
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
