#  Copyright (c) 2017-25, Oscar Dowson and RVSDDP.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v. 2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

# A cut on its way into the model: `_add_cut` builds one, and
# `_update_value_function` turns it into the `Cut2` that is stored.
mutable struct Cut
    iteration::Int64
    time::Float64
    intercept::Float64
    coefficients::Dict{Symbol,Float64}
    state::Dict{Symbol,Float64}
end

# The cost-to-go variable of a subproblem, together with the outgoing state
# variables its cuts are written in.
struct ConvexApproximation
    theta::JuMP.VariableRef
    states::Dict{Symbol,JuMP.VariableRef}
end

_magnitude(x) = abs(x) > 0 ? log10(abs(x)) : 0

function _dynamic_range_warning(intercept, coefficients)
    lo = hi = _magnitude(intercept)
    lo_v = hi_v = intercept
    for v in values(coefficients)
        i = _magnitude(v)
        if v < lo_v
            lo, lo_v = i, v
        elseif v > hi_v
            hi, hi_v = i, v
        end
    end
    if hi - lo > 10
        @warn(
            """Found a cut with a mix of small and large coefficients.
          The order of magnitude difference is $(hi - lo).
          The smallest cofficient is $(lo_v).
          The largest coefficient is $(hi_v).

      You can ignore this warning, but it may be an indication of numerical issues.

      Consider rescaling your model by using different units, e.g, kilometers instead
      of meters. You should also consider reducing the accuracy of your input data (if
      you haven't already). For example, it probably doesn't make sense to measure the
      inflow into a reservoir to 10 decimal places.""",
            maxlog = 1,
        )
    end
    return
end

function _add_cut(
    model::PolicyGraph{T},
    node::Node{T},
    V::ConvexApproximation,
    θᵏ::Float64,
    shift::Tuple{Float64, Int64},
    πᵏ::Dict{Symbol,Float64},
    xᵏ::Dict{Symbol,Float64},
    iteration::Int64,
    time::Float64,
) where {T}
    for (key, x) in xᵏ
        θᵏ -= πᵏ[key] * x
    end
    _dynamic_range_warning(θᵏ, πᵏ)
    cut = Cut(iteration, time, θᵏ, πᵏ, xᵏ)
    _add_cut_constraint_to_model(model, node, V, cut, shift)
    return
end

# Internal: the cut constraints that live in `node`'s own subproblem, in the
# order they were added. `_replay_cuts_into_replicas!` walks this list to bring
# a freshly built replica up to date with the master.
_owned_cuts(node::Node) = get!(() -> Cut2[], node.ext, :owned_cuts)::Vector{Cut2}

# Internal: add the cut `θ - Σᵢ coefficientsᵢ xᵢ ≥ rhs` (`≤` when maximizing) to
# `node`'s subproblem and return its constraint reference. This is the same
# constraint `_add_cut_constraint_to_model` builds on the master, and is what
# keeps each replica's subproblem identical to the master's.
function _add_cut_constraint_to_subproblem(
    node::Node,
    coefficients::Dict{Symbol,Float64},
    rhs::Float64,
)
    V = node.bellman_function.global_theta
    mod = JuMP.owner_model(V.theta)
    expr = @expression(
        mod,
        V.theta - sum(coefficients[i] * x for (i, x) in V.states)
    )
    if JuMP.objective_sense(mod) == MOI.MIN_SENSE
        return @constraint(mod, expr >= rhs)
    end
    return @constraint(mod, expr <= rhs)
end

#Mathis
function _add_cut_constraint_to_model(
    model::PolicyGraph{T},
    node::Node{T},
    V::ConvexApproximation, 
    cut::Cut, 
    shift::Tuple{Float64, Int64}
) where {T}
    mod = JuMP.owner_model(V.theta)
    expr = @expression(
        mod,
        V.theta - sum(cut.coefficients[i] * x for (i, x) in V.states)
    )
    if JuMP.objective_sense(mod) == MOI.MIN_SENSE
        rhs = cut.intercept - shift[1]
        csp = @constraint(mod, expr >= rhs)
        # Mirror the cut onto every replica of this node. Every worker of a
        # `parallel > 1` batch must solve exactly the approximation the master
        # holds; a replica that misses a cut would return duals for a stale
        # value function and so produce an invalid cut.
        replicas = Vector{JuMP.ConstraintRef}(undef, length(node.replicas))
        _parallel_foreach(length(node.replicas)) do r
            replicas[r] = _add_cut_constraint_to_subproblem(
                node.replicas[r],
                cut.coefficients,
                rhs,
            )
        end
        cutV = _update_value_function(
            model[node.children[1].term],
            cut,
            shift,
            csp,
            replicas,
        )
        push!(_owned_cuts(node), cutV)
    else
        @constraint(mod, expr <= cut.intercept)
        for replica in node.replicas
            _add_cut_constraint_to_subproblem(
                replica,
                cut.coefficients,
                cut.intercept,
            )
        end
    end
    #Get cst in node
    return
end

function _update_value_function(
    node::Node{T}, 
    cut::Cut, 
    shift::Tuple{Float64, Int64},
    csp::Union{Nothing, JuMP.ConstraintRef},
    constraint_replicas::Vector{JuMP.ConstraintRef} = JuMP.ConstraintRef[],
) where {T}

    vf=node.value_function

    md_V = vf.model
    cV = @constraint(md_V, vf.theta-sum(cut.coefficients[i]*x for (i,x) in vf.states)>=cut.intercept-shift[1])
    
    cutV = Cut2(
        cut.iteration,
        cut.time,
        cut.intercept,
        cut.coefficients,
        [shift],
        cV,
        csp,
        cut.state,
        constraint_replicas,
    )

    push!(node.value_function.cut_V, cutV)

    md_TV = vf.model_TV
    @constraint(md_TV, vf.theta_TV -sum(cut.coefficients[i]*x for (i,x) in vf.states_TV)>=cut.intercept)

    cutTV = Cut3(
        cut.intercept,
        cut.coefficients,
    )
    push!(node.value_function.cut_TV, cutTV)
    return cutV
end

# Internal struct: this struct is just a cache for arguments until we can build
# an actual instance of the type T at a later point.
struct InstanceFactory{T}
    args::Any
    kwargs::Any
    InstanceFactory{T}(args...; kwargs...) where {T} = new{T}(args, kwargs)
end

"""
    BellmanFunction

A representation of the value function. RVSDDP.jl uses the following unique
representation of the value function that is undocumented in the literature.

It represents the cost-to-go with convex "resource" state variables `x`.

In addition, we have three types of cuts:

 1) Single-cuts (also called "average" cuts in the literature), which involve
    the risk-adjusted expectation of the cost-to-go.
 2) Multi-cuts, which use a different cost-to-go term for each realization w.
 3) Risk-cuts, which correspond to the facets of the dual interpretation of a
    convex risk measure.

Therefore, ValueFunction returns a JuMP model of the following form:

```
V(x, b, y) =
    min: μᵀb + νᵀy + θ
    s.t. # "Single" / "Average" cuts
         μᵀb(j) + νᵀy(j) + θ >= α(j) + xᵀβ(j),          ∀ j ∈ J
         # "Multi" cuts
         μᵀb(k) + νᵀy(k) + φ(w) >= α(k, w) + xᵀβ(k, w), ∀w ∈ Ω, k ∈ K
         # "Risk-set" cuts
         θ ≥ Σ{p(k, w) * φ(w)}_w - μᵀb(k) - νᵀy(k),     ∀ k ∈ K
```
"""
struct BellmanFunction
    global_theta::ConvexApproximation
end

"""
    BellmanFunction(; lower_bound = -Inf, upper_bound = Inf)
"""
function BellmanFunction(; lower_bound = -Inf, upper_bound = Inf)
    return InstanceFactory{BellmanFunction}(;
        lower_bound = lower_bound,
        upper_bound = upper_bound,
    )
end

function bellman_term(bellman_function::BellmanFunction)
    return bellman_function.global_theta.theta
end

function initialize_bellman_function(
    factory::InstanceFactory{BellmanFunction},
    model::PolicyGraph{T},
    node::Node{T},
) where {T}
    lower_bound, upper_bound = -Inf, Inf
    if length(factory.args) > 0
        error(
            "Positional arguments $(factory.args) ignored in BellmanFunction.",
        )
    end
    for (kw, value) in factory.kwargs
        if kw == :lower_bound
            lower_bound = value
        elseif kw == :upper_bound
            upper_bound = value
        else
            error(
                "Keyword $(kw) not recognised as argument to BellmanFunction.",
            )
        end
    end
    if lower_bound == -Inf && upper_bound == Inf
        error("You must specify a finite bound on the cost-to-go term.")
    end
    if length(node.children) == 0
        lower_bound = upper_bound = 0.0
    end
    #Mathis
    Θᴳ = @variable(node.subproblem, base_name = "V_"*string(node.index))
    lower_bound > -Inf && JuMP.set_lower_bound(Θᴳ, lower_bound)
    upper_bound < Inf && JuMP.set_upper_bound(Θᴳ, upper_bound)

    cV= @constraint(node.value_function.model, node.value_function.theta >= lower_bound)
    cTV = @constraint(node.value_function.model_TV, node.value_function.theta_TV >= lower_bound)
    csp= @constraint(node.subproblem, Θᴳ >= lower_bound)

    cutV = Cut2(
        0,
        0.0,
        0.0,
        Dict{Symbol,Float64}(i => 0.0 for (i,x) in node.states),
        [(0.0, 1)],
        cV,
        csp,
        Dict(i => 0.0 for (i,x) in node.states),
    )
    push!(node.value_function.cut_V, cutV)

    cutTV = Cut3(
        0.0,
        Dict{Symbol,Float64}(i => 0.0 for (i,x) in node.states),
    )
    push!(node.value_function.cut_TV, cutTV)



    x′ = Dict(key => var.out for (key, var) in node.states)
    return BellmanFunction(ConvexApproximation(Θᴳ, x′))
end

function refine_bellman_function(
    model::PolicyGraph{T},
    node::Node{T},
    bellman_function::BellmanFunction,
    risk_measure::AbstractRiskMeasure,
    outgoing_state::Dict{Symbol,Float64},
    dual_variables::Vector{Dict{Symbol,Float64}},
    noise_supports::Vector,
    nominal_probability::Vector{Float64},
    objective_realizations::Vector{Float64},
    shift::Tuple{Float64, Int64},
    iteration::Int64,
    time::Float64,
) where {T}
    lock(node.lock)
    try
        return _refine_bellman_function_no_lock(
            model,
            node,
            bellman_function,
            risk_measure,
            outgoing_state,
            dual_variables,
            noise_supports,
            nominal_probability,
            objective_realizations,
            shift,
            iteration,
            time,
        )
    finally
        unlock(node.lock)
    end
end

function _refine_bellman_function_no_lock(
    model::PolicyGraph{T},
    node::Node{T},
    bellman_function::BellmanFunction,
    risk_measure::AbstractRiskMeasure,
    outgoing_state::Dict{Symbol,Float64},
    dual_variables::Vector{Dict{Symbol,Float64}},
    noise_supports::Vector,
    nominal_probability::Vector{Float64},
    objective_realizations::Vector{Float64},
    shift::Tuple{Float64, Int64},
    iteration::Int64,
    time::Float64,
) where {T}
    # Sanity checks.
    @assert length(dual_variables) ==
            length(noise_supports) ==
            length(nominal_probability) ==
            length(objective_realizations)
    # Preliminaries that are common to all cut types.
    risk_adjusted_probability = similar(nominal_probability)
    offset = adjust_probability(
        risk_measure,
        risk_adjusted_probability,
        nominal_probability,
        noise_supports,
        objective_realizations,
        model.objective_sense == MOI.MIN_SENSE,
    )
    return _add_average_cut(
        model,
        node,
        outgoing_state,
        risk_adjusted_probability,
        objective_realizations,
        dual_variables,
        offset,
        shift,
        iteration,
        time,
    )
end

function _add_average_cut(
    model::PolicyGraph{T},
    node::Node{T},
    outgoing_state::Dict{Symbol,Float64},
    risk_adjusted_probability::Vector{Float64},
    objective_realizations::Vector{Float64},
    dual_variables::Vector{Dict{Symbol,Float64}},
    offset::Float64,
    shift::Tuple{Float64, Int64},
    iteration::Int64,
    time::Float64,
) where {T}
    N = length(risk_adjusted_probability)
    @assert N == length(objective_realizations) == length(dual_variables)
    # Calculate the expected intercept and dual variables with respect to the
    # risk-adjusted probability distribution.
    πᵏ = Dict(key => 0.0 for key in keys(outgoing_state))
    θᵏ = offset
    for i in 1:length(objective_realizations)
        p = risk_adjusted_probability[i]
        θᵏ += p * objective_realizations[i]
        for (key, dual) in dual_variables[i]
            πᵏ[key] += p * dual
        end
    end
    _add_cut(
        model,
        node,
        node.bellman_function.global_theta,
        θᵏ,
        shift,
        πᵏ,
        outgoing_state,
        iteration,
        time,
    )
    return (theta = θᵏ - shift[1], pi = πᵏ, x = outgoing_state)
end


# If we are adding a multi-cut for the first time, then the local θ variables
# won't have been added.
# TODO(odow): a way to set different bounds for each variable in the multi-cut.


function compute_approx_value(
    model::PolicyGraph{T},
) where {T}
    horizon = length(model.nodes)
    res = compute_V(model[1].value_function, model.initial_root_state)
    return res
end

function compute_cost_end_of_horizon(
    model::PolicyGraph{T},
    step::Int64,
    incoming_state::Dict{Symbol,Float64},
) where {T}
    horizon = length(model.nodes)
    node_index=(step-1)%horizon + 1
    node = model[node_index]
    res = compute_V(node.value_function, incoming_state)

    for t in 0:(horizon-1)
        node_index_t = (node_index+t-1)%horizon + 1
        node_t = model[node_index_t]
        res+= node_t.delta[end]*node_t.discount_factor^t/(1-model.discount_factor^horizon)
    end
    return res*model.discount_factor^(step-1)
end