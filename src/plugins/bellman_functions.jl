#  Copyright (c) 2017-25, Oscar Dowson and RVSDDP.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public License,
#  v. 2.0. If a copy of the MPL was not distributed with this file, You can
#  obtain one at http://mozilla.org/MPL/2.0/.

mutable struct Cut
    iteration::Int64
    time::Float64
    intercept::Float64
    coefficients::Dict{Symbol,Float64}
    obj_y::Union{Nothing,NTuple{N,Float64} where {N}}
    belief_y::Union{Nothing,Dict{T,Float64} where {T}}
    non_dominated_count::Int
    constraint_ref::Union{Nothing,JuMP.ConstraintRef}
    state::Dict{Symbol,Float64}
end

mutable struct SampledState
    state::Dict{Symbol,Float64}
    obj_y::Union{Nothing,NTuple{N,Float64} where {N}}
    belief_y::Union{Nothing,Dict{T,Float64} where {T}}
    dominating_cut::Cut
    best_objective::Float64
end

mutable struct ConvexApproximation
    theta::JuMP.VariableRef
    states::Dict{Symbol,JuMP.VariableRef}
    objective_states::Union{Nothing,NTuple{N,JuMP.VariableRef} where {N}}
    belief_states::Union{Nothing,Dict{T,JuMP.VariableRef} where {T}}
    # Storage for cut selection
    cuts::Vector{Cut}
    sampled_states::Vector{SampledState}
    cuts_to_be_deleted::Vector{Cut}
    deletion_minimum::Int

    function ConvexApproximation(
        theta::JuMP.VariableRef,
        states::Dict{Symbol,JuMP.VariableRef},
        objective_states,
        belief_states,
        deletion_minimum::Int,
    )
        return new(
            theta,
            states,
            objective_states,
            belief_states,
            Cut[],
            SampledState[],
            Cut[],
            deletion_minimum,
        )
    end
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
    obj_y::Union{Nothing,NTuple{N,Float64}},
    belief_y::Union{Nothing,Dict{T,Float64}},
    iteration::Int64,
    time::Float64,
) where {N,T}
    for (key, x) in xᵏ
        θᵏ -= πᵏ[key] * x
    end
    _dynamic_range_warning(θᵏ, πᵏ)
    cut = Cut(iteration, time, θᵏ, πᵏ, obj_y, belief_y, 1, nothing, xᵏ)
    _add_cut_constraint_to_model(model, node, V, cut, shift)
    return
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
    yᵀμ = JuMP.AffExpr(0.0)
    if V.objective_states !== nothing
        for (y, μ) in zip(cut.obj_y, V.objective_states)
            JuMP.add_to_expression!(yᵀμ, y, μ)
        end
    end
    if V.belief_states !== nothing
        for (k, μ) in V.belief_states
            JuMP.add_to_expression!(yᵀμ, cut.belief_y[k], μ)
        end
    end
    expr = @expression(
        mod,
        V.theta + yᵀμ - sum(cut.coefficients[i] * x for (i, x) in V.states)
    )
    cut.constraint_ref = if JuMP.objective_sense(mod) == MOI.MIN_SENSE
        csp = @constraint(mod, expr >= cut.intercept-shift[1])
        _update_value_function(model[node.children[1].term], cut, shift, csp)
    else
        @constraint(mod, expr <= cut.intercept)
    end
    #Get cst in node
    return
end

function _update_value_function(
    node::Node{T}, 
    cut::Cut, 
    shift::Tuple{Float64, Int64},
    csp::Union{Nothing, JuMP.ConstraintRef}
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
    )

    push!(node.value_function.cut_V, cutV)

    md_TV = vf.model_TV
    @constraint(md_TV, vf.theta_TV -sum(cut.coefficients[i]*x for (i,x) in vf.states_TV)>=cut.intercept)

    cutTV = Cut3(
        cut.intercept,
        cut.coefficients,
    )
    push!(node.value_function.cut_TV, cutTV)
    return
end

@enum(CutType, SINGLE_CUT, MULTI_CUT)

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

It supports three types of state variables:

 1) x - convex "resource" states
 2) b - concave "belief" states
 3) y - concave "objective" states

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
mutable struct BellmanFunction
    cut_type::CutType
    global_theta::ConvexApproximation
    local_thetas::Vector{ConvexApproximation}
    # Cuts defining the dual representation of the risk measure.
    risk_set_cuts::Set{Vector{Float64}}
end

"""
    BellmanFunction(;
        lower_bound = -Inf,
        upper_bound = Inf,
        deletion_minimum::Int = 1,
        cut_type::CutType = MULTI_CUT,
    )
"""
function BellmanFunction(;
    lower_bound = -Inf,
    upper_bound = Inf,
    deletion_minimum::Int = 1,
    cut_type::CutType = MULTI_CUT,
)
    return InstanceFactory{BellmanFunction}(;
        lower_bound = lower_bound,
        upper_bound = upper_bound,
        deletion_minimum = deletion_minimum,
        cut_type = cut_type,
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
    lower_bound, upper_bound, deletion_minimum, cut_type =
        -Inf, Inf, 0, SINGLE_CUT
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
        elseif kw == :deletion_minimum
            deletion_minimum = value
        elseif kw == :cut_type
            cut_type = value
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



    # Objective-state and belief-state interpolation are unused; both fields
    # are always `nothing`, so there are no initial bounds to add.
    x′ = Dict(key => var.out for (key, var) in node.states)
    return BellmanFunction(
        cut_type,
        ConvexApproximation(Θᴳ, x′, nothing, nothing, deletion_minimum),
        ConvexApproximation[],
        Set{Vector{Float64}}(),
    )
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
    # The meat of the function.
    if bellman_function.cut_type == SINGLE_CUT
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
    else  # Add a multi-cut
        @assert bellman_function.cut_type == MULTI_CUT
        _add_locals_if_necessary(node, bellman_function, length(dual_variables))
        return _add_multi_cut(
            node,
            outgoing_state,
            risk_adjusted_probability,
            objective_realizations,
            dual_variables,
            offset,
        )
    end
end

#Mathis attention il faudrait un shift pour chaque enfant
function no_shift(
    model::PolicyGraph{T},
    node::Node{T},
    items_traj::Vector{BackwardPassItems{T, Noise}},
    outgoing_states::Vector{Dict{Symbol, Float64}},
) where {T}
    return (0.0, length(node.value_function.cut_V)+1)
end

function update_shift(
    model::PolicyGraph{T},
    node::Node{T},
    shift_k::Float64,
) where {T}
    iter = length(node.value_function.cut_V)+1
    for cut in node.value_function.cut_V
        if shift_k < cut.shift[end][1]
            push!(cut.shift, (shift_k, iter))
            set_normalized_rhs(cut.constraint_V, cut.intercept-shift_k)
            if cut.constraint_subproblem !== nothing
                set_normalized_rhs(cut.constraint_subproblem, cut.intercept-shift_k)
            end
        end
    end
end

function shift_update_random_forward(
    model::PolicyGraph{T},
    node::Node{T},
    items_traj::Vector{BackwardPassItems{T, Noise}},
    outgoing_states::Vector{Dict{Symbol, Float64}},
) where {T}
    res_traj=[0.0 for i in 1:length(items_traj)]
    shift=0.0
    for (i, items) in enumerate(items_traj)
        state = outgoing_states[i]
        θᵏ=0.0
        for j in 1:length(items.objectives)
            p = items.probability[j]
            θᵏ += p * items.objectives[j]
        end
        TVx=θᵏ
        Vx=compute_V(node.value_function, state)
        res_traj[i]=(TVx-Vx)
    end
    shift, k=findmin(res_traj)
    sol=Dict{Symbol,Float64}()
    two_stage=node.two_stage
    for (i,x) in outgoing_states[1]
        lb=two_stage.lower_bounds[i]
        ub=two_stage.upper_bounds[i]
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
    # Now add the average-cut to the subproblem. We include the objective-state
    # component μᵀy and the belief state (if it exists).
    obj_y =
        node.objective_state === nothing ? nothing : node.objective_state.state
    belief_y =
        node.belief_state === nothing ? nothing : node.belief_state.belief
    _add_cut(
        model,
        node, 
        node.bellman_function.global_theta,
        θᵏ,
        shift,
        πᵏ,
        outgoing_state,
        obj_y,
        belief_y,
        iteration,
        time,
    )
    return (
        theta = θᵏ-shift[1],
        pi = πᵏ,
        x = outgoing_state,
        obj_y = obj_y,
        belief_y = belief_y,
    )
end

function _add_multi_cut(
    node::Node,
    outgoing_state::Dict{Symbol,Float64},
    risk_adjusted_probability::Vector{Float64},
    objective_realizations::Vector{Float64},
    dual_variables::Vector{Dict{Symbol,Float64}},
    offset::Float64,
)
    N = length(risk_adjusted_probability)
    @assert N == length(objective_realizations) == length(dual_variables)
    bellman_function = node.bellman_function
    μᵀy = get_objective_state_component(node)
    JuMP.add_to_expression!(μᵀy, get_belief_state_component(node))
    for i in 1:length(dual_variables)
        _add_cut(
            bellman_function.local_thetas[i],
            objective_realizations[i],
            dual_variables[i],
            outgoing_state,
            node.objective_state === nothing ? nothing :
            node.objective_state.state,
            node.belief_state === nothing ? nothing : node.belief_state.belief,
        )
    end
    model = JuMP.owner_model(bellman_function.global_theta.theta)
    cut_expr = @expression(
        model,
        sum(
            risk_adjusted_probability[i] *
            bellman_function.local_thetas[i].theta for i in 1:N
        ) - (1 - sum(risk_adjusted_probability)) * μᵀy + offset
    )
    # TODO(odow): should we use `cut_expr` instead?
    ξ = copy(risk_adjusted_probability)
    if !(ξ in bellman_function.risk_set_cuts) || μᵀy != JuMP.AffExpr(0.0)
        push!(bellman_function.risk_set_cuts, ξ)
        if JuMP.objective_sense(model) == MOI.MIN_SENSE
            @constraint(model, bellman_function.global_theta.theta >= cut_expr)
        else
            @constraint(model, bellman_function.global_theta.theta <= cut_expr)
        end
    end
    return
end

# If we are adding a multi-cut for the first time, then the local θ variables
# won't have been added.
# TODO(odow): a way to set different bounds for each variable in the multi-cut.
function _add_locals_if_necessary(
    node::Node,
    bellman_function::BellmanFunction,
    N::Int,
)
    num_local_thetas = length(bellman_function.local_thetas)
    if num_local_thetas == N
        return # Do nothing. Already initialized.
    elseif num_local_thetas > 0
        error(
            "Expected $(N) local θ variables but there were " *
            "$(num_local_thetas).",
        )
    end
    global_theta = bellman_function.global_theta
    model = JuMP.owner_model(global_theta.theta)
    local_thetas = @variable(model, [1:N])
    if JuMP.has_lower_bound(global_theta.theta)
        JuMP.set_lower_bound.(
            local_thetas,
            JuMP.lower_bound(global_theta.theta),
        )
    end
    if JuMP.has_upper_bound(global_theta.theta)
        JuMP.set_upper_bound.(
            local_thetas,
            JuMP.upper_bound(global_theta.theta),
        )
    end
    for local_theta in local_thetas
        push!(
            bellman_function.local_thetas,
            ConvexApproximation(
                local_theta,
                global_theta.states,
                node.objective_state === nothing ? nothing :
                node.objective_state.μ,
                node.belief_state === nothing ? nothing : node.belief_state.μ,
                global_theta.deletion_minimum,
            ),
        )
    end
    return
end

"""
    write_cuts_to_file(
        model::PolicyGraph{T},
        filename::String;
        kwargs...,
    ) where {T}

Write the cuts that form the policy in `model` to `filename` in JSON format.

## Keyword arguments

 - `node_name_parser` is a function which converts the name of each node into a
    string representation. It has the signature: `node_name_parser(::T)::String`.

 - `write_only_selected_cuts` write only the selected cuts to the json file.
    Defaults to false.
"""
function write_cuts_to_file(
    model::PolicyGraph{T},
    filename::String;
    node_name_parser::Function = string,
    write_only_selected_cuts::Bool = false,
) where {T}
    cuts = Dict{String,Any}[]
    for (node_name, node) in model.nodes
        if node.objective_state !== nothing || node.belief_state !== nothing
            error(
                "Unable to write cuts to file because model contains " *
                "objective states or belief states.",
            )
        end
        node_cuts = Dict(
            "node" => node_name_parser(node_name),
            "single_cuts" => Dict{String,Any}[],
            "multi_cuts" => Dict{String,Any}[],
            "risk_set_cuts" => Vector{Float64}[],
        )
        oracle = node.bellman_function.global_theta
        for (cut, state) in zip(oracle.cuts, oracle.sampled_states)
            if write_only_selected_cuts && cut.constraint_ref === nothing
                continue
            end
            intercept = cut.intercept
            for (key, π) in cut.coefficients
                intercept += π * state.state[key]
            end
            push!(
                node_cuts["single_cuts"],
                Dict(
                    "intercept" => intercept,
                    "coefficients" => copy(cut.coefficients),
                    "state" => copy(state.state),
                ),
            )
        end
        for (i, theta) in enumerate(node.bellman_function.local_thetas)
            for (cut, state) in zip(theta.cuts, theta.sampled_states)
                if write_only_selected_cuts && cut.constraint_ref === nothing
                    continue
                end
                intercept = cut.intercept
                for (key, π) in cut.coefficients
                    intercept += π * state.state[key]
                end
                push!(
                    node_cuts["multi_cuts"],
                    Dict(
                        "realization" => i,
                        "intercept" => intercept,
                        "coefficients" => copy(cut.coefficients),
                        "state" => copy(state.state),
                    ),
                )
            end
        end
        for p in node.bellman_function.risk_set_cuts
            push!(node_cuts["risk_set_cuts"], p)
        end
        push!(cuts, node_cuts)
    end
    open(filename, "w") do io
        return write(io, JSON.json(cuts))
    end
    return
end

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