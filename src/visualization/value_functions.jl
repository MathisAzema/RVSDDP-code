#  Copyright (c) 2017-25, Oscar Dowson and RVSDDP.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
    ValueFunction

A representation of the value function. RVSDDP.jl uses the following unique representation of
the value function that is undocumented in the literature.

It supports three types of state variables:

 1) x - convex "resource" states
 2) b - concave "belief" states
 3) y - concave "objective" states

In addition, we have three types of cuts:

 1) Single-cuts (also called "average" cuts in the literature), which involve the
    risk-adjusted expectation of the cost-to-go.
 2) Multi-cuts, which use a different cost-to-go term for each realization w.
 3) Risk-cuts, which correspond to the facets of the dual interpretation of a coherent risk
    measure.

Therefore, ValueFunction returns a JuMP model of the following form:

    V(x, b, y) = min: μᵀb + νᵀy + θ
                 s.t. # "Single" / "Average" cuts
                      μᵀb(j) + νᵀy(j) + θ >= α(j) + xᵀβ(j), ∀ j ∈ J
                      # "Multi" cuts
                      μᵀb(k) + νᵀy(k) + φ(w) >= α(k, w) + xᵀβ(k, w), ∀w ∈ Ω, k ∈ K
                      # "Risk-set" cuts
                      θ ≥ Σ{p(k, w) * φ(w)}_w - μᵀb(k) - νᵀy(k), ∀ k ∈ K
"""
struct ValueFunction{
    O<:Union{Nothing,NTuple{N,JuMP.VariableRef} where {N}},
    B<:Union{Nothing,Dict{T,JuMP.VariableRef} where {T}},
}
    index::Any
    model::JuMP.Model
    theta::JuMP.VariableRef
    states::Dict{Symbol,JuMP.VariableRef}
    objective_state::O
    belief_state::B
end

function Base.show(io::IO, v::ValueFunction)
    print(io, "A value function for node $(v.index)")
    return
end

function JuMP.set_optimizer(v::ValueFunction, optimizer)
    set_optimizer(v.model, optimizer)
    set_silent(v.model)
    return
end

function _add_to_value_function(
    model::JuMP.Model,
    states::Dict{Symbol,JuMP.VariableRef},
    objective_state,
    belief_state,
    convex_approximation::ConvexApproximation,
    theta_name::String,
)
    theta = @variable(model, base_name = theta_name)
    if objective_sense(model) == MOI.MIN_SENSE
        set_lower_bound(theta, lower_bound(convex_approximation.theta))
    else
        set_upper_bound(theta, upper_bound(convex_approximation.theta))
    end
    for cut in convex_approximation.cuts
        cut_expr = @expression(
            model,
            cut.intercept +
            sum(coef * states[key] for (key, coef) in cut.coefficients)
        )
        if objective_state !== nothing
            @assert cut.obj_y !== nothing
            cut_expr = @expression(
                model,
                cut_expr -
                sum(y * μ for (y, μ) in zip(cut.obj_y, objective_state))
            )
        end
        if belief_state !== nothing
            @assert cut.belief_y !== nothing
            cut_expr = @expression(
                model,
                cut_expr -
                sum(cut.belief_y[key] * μ for (key, μ) in belief_state)
            )
        end
        if objective_sense(model) == MOI.MIN_SENSE
            @constraint(model, theta >= cut_expr)
        else
            @constraint(model, theta <= cut_expr)
        end
    end
    return theta
end

function ValueFunction(model::PolicyGraph{T}; node::T) where {T}
    return ValueFunction(model[node])
end

function ValueFunction(node::Node{T}) where {T}
    b = node.bellman_function
    sense = objective_sense(node.subproblem)
    model = Model()
    if node.optimizer !== nothing
        set_optimizer(model, node.optimizer)
        set_silent(model)
    end
    set_objective_sense(model, sense)
    states = Dict{Symbol,VariableRef}(
        key => @variable(model, base_name = "$(key)") for
        (key, x) in node.states
    )
    objective_state = if node.objective_state === nothing
        nothing
    else
        tuple(
            VariableRef[
                @variable(
                    model,
                    lower_bound = lower_bound(μ),
                    upper_bound = upper_bound(μ),
                    base_name = "_objective_state_$(i)"
                ) for (i, μ) in enumerate(node.objective_state.μ)
            ]...,
        )
    end
    belief_state = if node.belief_state === nothing
        nothing
    else
        Dict{T,VariableRef}(
            key => @variable(
                model,
                lower_bound = lower_bound(μ),
                upper_bound = upper_bound(μ),
                base_name = "_belief_$(key)"
            ) for (key, μ) in node.belief_state.μ
        )
    end
    global_theta = _add_to_value_function(
        model,
        states,
        objective_state,
        belief_state,
        b.global_theta,
        "V",
    )
    local_thetas = VariableRef[
        _add_to_value_function(
            model,
            states,
            belief_state,
            objective_state,
            l,
            "v$(i)",
        ) for (i, l) in enumerate(b.local_thetas)
    ]
    for risk_set in b.risk_set_cuts
        expr = @expression(
            model,
            sum(p * v for (p, v) in zip(risk_set, local_thetas))
        )
        if sense == MOI.MIN_SENSE
            @constraint(model, global_theta >= expr)
        else
            @constraint(model, global_theta <= expr)
        end
    end
    return ValueFunction(
        node.index,
        model,
        global_theta,
        states,
        objective_state,
        belief_state,
    )
end

"""
    evaluate(
        V::ValueFunction,
        point::Dict{Union{Symbol,String},<:Real}
        objective_state = nothing,
        belief_state = nothing
    )

Evaluate the value function `V` at `point` in the state-space.

Returns a tuple containing the height of the function, and the subgradient
w.r.t. the convex state-variables.

## Examples

```julia
evaluate(V, Dict(:volume => 1.0))
```

If the state variable is constructed like
`@variable(sp, volume[1:4] >= 0, RVSDDP.State, initial_value = 0.0)`, use `[i]` to
index the state variable:
```julia
evaluate(V, Dict(Symbol("volume[1]") => 1.0))
```

You can also use strings or symbols for the keys.
```julia
evaluate(V, Dict("volume[1]" => 1))
```
"""
function evaluate(
    V::ValueFunction,
    point::Dict{Symbol,Float64};
    objective_state = nothing,
    belief_state = nothing,
)
    for (state, val) in point
        fix(V.states[state], val; force = true)
    end
    saddle = AffExpr(0.0)
    if V.objective_state !== nothing
        @assert objective_state !== nothing
        for (y, x) in zip(objective_state, V.objective_state)
            add_to_expression!(saddle, y, x)
        end
    end
    if V.belief_state !== nothing
        @assert belief_state !== nothing
        for (key, x) in V.belief_state
            add_to_expression!(saddle, belief_state[key], x)
        end
    end
    @objective(V.model, objective_sense(V.model), V.theta + saddle)
    optimize!(V.model)
    obj = objective_value(V.model)
    duals = Dict{Symbol,Float64}()
    sign = objective_sense(V.model) == MOI.MIN_SENSE ? 1.0 : -1.0
    for (key, var) in V.states
        duals[key] = sign * dual(FixRef(var))
    end
    return obj, duals
end

# Define a fallback method to allow users to write things like `Dict("x" => 1)`.
function evaluate(V::ValueFunction, point; kwargs...)
    return evaluate(
        V,
        Dict(Symbol(k) => convert(Float64, v)::Float64 for (k, v) in point);
        kwargs...,
    )
end

"""
    evalute(V::ValueFunction{Nothing, Nothing}; kwargs...)

Evalute the value function `V` at the point in the state-space specified by
`kwargs`.

## Examples

    evaluate(V; volume = 1)
"""
function evaluate(V::ValueFunction{Nothing,Nothing}; kwargs...)
    return evaluate(V, Dict(k => float(v) for (k, v) in kwargs))
end

struct Point{Y,B}
    x::Dict{Symbol,Float64}
    y::Y
    b::B
end
Point(x::Dict{Symbol,Float64}) = Point(x, nothing, nothing)

function height(V::ValueFunction{Y,B}, x::Point{Y,B}) where {Y,B}
    return evaluate(V, x.x; objective_state = x.y, belief_state = x.b)[1]
end

function get_axis(x::Vector{Dict{K,V}}) where {K,V}
    @assert length(x) >= 2
    changing_key = nothing
    for (key, val) in x[1]
        if val == x[2][key]
            continue
        elseif changing_key !== nothing
            error("Too many elements are changing")
        end
        changing_key = key
    end
    return changing_key === nothing ? nothing : [xi[changing_key] for xi in x]
end

function get_axis(x::Vector{NTuple{N,T}}) where {N,T}
    @assert length(x) >= 2
    changing_index = nothing
    for i in 1:N
        if x[1][i] == x[2][i]
            continue
        elseif changing_index !== nothing
            error("Too many elements are changing")
        end
        changing_index = i
    end
    return changing_index === nothing ? nothing :
           [xi[changing_index] for xi in x]
end

get_axis(::Vector{Nothing}) = nothing

function get_axis(X::Vector{Point{Y,B}}) where {Y,B}
    for f in [x -> x.x, x -> x.y, x -> x.b]
        x = get_axis(f.(X))
        x !== nothing && return x
    end
    return nothing
end

function get_data(V::ValueFunction{Y,B}, X::Vector{Point{Y,B}}) where {Y,B}
    x = get_axis(X)
    if x === nothing
        error("Unable to detect changing dimension")
    end
    y = height.(Ref(V), X)
    return x, y, Float64[]
end

function get_data(V::ValueFunction{Y,B}, X::Matrix{Point{Y,B}}) where {Y,B}
    x = get_axis(collect(X[:, 1]))
    if x === nothing
        error("Unable to detect changing row")
    end
    y = get_axis(collect(X[1, :]))
    if y === nothing
        error("Unable to detect changing column")
    end
    z = height.(Ref(V), X)
    return [i for _ in y for i in x], [i for i in y for _ in x], vec(z)
end

function plot(
    V::ValueFunction{Y,B},
    X::Array{Point{Y,B}};
    filename::String = joinpath(
        tempdir(),
        string(Random.randstring(), ".html"),
    ),
    open::Bool = true,
) where {Y,B}
    x, y, z = get_data(V, X)
    fill_template(
        filename,
        "<!--X-->" => JSON.json(x),
        "<!--Y-->" => JSON.json(y),
        "<!--Z-->" => JSON.json(z);
        template = joinpath(@__DIR__, "value_functions.html"),
        launch = open,
    )
    return
end

function plot(
    V::ValueFunction{Nothing,Nothing};
    filename::String = joinpath(
        tempdir(),
        string(Random.randstring(), ".html"),
    ),
    open::Bool = true,
    kwargs...,
)
    d = Dict{Symbol,Float64}()
    variables = Symbol[]
    for (key, val) in kwargs
        if isa(val, AbstractVector)
            push!(variables, key)
        else
            d[key] = float(val)
        end
    end
    if length(variables) == 1
        points = Point{Nothing,Nothing}[]
        key = variables[1]
        for val in kwargs[key]
            d2 = copy(d)
            d2[key] = val
            push!(points, Point(d2))
        end
        return plot(V, points; filename = filename, open = open)
    elseif length(variables) == 2
        k1, k2 = variables
        N1, N2 = length(kwargs[k1]), length(kwargs[k2])
        points = Array{Point{Nothing,Nothing},2}(undef, N1, N2)
        for i in 1:N1
            for j in 1:N2
                d2 = copy(d)
                d2[k1] = kwargs[k1][i]
                d2[k2] = kwargs[k2][j]
                points[i, j] = Point(d2)
            end
        end
        return plot(V, points; filename = filename, open = open)
    end
    return error(
        "Can only plot 1- or 2-dimensional value functions. You provided " *
        "$(length(variables)).",
    )
end

function compare_two_models(
    model1::PolicyGraph{T},
    model2::PolicyGraph{T};
    replications::Int=1,
    TimeHorizon::Int=1,
    discount_factor::Float64=0.99,
) where {T}
    simulations1 = simulate(
        model1,
        replications,
        [:inflow];
        sampling_scheme=InSampleMonteCarlo(max_depth=TimeHorizon),
    )

    oos1 = [sum((discount_factor^(i-1))*simulations1[k][i][:stage_objective] for i in 1:TimeHorizon) for k in 1:replications]


    Scenario=[[((i-1)%length(model1.nodes)+1, simulations1[k][i][:inflow]) for i in 1:TimeHorizon] for k in 1:replications]

    Noise_scenario=Historical(Scenario)
    simulations2 = simulate(
        model2,
        replications,
        [:inflow];
        sampling_scheme=Noise_scenario,
    )

    oos2 = [sum((discount_factor^(i-1))*simulations2[k][i][:stage_objective] for i in 1:TimeHorizon) for k in 1:replications]
    return Statistics.mean(oos1), Statistics.std(oos1), Statistics.mean(oos2), Statistics.std(oos2)
end


# function compare_models(
#     list_of_models::Vector{PolicyGraph{T}};
#     replications_min::Int=1,
#     replications_max::Int=1,
#     TimeHorizon::Int=1,
#     discount_factor::Float64=0.99,
# ) where {T}
#     oos=[]

#     Scenario=[Serialization.deserialize("scenario_oos/scenario_$k.jls") for k in replications_min:replications_max]

#     Noise_scenario=Historical(Scenario)

#     for i in 1:length(list_of_models)
#         model2=list_of_models[i]
#         simulations2 = simulate(
#             model2,
#             replications,
#             [:inflow];
#             sampling_scheme=Noise_scenario,
#         )
#         oos2 = [sum((discount_factor^(i-1))*simulations2[k][i][:stage_objective] for i in 1:TimeHorizon) for k in 1:replications]
#         push!(oos, oos2)
#     end

#     return oos
# end

# function generate_scenario(
#     list_of_models::Vector{PolicyGraph{T}};
#     replications::Int=1,
#     TimeHorizon::Int=1,
#     discount_factor::Float64=0.99,
# ) where {T}
#     oos=[]
#     model1=list_of_models[1]
#     simulations1 = simulate(
#         model1,
#         replications,
#         [:inflow, :storedEnergy];
#         sampling_scheme=InSampleMonteCarlo(max_depth=TimeHorizon),
#     )
#     oos1 = [sum((discount_factor^(i-1))*simulations1[k][i][:stage_objective] for i in 1:TimeHorizon) for k in 1:replications]
#     push!(oos, oos1)

#     Scenario=[[((i-1)%length(model1.nodes)+1, simulations1[k][i][:inflow]) for i in 1:TimeHorizon] for k in 1:replications]

#     Noise_scenario=Historical(Scenario)

#     for k in 1:replications
#         Serialization.serialize("scenario_oos/scenario_$k.jls", Scenario[k])
#     end
#     return Scenario
# end

function is_active(
    node::Node,
    intercept::Float64, 
    coef::Dict{Symbol,Float64},
    tol::Float64
)
    vf=node.value_function
    model = node.value_function.model
    @objective(model, Max, intercept - vf.theta + sum(a * vf.states[i] for (i,a) in coef))
    JuMP.optimize!(model)
    if JuMP.objective_value(model)>=-tol
        return 1
    else
        return 0
    end
end

# function evolution_cuts(node::Node, model_jensen::PolicyGraph)
#     N = length(model_jensen[node.index].value_function.cut_V)
#     evol_cut = [[0 for k in i:N] for i in 1:N]
#     V=node.bellman_function.global_theta
#     vf=node.value_function
#     for (i,cut) in enumerate(model_jensen[node.index].value_function.cut_V[1:N])
#         intercept = cut.intercept
#         coefficient=cut.coefficients
#         shift=cut.shift[end]
#         cV=@constraint(vf.model, vf.theta -sum(coefficient[i]*x for (i,x) in vf.states)>=intercept)
#         @constraint(vf.model_TV, vf.theta_TV -sum(coefficient[i]*x for (i,x) in vf.states_TV)>=intercept + shift)
#         cS=@constraint(node.subproblem, V.theta -sum(coefficient[i]*x for (i,x) in V.states)>=intercept)
#         push!(vf.cut_V, Cut2(intercept, coefficient, [shift], cV, cS, cut.states))
#         for (k,cutb) in enumerate(model_jensen[node.index].value_function.cut_V[1:i])
#             interceptb = cutb.intercept
#             coefficientb=cutb.coefficients
#             evol_cut[k][i-k+1]=is_active(node, interceptb, coefficientb, 0.01)
#             # println((i, k,is_active(node, interceptb, coefficientb, 0.01)))
#         end
#     end
#     return evol_cut
# end

function count_active_cuts(
    node::Node, 
    tol::Float64
)
    active_cuts = 0
    for (k,cutb) in enumerate(node.value_function.cut_V)
        interceptb = cutb.intercept
        coefficientb=cutb.coefficients 
        active_cuts+=is_active(node, interceptb-cutb.shift[end][1], coefficientb, tol)
    end
    # println("Node $(node.index) has $(active_cuts) active cuts")
    return active_cuts
end

function count_all_active_cuts(
    model::PolicyGraph{T}, 
    tol::Float64
)  where {T}
    res = [0.0 for (index,node) in model.nodes]
    for (index,node) in model.nodes
        res[index] = count_active_cuts(node, tol)
    end
    # println("Total number of active cuts: $(sum(res))")
    return res
end

# function add_cuts(model_copy::PolicyGraph, model_to_copy::PolicyGraph, iteration::Int64)
#     T=length(model_copy.nodes)
#     for node_index in keys(model_copy.nodes)
#         node=model_copy[node_index]
#         vf=node.value_function
#         index=node.index == 1 ? T : node.index - 1
#         V=model_copy[index].bellman_function.global_theta
#         for cut in model_to_copy[node_index].value_function.cut_V[2:end]
#             if cut.iteration<=iteration
#                 intercept = cut.intercept
#                 coefficient=cut.coefficients
#                 shift=cut.shift[end]
#                 cV=@constraint(vf.model, vf.theta -sum(coefficient[i]*x for (i,x) in vf.states)>=intercept)
#                 @constraint(vf.model_TV, vf.theta_TV -sum(coefficient[i]*x for (i,x) in vf.states_TV)>=intercept + shift)
#                 cS=@constraint(model_copy[index].subproblem, V.theta -sum(coefficient[i]*x for (i,x) in V.states)>=intercept)
#                 push!(vf.cut_V, Cut2(cut.iteration, cut.time, intercept, coefficient, [shift], cV, cS, cut.state))
#             end
#         end
#     end
# end

# function add_cuts(model::PolicyGraph, iteration::Int64, filename::String)
#     T=length(model_copy.nodes)
#     for node_index in keys(model_copy.nodes)
#         node=model_copy[node_index]
#         vf=node.value_function
#         index=node.index == 1 ? T : node.index - 1
#         V=model_copy[index].bellman_function.global_theta
#         for cut in model_to_copy[node_index].value_function.cut_V[2:end]
#             if cut.iteration<=iteration
#                 intercept = cut.intercept
#                 coefficient=cut.coefficients
#                 shift=cut.shift[end]
#                 cV=@constraint(vf.model, vf.theta -sum(coefficient[i]*x for (i,x) in vf.states)>=intercept)
#                 @constraint(vf.model_TV, vf.theta_TV -sum(coefficient[i]*x for (i,x) in vf.states_TV)>=intercept + shift)
#                 cS=@constraint(model_copy[index].subproblem, V.theta -sum(coefficient[i]*x for (i,x) in V.states)>=intercept)
#                 push!(vf.cut_V, Cut2(cut.iteration, cut.time, intercept, coefficient, [shift], cV, cS, cut.state))
#             end
#         end
#     end
# end

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
            state = JSON.parse(row.state)
        )
        push!(cuts, cut)
    end
    return cuts
end

using DataFrames

function _add_cuts_iter(model::PolicyGraph, iteration::Int64, folder::String)
    if isfile("$(folder)/cuts.csv")
        df_cuts = CSV.read("$(folder)/cuts.csv", DataFrame)

        cuts = reconstruct_cuts(df_cuts)

        T=length(model.nodes)
        limit = Dict()
        for node_index in keys(model.nodes)
            node=model[node_index]
            vf=node.value_function
            index=node.index == 1 ? T : node.index - 1
            V=model[index].bellman_function.global_theta
            lim= 0
            for cut in cuts[1:end]
                if cut.iteration<=iteration && cut.node == node_index
                    lim+=1
                end
            end
            limit[node_index] = lim
         end

        for cut in cuts
            if 1<=cut.iteration<=iteration
                node_index = cut.node
                node=model[node_index]
                vf=node.value_function
                index=node.index == 1 ? T : node.index - 1
                V=model[index].bellman_function.global_theta
                intercept = cut.intercept
                coefficient=Dict(Symbol(i) => val for (i, val) in cut.coefficients)
                state = Dict(Symbol(i) => val for (i, val) in cut.state)
                shift = 0.0
                i = 1
                while i<= length(cut.shift) && cut.shift[i][2]<=limit[node_index]
                    i += 1
                end
                shift = [(s[1], s[2]) for s in cut.shift[1:i-1]]
                cV=@constraint(vf.model, vf.theta -sum(coefficient[i]*x for (i,x) in vf.states)>=intercept - shift[end][1])
                @constraint(vf.model_TV, vf.theta_TV -sum(coefficient[i]*x for (i,x) in vf.states_TV)>=intercept)
                cS=@constraint(model[index].subproblem, V.theta -sum(coefficient[i]*x for (i,x) in V.states)>=intercept-shift[end][1])
                push!(vf.cut_V, Cut2(cut.iteration, cut.time, intercept, coefficient, shift, cV, cS, state))
            end
        end

        df_approx_value = CSV.read("$(folder)/approx_values.csv", DataFrame)
        model.approx_value = [row.approx_value for row in eachrow(df_approx_value) if row.iteration<=iteration]

        df_delta = CSV.read("$(folder)/deltas.csv", DataFrame)
        for (_, node) in model.nodes
            node.delta = [row.delta for row in eachrow(df_delta) if row.node == node.index && row.iteration <= iteration]
         end
    else
        println("Fichier $folder non trouvé")
        return
    end
    return
end

function _add_cuts(model::PolicyGraph, time::Int64, folder::String)
    if isfile("$(folder)/cuts.csv")
        df_cuts = CSV.read("$(folder)/cuts.csv", DataFrame)

        cuts = reconstruct_cuts(df_cuts)

        T=length(model.nodes)
        limit = Dict()
        iteration_max=0
        for node_index in keys(model.nodes)
            node=model[node_index]
            vf=node.value_function
            index=node.index == 1 ? T : node.index - 1
            V=model[index].bellman_function.global_theta
            lim= 0
            for cut in cuts[1:end]
                if cut.time<=time && cut.node == node_index
                    lim+=1
                    iteration_max = max(iteration_max, cut.iteration)
                end
            end
            limit[node_index] = lim
         end

        for cut in cuts
            if 1<=cut.iteration && cut.time<=time
                node_index = cut.node
                node=model[node_index]
                vf=node.value_function
                index=node.index == 1 ? T : node.index - 1
                V=model[index].bellman_function.global_theta
                intercept = cut.intercept
                coefficient=Dict(Symbol(i) => val for (i, val) in cut.coefficients)
                state = Dict(Symbol(i) => val for (i, val) in cut.state)
                shift = 0.0
                i = 1
                while i<= length(cut.shift) && cut.shift[i][2]<=limit[node_index]
                    i += 1
                end
                shift = [(s[1], s[2]) for s in cut.shift[1:i-1]]
                cV=@constraint(vf.model, vf.theta -sum(coefficient[i]*x for (i,x) in vf.states)>=intercept - shift[end][1])
                @constraint(vf.model_TV, vf.theta_TV -sum(coefficient[i]*x for (i,x) in vf.states_TV)>=intercept)
                cS=@constraint(model[index].subproblem, V.theta -sum(coefficient[i]*x for (i,x) in V.states)>=intercept-shift[end][1])
                push!(vf.cut_V, Cut2(cut.iteration, cut.time, intercept, coefficient, shift, cV, cS, state))
            end
        end

        df_approx_value = CSV.read("$(folder)/approx_values.csv", DataFrame)
        model.approx_value = [row.approx_value for row in eachrow(df_approx_value) if row.iteration<=iteration_max]

        df_delta = CSV.read("$(folder)/deltas.csv", DataFrame)
        for (_, node) in model.nodes
            node.delta = [row.delta for row in eachrow(df_delta) if row.node == node.index && row.iteration <= iteration_max]
         end
    else
        println("Fichier $folder non trouvé")
        return
    end
    return
end

function _add_cuts_finite(model::PolicyGraph, time::Int64, folder::String)
    if isfile("$(folder)/cuts.csv")
        df_cuts = CSV.read("$(folder)/cuts.csv", DataFrame)

        cuts = reconstruct_cuts(df_cuts)

        T=length(model.nodes)
        limit = Dict()
        iteration_max=0
        for node_index in keys(model.nodes)
            node=model[node_index]
            vf=node.value_function
            index=node.index == 1 ? T : node.index - 1
            V=model[index].bellman_function.global_theta
            lim= 0
            for cut in cuts[1:end]
                if cut.time<=time && cut.node == node_index
                    lim+=1
                    iteration_max = max(iteration_max, cut.iteration)
                end
            end
            limit[node_index] = lim
         end

        for cut in cuts
            if 1<=cut.iteration && cut.time<=time && cut.node >=2
                node_index = cut.node
                node=model[node_index]
                vf=node.value_function
                index=node.index == 1 ? T : node.index - 1
                V=model[index].bellman_function.global_theta
                intercept = cut.intercept
                coefficient=Dict(Symbol(i) => val for (i, val) in cut.coefficients)
                state = Dict(Symbol(i) => val for (i, val) in cut.state)
                shift = 0.0
                i = 1
                while i<= length(cut.shift) && cut.shift[i][2]<=limit[node_index]
                    i += 1
                end
                shift = [(s[1], s[2]) for s in cut.shift[1:i-1]]
                cV=@constraint(vf.model, vf.theta -sum(coefficient[i]*x for (i,x) in vf.states)>=intercept - shift[end][1])
                @constraint(vf.model_TV, vf.theta_TV -sum(coefficient[i]*x for (i,x) in vf.states_TV)>=intercept)
                cS=@constraint(model[index].subproblem, V.theta -sum(coefficient[i]*x for (i,x) in V.states)>=intercept-shift[end][1])
                push!(vf.cut_V, Cut2(cut.iteration, cut.time, intercept, coefficient, shift, cV, cS, state))
            end
        end

        df_approx_value = CSV.read("$(folder)/approx_values.csv", DataFrame)
        model.approx_value = [row.approx_value for row in eachrow(df_approx_value) if row.iteration<=iteration_max]

        df_delta = CSV.read("$(folder)/deltas.csv", DataFrame)
        for (_, node) in model.nodes
            node.delta = [row.delta for row in eachrow(df_delta) if row.node == node.index && row.iteration <= iteration_max]
         end
    else
        println("Fichier $folder non trouvé")
        return
    end
    return
end