#  Copyright (c) 2017-25, Oscar Dowson and RVSDDP.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

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
    return res
end


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
        model.approx_value = [(row.time, row.approx_value) for row in eachrow(df_approx_value) if row.iteration<=iteration]

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

function _add_cuts_time(model::PolicyGraph, time::Int64, folder::String)
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
        model.approx_value = [(row.time, row.approx_value) for row in eachrow(df_approx_value) if row.iteration<=iteration_max]

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
        model.approx_value = [(row.time, row.approx_value) for row in eachrow(df_approx_value) if row.iteration<=iteration_max]

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

function add_cuts_to_model(model_copy::PolicyGraph, model_to_copy::PolicyGraph, iteration::Int64)
    T=length(model_copy.nodes)
    for node_index in keys(model_copy.nodes)
        node=model_copy[node_index]
        vf=node.value_function
        index=node.index == 1 ? T : node.index - 1
        V=model_copy[index].bellman_function.global_theta
        for cut in model_to_copy[node_index].value_function.cut_V[2:end]
            if cut.iteration<=iteration
                intercept = cut.intercept
                coefficient=cut.coefficients
                shift=cut.shift[end]
                cV=@constraint(vf.model, vf.theta -sum(coefficient[i]*x for (i,x) in vf.states)>=intercept)
                @constraint(vf.model_TV, vf.theta_TV -sum(coefficient[i]*x for (i,x) in vf.states_TV)>=intercept + shift[1])
                cS=@constraint(model_copy[index].subproblem, V.theta -sum(coefficient[i]*x for (i,x) in V.states)>=intercept)
                push!(vf.cut_V, Cut2(cut.iteration, cut.time, intercept, coefficient, [shift], cV, cS, cut.state))
            end
        end
    end
end