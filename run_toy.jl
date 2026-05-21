import Pkg
# Pkg.instantiate()
Pkg.activate(".")

using Distributed

Nbworkers = 15
if nworkers() >= Nbworkers+1
    rmprocs(workers())
    addprocs(Nbworkers)
else
    addprocs(Nbworkers - nworkers())
end

@everywhere import Pkg
@everywhere Pkg.activate(".")
@everywhere using Random
@everywhere using RVSDDP
@everywhere using Gurobi
@everywhere const GRB_ENV = Gurobi.Env()
@everywhere optimizer=() -> Gurobi.Optimizer(GRB_ENV)

@everywhere function subproblem_builder(subproblem::Model, node::Int, discount_factor::Float64)
    # State variables
    N=1
    @variable(subproblem, 0 <= volume[1:N] <= 200, RVSDDP.State, initial_value = 50)
    # Control variables
    @variables(subproblem, begin
        thermal_generation[1:4] >= 0
        thermal_generation_tot >= 0
        hydro_generation[1:N] >= 0
        hydro_spill[1:N] >= 0
        deficit >= 0
    end)
    # Random variables
    @variable(subproblem, inflow)
    Ω = [20.0, 80.0]
    # Ω = [40.0]
    P = [1 / length(Ω) for _ in Ω]
    # Ω = [70.0]
    # P = [1.0]
    RVSDDP.parameterize(subproblem, Ω, P) do ω
        return JuMP.fix(inflow, ω)
    end

    capa_th = [10 15 10 5] 
    c_th = [1 2 5 10]
    # Transition function and constraints
    @constraints(
        subproblem,
        begin
            [i in 1:N], volume[i].out == volume[i].in - hydro_generation[i] - hydro_spill[i] + inflow
            [i in 1:N], hydro_generation[i] <= 100
            [i in 1:1], thermal_generation[i] <= capa_th[i]
            thermal_generation_tot==sum(thermal_generation[i] for i in 1:4)
            deficit + sum(hydro_generation[i] for i in 1:N) + thermal_generation_tot == 60
        end
    )
    # Stage-objective
    @stageobjective(subproblem, 1*(50*sum(hydro_spill[i] for i in 1:N) + 50 * deficit+ sum(thermal_generation[i]*c_th[i] for i in 1:4)))
    return subproblem
end

@everywhere graph=RVSDDP.InfiniteLinearGraph(1);

@everywhere using CSV, DataFrames, JSON

@everywhere function rvsddp_job(seed, parallel, cut_max, shift_function, discount_factor, refine_mode)
    model = RVSDDP.PolicyGraph(
        subproblem_builder,
        graph;
        sense = :Min,
        lower_bound = 0.0,
        optimizer = optimizer,
        discount_factor=discount_factor,
    )

    Random.seed!(seed)
    Cuts=RVSDDP.train(model; refine_mode=refine_mode, parallel=parallel, sampling_scheme=RVSDDP.InSampleMonteCarlo(max_depth=10000000, rollout_limit = i -> i, parallel=parallel), cut_limit = cut_max, infinite = true, shift_function=shift_function); 

    cuts_data = []
    for (_, node) in model.nodes
        for cut in node.value_function.cut_V
            push!(cuts_data, Dict(
                :node => node.index,
                :iteration => cut.iteration,
                :time => cut.time,
                :intercept => cut.intercept,
                :coefficients => JSON.json(cut.coefficients),
                :shift => JSON.json(cut.shift),
                :state => JSON.json(cut.state)
            ))
        end
    end

    # Créer une DataFrame
    df_cuts = DataFrame(cuts_data)

    folder1 = "results_toy/$(shift_function)_$(refine_mode)_parallel_$(parallel)"
    if !isdir(folder1)
        mkdir(folder1)
    end

    folder2 = "$(folder1)/$(discount_factor)"
    if !isdir(folder2)
        mkdir(folder2)
    end

    folder3 = "$(folder1)/$(discount_factor)/seed_$(seed)_cut_$(cut_max)"
    if !isdir(folder3)
        mkdir(folder3)
    end

    # Sauvegarder en CSV
    CSV.write("$(folder3)/cuts.csv", df_cuts)

    delta_data = []
    for (_, node) in model.nodes
        for (iter,delta) in enumerate(node.delta)
            push!(delta_data, Dict(
                :node => node.index,
                :iteration => iter,
                :delta => delta,
            ))
        end
    end

    CSV.write("$(folder3)/deltas.csv", DataFrame(delta_data))

    approx_value_data = []
    for (iter,val) in enumerate(model.approx_value)
        push!(approx_value_data, Dict(
            :iteration => iter,
            :approx_value => val,
        ))
    end

    CSV.write("$(folder3)/approx_values.csv", DataFrame(approx_value_data))
end

function run_rvsddp(seed_list, parallel, cut_max_list, shift_function_list, discount_factor_list, refine_mode_list)
    for shift_function in shift_function_list
        for refine_mode in refine_mode_list
            folder1 = "results_toy/$(shift_function)_$(refine_mode)_parallel_$(parallel)"
            if !isdir(folder1)
                mkdir(folder1)
            end
            for discount_factor in discount_factor_list
                folder2 = "$(folder1)/$(discount_factor)"
                if !isdir(folder2)
                    mkdir(folder2)
                end
            end
        end
    end

    combos = [(seed, parallel, cut_max, shift_function, discount_factor, refine_mode) for seed in seed_list for cut_max in cut_max_list for shift_function in shift_function_list for discount_factor in discount_factor_list for refine_mode in refine_mode_list]

    results = pmap(combos) do (seed, parallel, cut_max, shift_function, discount_factor, refine_mode)
        rvsddp_job(seed, parallel, cut_max, shift_function, discount_factor, refine_mode)
    end
    return 
end

@everywhere function evaluate_job(folder, iter, N, discount_factor)

    TimeHorizon = Int(round(log(0.001)/log(discount_factor)))

    model = RVSDDP.PolicyGraph(
        subproblem_builder,
        graph;
        sense = :Min,
        lower_bound = 0.0,
        optimizer = optimizer,
        discount_factor=discount_factor,
    )

    RVSDDP.add_cuts(model, iter, folder);

    Random.seed!(12345)

    simulations= RVSDDP.simulate(
            model,
            N;
            sampling_scheme = RVSDDP.InSampleMonteCarlo(max_depth=TimeHorizon),
        )
    oos_horizon = [sum((discount_factor^(t-1))*simulations[k][t][:stage_objective] for t in 1:TimeHorizon) for k in 1:N]
    oos_end_of_horizon = [simulations[k][TimeHorizon][:cost_end_of_horizon] for k in 1:N]

    folder_res = "$(folder)/oos"
    if !isdir(folder_res)
        mkdir(folder_res)
    end

    CSV.write("$(folder_res)/oos_horizon_$(iter)_$(TimeHorizon).csv", DataFrame(iteration=1:N, oos_horizon=oos_horizon))
    CSV.write("$(folder_res)/oos_end_of_horizon_$(iter)_$(TimeHorizon).csv", DataFrame(iteration=1:N, oos_end_of_horizon=oos_end_of_horizon))

end

function run_evaluate(seed_list, cut_max_list, shift_function_list, discount_factor_list, iter_list, refine_mode_list, N_list)
    combos = [("results_toy/$(shift_function)_$(refine_mode)_parallel_$(parallel)/$(discount_factor)/seed_$(seed)_cut_$(cut_max)", iter, N, discount_factor) for seed in seed_list for cut_max in cut_max_list for shift_function in shift_function_list for discount_factor in discount_factor_list for refine_mode in refine_mode_list for iter in iter_list for N in N_list]

    results = pmap(combos) do (folder, iter, N, discount_factor)
        evaluate_job(folder, iter, N, discount_factor)
    end
    return 
end