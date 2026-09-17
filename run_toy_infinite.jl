import Pkg
# Pkg.instantiate()
Pkg.activate(".")

using Distributed

# Cores given to each training run. `RVSDDP.train(; parallel = k)` now really
# spreads the k trajectories of a batch over k cores, but a Distributed worker
# is started with one thread unless it is told otherwise, so set this to the
# `parallel` value used below to get that speed-up.
#
# Keep Nbworkers * ThreadsPerWorker <= the number of physical cores: these
# experiments are wall-clock limited, so oversubscribing the machine would
# distort the reported times. Leaving it at 1 reproduces the previous
# behaviour exactly (the batch is then simulated one trajectory at a time).
ThreadsPerWorker = 1
Nbworkers = 15
if nworkers() >= Nbworkers+1
    rmprocs(workers())
    addprocs(Nbworkers; exeflags = "-t $(ThreadsPerWorker)")
else
    addprocs(Nbworkers - nworkers(); exeflags = "-t $(ThreadsPerWorker)")
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
    P = [1 / length(Ω) for _ in Ω]
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

@everywhere function rvsddp_job(seed, parallel, iter_max, shift_function, discount_factor, refine_scheme)
    model = RVSDDP.PolicyGraph(
        subproblem_builder,
        graph;
        sense = :Min,
        lower_bound = 0.0,
        optimizer = optimizer,
        discount_factor=discount_factor,
    )

    Random.seed!(seed)
    Cuts=RVSDDP.train(model; refine_scheme=refine_scheme, parallel=parallel, sampling_scheme=RVSDDP.InSampleMonteCarlo(rollout_limit = i -> i), iteration_limit = iter_max, infinite = true, shift_function=shift_function); 

    folder3 = "results_toy/$(RVSDDP.method_label(shift_function, refine_scheme))_parallel_$(parallel)/$(discount_factor)/seed_$(seed)_iter_$(iter_max)"
    mkpath(folder3)

    # Voir run_msppy.jl : cuts.arrow + shift_events.arrow.
    RVSDDP.save_cuts(model, folder3)

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
            # `val` is the (elapsed time, value) pair `backward_pass` records.
            # Split it here: written as a tuple it would land in a single
            # stringified column, and `_add_cuts` reads the two back separately.
            :time => val[1],
            :approx_value => val[2],
        ))
    end

    CSV.write("$(folder3)/approx_values.csv", DataFrame(approx_value_data))
end

function run_toy_infinite(seed_list, parallel, iter_max_list, shift_function_list, discount_factor_list, refine_scheme_list)
    for shift_function in shift_function_list
        for refine_scheme in refine_scheme_list
            folder1 = "results_toy/$(RVSDDP.method_label(shift_function, refine_scheme))_parallel_$(parallel)"
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

    combos = [(seed, parallel, iter_max, shift_function, discount_factor, refine_scheme) for seed in seed_list for iter_max in iter_max_list for shift_function in shift_function_list for discount_factor in discount_factor_list for refine_scheme in refine_scheme_list]

    results = pmap(combos) do (seed, parallel, iter_max, shift_function, discount_factor, refine_scheme)
        rvsddp_job(seed, parallel, iter_max, shift_function, discount_factor, refine_scheme)
    end
    return 
end

# `active_only` replays only the cuts `run_active_toy` recorded as active at
# `iter_limit`, which leaves the policy unchanged --- the cuts left out are
# dominated everywhere on the state box --- while making every subproblem of the
# simulation that much smaller. It falls back on the full replay, with a
# warning, when `run_active_toy` has not been run on `folder` for that
# iteration count.
@everywhere function evaluate_job(folder, iter_limit, N, discount_factor; active_only = true)

    TimeHorizon = Int(ceil(log(0.001)/(log(discount_factor))))

    model = RVSDDP.PolicyGraph(
        subproblem_builder,
        graph;
        sense = :Min,
        lower_bound = 0.0,
        optimizer = optimizer,
        discount_factor=discount_factor,
    )

    RVSDDP._add_cuts_iter(model, iter_limit, folder; active_only = active_only);

    Random.seed!(12345)

    simulations= RVSDDP.simulate(
            model,
            N;
            sampling_scheme = RVSDDP.InSampleMonteCarlo(max_depth=TimeHorizon),
        )

    oos_horizon = [sum((discount_factor^(t-1))*simulations[k][t][:stage_objective] for t in 1:TimeHorizon) for k in 1:N]
    oos_5 = [sum((discount_factor^(t-1))*simulations[k][t][:stage_objective] for t in 1:min(5*12,TimeHorizon)) for k in 1:N]
    oos_10 = [sum((discount_factor^(t-1))*simulations[k][t][:stage_objective] for t in 1:min(10*12,TimeHorizon)) for k in 1:N]
    oos_end_of_horizon = [simulations[k][TimeHorizon][:cost_end_of_horizon] for k in 1:N]

    folder_res = "$(folder)/oos"
    if !isdir(folder_res)
        mkdir(folder_res)
    end

    CSV.write("$(folder_res)/oos_horizon_$(iter_limit)_$(TimeHorizon)_$N.csv", DataFrame(iteration=1:N, oos_horizon=oos_horizon))
    CSV.write("$(folder_res)/oos_end_of_horizon_$(iter_limit)_$(TimeHorizon)_$N.csv", DataFrame(iteration=1:N, oos_end_of_horizon=oos_end_of_horizon))
    CSV.write("$(folder_res)/oos_5_$(iter_limit)_$(TimeHorizon)_$N.csv", DataFrame(iteration=1:N, oos_horizon=oos_5))
    CSV.write("$(folder_res)/oos_10_$(iter_limit)_$(TimeHorizon)_$N.csv", DataFrame(iteration=1:N, oos_horizon=oos_10))

end

function run_evaluate(seed_list, parallel, iter_max_list, shift_function_list, discount_factor_list, iter_list, refine_scheme_list, N_list; active_only = true)
    combos = [("results_toy/$(RVSDDP.method_label(shift_function, refine_scheme))_parallel_$(parallel)/$(discount_factor)/seed_$(seed)_iter_$(iter_max)", iter_limit, N, discount_factor) for seed in seed_list for iter_max in iter_max_list for shift_function in shift_function_list for discount_factor in discount_factor_list for refine_scheme in refine_scheme_list for iter_limit in iter_list for N in N_list]

    results = pmap(combos) do (folder, iter_limit, N, discount_factor)
        evaluate_job(folder, iter_limit, N, discount_factor; active_only = active_only)
    end
    return 
end

@everywhere function active_job_toy(folder, iter_list, discount_factor)

    active_cuts_data = []
    # Which cuts are active, keyed by iteration limit. `active_cuts.csv` keeps
    # only their number, as before; this is what `evaluate_job` replays from.
    indices_by_limit = Dict{Int,Dict{Int,Vector{Int}}}()
    for iter_limit in iter_list
        model = RVSDDP.PolicyGraph(
            subproblem_builder,
            graph;
            sense = :Min,
            lower_bound = 0.0,
            optimizer = optimizer,
            discount_factor=discount_factor,
        )

        RVSDDP._add_cuts_iter(model, iter_limit, folder);

        active_cuts = RVSDDP.all_active_cut_indices(model, 1e-4)
        indices_by_limit[iter_limit] = active_cuts

        for t in 1:1
            push!(active_cuts_data, Dict(
                :time => iter_limit,
                :stage => t,
                :num_active_cuts => length(active_cuts[t]),
                :num_cuts => length(model[t].value_function.cut_V),
            ))
        end
    end

    CSV.write("$(folder)/active_cuts.csv", DataFrame(active_cuts_data))
    RVSDDP.save_active_cut_indices(folder, indices_by_limit)

end

function run_active_toy(seed_list, parallel, iter_max_list, shift_function_list, discount_factor_list, refine_scheme_list, iter_list)
    combos = [("results_toy/$(RVSDDP.method_label(shift_function, refine_scheme))_parallel_$(parallel)/$(discount_factor)/seed_$(seed)_iter_$(iter_max)", iter_list, discount_factor) for seed in seed_list for iter_max in iter_max_list for shift_function in shift_function_list for discount_factor in discount_factor_list for refine_scheme in refine_scheme_list]

    results = pmap(combos) do (folder, iter, discount_factor)
        active_job_toy(folder, iter, discount_factor)
    end
    return 
end