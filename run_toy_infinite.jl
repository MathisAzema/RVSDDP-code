# Experiences sur le probleme jouet de la section 6.1 (figures 3 a 5).
#
# Reproduire les resultats de l'article
# -------------------------------------
# Depuis la racine du depot (la premiere fois seulement :
# `julia --project=. -e 'import Pkg; Pkg.instantiate()'`), lancer `julia`, puis :
#
#     include("run_toy_infinite.jl")
#
#     # 1. Entrainement : 100 iterations, beta = 0.99, L(k) = k, batch de 1,
#     #    pour RV-SDDP (random_shift) et Cyclic SDDP (no_shift).
#     run_toy_infinite(1:10, 1, [100], [RVSDDP.random_shift, RVSDDP.no_shift], [0.99], [RVSDDP.refine_all])
#
#     # 2. Coupes actives apres chaque iteration : panneau de droite de la
#     #    figure 5, et coupes que l'etape 3 rejoue. A faire avant l'etape 3.
#     run_active_toy(1:10, 1, [100], [RVSDDP.random_shift, RVSDDP.no_shift], [0.99], [RVSDDP.refine_all], 1:100)
#
#     # 3. Evaluation hors echantillon (Protocole 1, N = 1000) toutes les trois
#     #    iterations. Attention : ici la liste d'iterations vient AVANT le schema.
#     run_evaluate(1:10, 1, [100], [RVSDDP.random_shift, RVSDDP.no_shift], [0.99], 1:3:100, [RVSDDP.refine_all], [1000])
#
#     # 4. Les figures sont ensuite tracees par `results_toy.ipynb` (la graine 1
#     #    pour les figures 3 et 4, les dix graines pour la figure 5).

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

@everywhere Gurobi.GRBsetintparam(GRB_ENV, "Threads", 1)
@everywhere optimizer=() -> Gurobi.Optimizer(GRB_ENV)

# Probleme jouet de la section 6.1 de l'article, equation (18). Doit rester
# identique au `subproblem_builder` de notebook_toy.ipynb et results_toy.ipynb,
# qui rejouent les coupes ecrites ici.
@everywhere function subproblem_builder(subproblem::Model, node::Int, discount_factor::Float64)
    # State variables
    @variable(subproblem, 0 <= volume <= 200, RVSDDP.State, initial_value = 50)
    # Control variables
    @variables(subproblem, begin
        thermal_generation >= 0
        hydro_generation >= 0
        hydro_spill >= 0
        deficit >= 0
    end)
    # Random variables
    @variable(subproblem, inflow)
    Ω = [20.0, 90.0]
    P = [1 / length(Ω) for _ in Ω]
    RVSDDP.parameterize(subproblem, Ω, P) do ω
        return JuMP.fix(inflow, ω)
    end

    # Transition function and constraints
    @constraints(
        subproblem,
        begin
            volume.out == volume.in - hydro_generation - hydro_spill + inflow
            hydro_generation <= 100
            thermal_generation <= 30
            deficit + hydro_generation + thermal_generation == 60
        end
    )
    # Stage-objective
    @stageobjective(subproblem, 50*hydro_spill + 50 * deficit+ 2*thermal_generation)
    return subproblem
end

@everywhere graph=RVSDDP.InfiniteLinearGraph(1);

@everywhere using CSV, DataFrames, JSON

# Rend au systeme la memoire du job qui vient de se terminer sur ce worker. Ses
# modeles ne sont liberes (finaliseurs Gurobi compris) qu'au passage du
# ramasse-miettes, qu'un worker inactif ne declenche jamais : sans cet appel, la
# memoire d'un run de 10 h resterait occupee tant que le worker vit, par exemple
# pendant qu'il attend la fin des autres. `malloc_trim` rend ensuite au systeme
# ce que glibc garde de cote. Le socle du worker (Julia, paquets, code compile,
# ~0,9 Go) reste, lui, jusqu'a la fin de la session. A appeler une fois le job
# revenu, pas dans le job : ses variables locales le garderaient en vie.
@everywhere function release_memory()
    GC.gc(true)
    GC.gc(true)   # le second passage recupere ce que les finaliseurs ont libere
    Sys.islinux() && ccall(:malloc_trim, Cint, (Cint,), 0)
    return
end

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

    RVSDDP.save_cuts(model, folder3)=
    RVSDDP.save_parallel_timings(model, folder3)

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
        release_memory()
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
        release_memory()
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
                # La borne inferieure psi0 n'est pas une coupe, voir run_msppy.jl.
                :num_active_cuts => count(>(1), active_cuts[t]),
                :num_cuts => RVSDDP.count_cuts(model[t]),
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
        release_memory()
    end
    return 
end
