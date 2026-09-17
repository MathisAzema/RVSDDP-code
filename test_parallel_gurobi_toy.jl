# TEST/DIAGNOSTIC ONLY -- standalone terminal version of the Gurobi
# thread-safety check, using the small toy problem (notebook_toy.ipynb)
# instead of the Brazilian model (see test_parallel_gurobi.jl for that one,
# and the TEST/DIAGNOSTIC ONLY comments in src/core/algorithm.jl,
# src/core/cuts.jl, src/core/forward_passes.jl and
# src/user_interface.jl).
#
# Usage: start ONE persistent Julia REPL (not `julia script.jl`, which exits
# after running once) with enough threads:
#
#     julia -t 10 --project=.
#
# then, inside that REPL:
#
#     julia> include("test_parallel_gurobi_toy.jl")   # slow: imports + Gurobi env + runs once
#     julia> run_diagnostic()                          # fast: re-run as many times as you like
#     julia> run_diagnostic(n_trials = 20, parallel = 4)
#
# `-t 10` is what actually gives this Julia process 10 threads; it must be
# >= max_parallel/parallel used by run_diagnostic. There is no
# Distributed/addprocs here on purpose, to avoid the "worker processes
# default to 1 thread each" trap -- everything runs in this one process.

import Pkg
Pkg.activate(".")

using RVSDDP
using Random
using Statistics
using Gurobi

# `check_replicas` lives in test_parallel_check.jl rather than in the package:
# nothing in the algorithm calls it, it is a diagnostic. Including that file
# only defines the function, it does not run its own checks.
include(joinpath(@__DIR__, "test_parallel_check.jl"))

const GRB_ENV = Gurobi.Env()
optimizer = () -> Gurobi.Optimizer(GRB_ENV)

println("Threads.nthreads() = ", Threads.nthreads())
if Threads.nthreads() == 1
    error("Started with only 1 thread. Re-run as `julia -t 10 --project=. test_parallel_gurobi_toy.jl`.")
end

discount_factor = 0.99
period = 1

function subproblem_builder(subproblem::Model, node::Int, discount_factor::Float64)
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

graph = RVSDDP.InfiniteLinearGraph(period);

# ------------------------------------------------------------------
# Diagnostic: train `n_trials` fresh models with `parallel = parallel`
# (real replicas + Threads.@threads), and flag any run whose lower_bound
# is wildly off from the others -- that's what caught the HiGHS issue on
# the Brazilian model. DO NOT trust a single run/trial: call
# `run_diagnostic()` several times.
#
# Everything above this point (imports, Gurobi env, builder, graph) only
# needs to run once per session -- that's the slow part (precompilation,
# JIT warmup). Once you've `include`d this file, just call
# `run_diagnostic()` again from the same REPL as many times as you want;
# it reuses everything already loaded and only pays the cost of training.
#
# Unlike the Brazilian model, this toy problem is tiny and trains almost
# instantly, so we control the workload with `iteration_limit` (as in
# notebook_toy.ipynb) rather than `time_limit`.
# ------------------------------------------------------------------

function run_diagnostic_toy(; n_trials::Int = 1, parallel::Int = 10, iteration_limit::Int = 2)
    println("Threads.nthreads() = ", Threads.nthreads())
    results = Float64[]
    for trial in 1:n_trials
        model_cyclic_sddp = RVSDDP.PolicyGraph(
            subproblem_builder,
            graph;
            sense = :Min,
            lower_bound = 0.0,
            optimizer = optimizer,
            discount_factor = discount_factor,
            max_parallel = parallel,
        )

        Random.seed!(trial)
        RVSDDP.train(
            model_cyclic_sddp;
            refine_mode = 0,
            parallel = parallel,
            sampling_scheme = RVSDDP.InSampleMonteCarlo(
                max_depth = 10000,
                rollout_limit = i -> period * i,
                parallel = parallel,
            ),
            iteration_limit = iteration_limit,
            infinite = true,
            shift_function = RVSDDP.no_shift,
        )

        # V_0(x_0): the value function approximation at the initial state,
        # i.e. exactly what RVSDDP.train already stores as model.approx_value
        # after every backward pass.
        v = model_cyclic_sddp.approx_value[end][2]
        push!(results, v)
        total_cuts = sum(length(node.value_function.cut_V) for node in values(model_cyclic_sddp.nodes))
        # The property the parallel batch rests on: every worker's replica of a
        # subproblem must still carry exactly the cuts the master has. A replica
        # that drifted would produce duals for a stale value function, and so an
        # invalid cut -- which is what a wrong lower bound really means here.
        check_replicas(model_cyclic_sddp)
        println("trial $trial (seed=$trial): lower_bound = $v, total cuts = $total_cuts")
        # for cut in model_cyclic_sddp.nodes[1].value_function.cut_V
        #     println("  cut: ", cut)
        # end
    end

    # med = Statistics.median(results)
    # anomaly = false
    # for (trial, v) in enumerate(results)
    #     if v > 10 * med || v < med / 10
    #         println("  <<<< ANOMALY at trial $trial: $v vs median $med")
    #         anomaly = true
    #     end
    # end
    # println(anomaly ? "NOT SAFE: at least one anomalous run." : "No anomaly this run -- call run_diagnostic() a few more times before concluding it's safe.")
    return results
end

# Runs once when you `include` this file. After that, just call
# `run_diagnostic()` again directly from the REPL -- no need to re-include.
run_diagnostic_toy()
