# Verification of the real `parallel = k` batch (per-worker subproblem replicas
# + threaded forward/backward passes).
#
# This file has two halves.
#
# 1. `check_replicas(model)` -- a diagnostic you can call on ANY trained model,
#    including the Brazilian one. It is deliberately NOT part of the RVSDDP
#    package: nothing in the algorithm calls it. Use it with
#
#        include("test_parallel_check.jl")
#        check_replicas(model)
#
#    which is what test_parallel_gurobi.jl and test_parallel_gurobi_toy.jl do.
#    Including this file does not run the checks below.
#
# 2. A self-contained check of the parallel batch, run when this file is
#    executed as a script (or via `run_parallel_checks()`). It complements
#    test_parallel_gurobi*.jl: those train a few models and eyeball whether a
#    lower bound looks wildly wrong, which only catches gross failures. This one
#    checks the two properties the batch actually rests on, exactly:
#
#      REPRODUCIBILITY -- the same seed must give bit-identical cuts, both when
#      the run is repeated in this process and across thread counts. A data race
#      would break this.
#
#      REPLICA SYNC -- every replica must still describe the same subproblem as
#      the master. A replica missing a cut (or a shift) would return duals for a
#      stale value function and generate an invalid cut; this is the failure
#      mode that the earlier prototype hit.
#
#    Run it twice, with different thread counts. The first run writes a
#    fingerprint file, the second compares against it:
#
#        julia -t 1 --project=. test_parallel_check.jl
#        julia -t 4 --project=. test_parallel_check.jl
#
#    The toy model of notebook_toy.ipynb is used because it exercises every code
#    path (replica construction, cut mirroring, shift mirroring, cut replay) in
#    seconds.

import Pkg
Pkg.activate(@__DIR__)

using RVSDDP
using Random
using Printf

# ============================================================================ #
#  Part 1: the diagnostic, usable on any trained model.
# ============================================================================ #

"""
    check_replicas(model::RVSDDP.PolicyGraph; kwargs...)

Check that every subproblem replica used by a `parallel > 1` batch still
describes exactly the same problem as the master node.

This is the property the whole parallel batch rests on: worker `r` computes its
cuts on `node.replicas[r-1]`, so if a replica ever missed a cut (or a shift), it
would return duals for a stale value function and generate an invalid cut --- a
failure that shows up as a bound above the true optimum, not as a crash.

The check re-solves each node and each of its replicas at the same incoming
states and noise terms and compares the optimal values, which must agree to
solver tolerance. It returns the largest absolute difference found (`0.0` when
the model has no replicas, i.e. when there is nothing to check).

## Keyword arguments

 - `n_points::Int = 5`: how many incoming states to probe per node. They are
   drawn from the state box, plus the initial state.
 - `max_noise::Int = 10`: probe at most this many noise terms per node; useful
   on models with many realizations.
 - `atol::Float64 = 1e-6`: tolerance above which a difference is reported.
 - `seed::Int = 1234`: seed of the probe points, so the check is reproducible.
 - `verbose::Bool = true`: print a one-line verdict.

## Example

```julia
include("test_parallel_check.jl")
RVSDDP.train(model; parallel = 10, ...)
check_replicas(model)
```
"""
function check_replicas(
    model::RVSDDP.PolicyGraph{T};
    n_points::Int = 5,
    max_noise::Int = 10,
    atol::Float64 = 1e-6,
    seed::Int = 1234,
    verbose::Bool = true,
) where {T}
    rng = Random.MersenneTwister(seed)
    worst, worst_at, n_probe = 0.0, nothing, 0
    for (index, node) in model.nodes
        if isempty(node.replicas)
            continue
        end
        noises = node.noise_terms
        if length(noises) > max_noise
            step = max(div(length(noises), max_noise), 1)
            noises = noises[1:step:end]
        end
        for state in _replica_probe_states(model, node, n_points, rng)
            for noise in noises
                reference = RVSDDP.solve_subproblem(
                    model,
                    node,
                    state,
                    noise.term;
                    duality_handler = nothing,
                )
                for (r, replica) in enumerate(node.replicas)
                    candidate = RVSDDP.solve_subproblem(
                        model,
                        replica,
                        state,
                        noise.term;
                        duality_handler = nothing,
                    )
                    n_probe += 1
                    difference = abs(reference.objective - candidate.objective)
                    if difference > worst
                        worst, worst_at = difference, (index, r)
                    end
                end
            end
        end
    end
    if verbose
        if n_probe == 0
            println(
                "check_replicas: this model has no replicas (it was never " *
                "trained with `parallel > 1`), nothing to check.",
            )
        elseif worst <= atol
            @printf(
                "check_replicas: %d probes, max |Δobjective| = %.3e -- OK, every replica matches the master.\n",
                n_probe,
                worst,
            )
        else
            @printf(
                "check_replicas: %d probes, max |Δobjective| = %.3e at node %s replica %d -- MISMATCH, a replica is out of sync with the master.\n",
                n_probe,
                worst,
                string(worst_at[1]),
                worst_at[2],
            )
        end
    end
    return worst
end

# Incoming states at which to compare a node with its replicas. The
# state box comes from the master's recorded state bounds (replicas do not have
# them); the initial state is always included because it is the point the bound
# is reported at.
function _replica_probe_states(
    model::RVSDDP.PolicyGraph{T},
    node::RVSDDP.Node{T},
    n_points::Int,
    rng::Random.AbstractRNG,
) where {T}
    keys_ = collect(keys(node.states))
    states = Dict{Symbol,Float64}[]
    initial = Dict{Symbol,Float64}(
        k => get(model.initial_root_state, k, 0.0) for k in keys_
    )
    push!(states, initial)
    for _ in 1:n_points
        state = Dict{Symbol,Float64}()
        for k in keys_
            lower = get(node.state_lower_bounds, k, NaN)
            upper = get(node.state_upper_bounds, k, NaN)
            state[k] = if isfinite(lower) && isfinite(upper)
                lower + rand(rng) * (upper - lower)
            else
                initial[k]
            end
        end
        push!(states, state)
    end
    return states
end

# ============================================================================ #
#  Part 2: the self-check of the parallel batch (toy model). Only runs when this
#  file is executed as a script, so that `include`ing it stays cheap and side
#  effect free -- in particular no second Gurobi environment is created.
# ============================================================================ #

using Gurobi

# The Gurobi environment is created on first use, not on `include`, so that
# loading this file into a process that already has one (test_parallel_gurobi.jl
# and friends) costs nothing and clashes with nothing.
const _CHECK_ENV = Ref{Union{Nothing,Gurobi.Env}}(nothing)

function check_optimizer()
    if _CHECK_ENV[] === nothing
        _CHECK_ENV[] = Gurobi.Env()
    end
    env = _CHECK_ENV[]
    return () -> Gurobi.Optimizer(env)
end

const DISCOUNT = 0.99
const PERIOD = 1
const FINGERPRINT_DIR = joinpath(@__DIR__, "results_parallel_check")

function subproblem_builder(subproblem::Model, node::Int, discount_factor::Float64)
    @variable(subproblem, 0 <= volume <= 200, RVSDDP.State, initial_value = 50)
    @variables(subproblem, begin
        thermal_generation >= 0
        hydro_generation >= 0
        hydro_spill >= 0
        deficit >= 0
    end)
    @variable(subproblem, inflow)
    Ω = [20.0, 90.0]
    P = [1 / length(Ω) for _ in Ω]
    RVSDDP.parameterize(subproblem, Ω, P) do ω
        return JuMP.fix(inflow, ω)
    end
    @constraints(subproblem, begin
        volume.out == volume.in - hydro_generation - hydro_spill + inflow
        hydro_generation <= 100
        thermal_generation <= 30
        deficit + hydro_generation + thermal_generation == 60
    end)
    @stageobjective(subproblem, 50 * hydro_spill + 50 * deficit + 2 * thermal_generation)
    return subproblem
end

const GRAPH = RVSDDP.InfiniteLinearGraph(PERIOD)

function train_once(; parallel::Int, seed::Int, iterations::Int, shift)
    model = RVSDDP.PolicyGraph(
        subproblem_builder,
        GRAPH;
        sense = :Min,
        lower_bound = 0.0,
        optimizer = check_optimizer(),
        discount_factor = DISCOUNT,
        max_parallel = parallel,
    )
    Random.seed!(seed)
    RVSDDP.train(
        model;
        refine_mode = 0,
        parallel = parallel,
        sampling_scheme = RVSDDP.InSampleMonteCarlo(
            max_depth = 10_000,
            rollout_limit = i -> PERIOD * i,
            parallel = parallel,
        ),
        iteration_limit = iterations,
        infinite = true,
        shift_function = shift,
    )
    return model
end

# Every cut of every node, shifts included: two runs agree only if they produced
# exactly the same value function approximation.
function fingerprint(model)
    io = IOBuffer()
    for index in sort(collect(keys(model.nodes)))
        for (i, cut) in enumerate(model[index].value_function.cut_V)
            @printf(io, "%s %d %.17g %.17g", string(index), i, cut.intercept, cut.shift[end][1])
            for (k, v) in sort(collect(cut.coefficients))
                @printf(io, " %s=%.17g", k, v)
            end
            println(io)
        end
    end
    return String(take!(io))
end

const CASES = [
    (name = "cyclic-SDDP (no shift)", shift = RVSDDP.no_shift),
    (name = "RV-SDDP (random shift)", shift = RVSDDP.random_shift),
]

function run_parallel_checks()
    println("Threads.nthreads() = ", Threads.nthreads())
    if Threads.nthreads() == 1
        println("(single thread: the batch runs sequentially -- this run is the reference)")
    end
    println()
    failures = String[]
    report = IOBuffer()
    for case in CASES, parallel in (1, 2, 4)
        label = @sprintf("%-24s parallel=%d", case.name, parallel)

        # --- check 1a: repeating the run in this process changes nothing ------
        model = train_once(; parallel = parallel, seed = 1, iterations = 25, shift = case.shift)
        fp = fingerprint(model)
        again = fingerprint(train_once(; parallel = parallel, seed = 1, iterations = 25, shift = case.shift))
        repeatable = fp == again
        repeatable || push!(failures, "$label: two runs with the same seed disagree (data race?)")

        # --- check 2: the replicas still match the master --------------------
        worst = check_replicas(model; verbose = false)
        worst <= 1e-6 || push!(failures, "$label: replica out of sync, max |Δobjective| = $worst")

        @printf("%s  repeatable=%-5s  replica max|Δ|=%.1e  cuts=%d  nreplicas=%d\n",
                label, repeatable, worst,
                length(model[1].value_function.cut_V), length(model[1].replicas))
        print(report, label, "\n", fp)
    end

    # --- check 1b: the result does not depend on the number of threads --------
    mkpath(FINGERPRINT_DIR)
    mine = joinpath(FINGERPRINT_DIR, "fingerprint_t$(Threads.nthreads()).txt")
    write(mine, String(take!(report)))
    println("\nfingerprint written to $(relpath(mine, @__DIR__))")
    others = filter(f -> startswith(f, "fingerprint_t") && f != basename(mine),
                    readdir(FINGERPRINT_DIR))
    if isempty(others)
        println("no other thread count recorded yet -- re-run with a different `-t N` to compare.")
    else
        for other in others
            if read(mine, String) == read(joinpath(FINGERPRINT_DIR, other), String)
                println("identical to $other  -- OK, the result does not depend on the thread count.")
            else
                push!(failures, "results differ between $(basename(mine)) and $other")
            end
        end
    end

    println()
    if isempty(failures)
        println("ALL CHECKS PASSED")
    else
        println("FAILURES:")
        foreach(f -> println("  - ", f), failures)
    end
    return isempty(failures)
end


if abspath(PROGRAM_FILE) == @__FILE__
    run_parallel_checks()
end
