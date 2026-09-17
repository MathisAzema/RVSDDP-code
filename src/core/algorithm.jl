#  Copyright (c) 2017-25, Oscar Dowson and RVSDDP.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

# The training loop: one `iteration` is a forward pass followed by a backward
# pass, and `_training_loop` repeats it until a stopping rule fires. Also holds
# `Trajectory`, the batch of trial states the two passes exchange.

mutable struct Trajectory{T}
    scenario_path::Vector{Tuple{T, Any}}
    sampled_states::Vector{Dict{Symbol,Float64}}
    cumulative_value::Float64
end

struct IterationResult{T}
    pid::Int
    bound::Float64
    cumulative_value::Float64
    has_converged::Bool
    status::Symbol
    cuts::Dict{T,Vector{Any}}
    numerical_issue::Bool
end

function iteration(model::PolicyGraph{T}, options::Options) where {T}
    model.ext[:numerical_issue] = false
    @_timeit_threadsafe model.timer_output "forward_pass" begin
        forward_trajectory = forward_pass(model, options, options.forward_pass)
        options.forward_pass_callback(forward_trajectory)
    end
    @_timeit_threadsafe model.timer_output "backward_pass" begin
        cuts = backward_pass(
            model,
            options,
            forward_trajectory,
        )
    end
    # The lower bound reported in the log: V(x0), which `backward_pass` has
    # just pushed onto `model.approx_value`. Note that the shift mechanism
    # breaks the lower-bound property, so this is only a valid bound for an
    # unshifted run.
    bound = model.approx_value[end][2]
    lock(options.lock)
    try
        push!(
            options.log,
            Log(
                length(options.log) + 1,
                sum([length(node.value_function.cut_V) for (_,node) in model.nodes]),
                bound,
                forward_trajectory[1].cumulative_value,
                time() - options.start_time,
                max(Threads.threadid(), Distributed.myid()),
                lock(() -> model.ext[:total_solves], model.lock),
                duality_log_key(options.duality_handler),
                lock(() -> model.ext[:numerical_issue], model.lock),
                cuts,
            ),
        )
        has_converged, status =
            convergence_test(model, options.log, options.stopping_rules)
        return IterationResult(
            max(Threads.threadid(), Distributed.myid()),
            bound,
            forward_trajectory[1].cumulative_value,
            has_converged,
            status,
            cuts,
            lock(() -> model.ext[:numerical_issue], model.lock),
        )
    finally
        unlock(options.lock)
    end
end

"""
    termination_status(model::PolicyGraph)::Symbol

Query the reason why the training stopped.
"""
function termination_status(model::PolicyGraph)
    if model.most_recent_training_results === nothing
        return :model_not_solved
    end
    return model.most_recent_training_results.status
end

function _should_log(options)
    return options.print_level > 0 && options.log_frequency(options.log)
end

function log_iteration(options; force_if_needed::Bool = false)
    force_if_needed &= options.last_log_iteration[] != length(options.log)
    if force_if_needed || _should_log(options)
        print_helper(print_iteration, options.log_file_handle, options.log[end])
        flush(options.log_file_handle)
        options.last_log_iteration[] = length(options.log)
    end
    return
end

# Internal: run iterations until a stopping rule fires.
#
# The batch of an iteration is already spread over the available cores (see
# `_parallel_foreach`), and its cuts mutate the shared model, so iterations
# themselves are run one after another.
function _training_loop(model::PolicyGraph{T}, options::Options) where {T}
    status = nothing
    while status === nothing
        # Disable CTRL+C so that InterruptExceptions can be thrown only between
        # each iteration. Note that if the user presses CTRL+C during an
        # iteration, then this will be cached and re-thrown as disable_sigint
        # exits.
        status = disable_sigint() do
            result = iteration(model, options)
            options.post_iteration_callback(result)
            log_iteration(options)
            if result.has_converged
                return result.status
            end
            return nothing
        end
    end
    return status
end

"""
    RVSDDP.train(model::PolicyGraph; kwargs...)

Train the policy for `model`.

## Keyword arguments

 - `iteration_limit::Int`: number of iterations to conduct before termination.

 - `time_limit::Float64`: number of seconds to train before termination.

 - `stoping_rules`: a vector of [`RVSDDP.AbstractStoppingRule`](@ref)s. There is
   no default; you must specify this, `iteration_limit`, `time_limit`, or
   `cut_limit`.

 - `print_level::Int`: control the level of printing to the screen. Defaults to
    `1`. Set to `0` to disable all printing.

 - `log_file::String`: filepath at which to write a log of the training
   progress. Defaults to `RVSDDP.log`.

 - `log_frequency::Int`: control the frequency with which the logging is
    outputted (iterations/log). It must be at least `1`. Defaults to `1`.

 - `log_every_seconds::Float64`: control the frequency with which the logging is
   outputted (seconds/log). Defaults to `0.0`.

 - `log_every_iteration::Bool`; over-rides `log_frequency` and `log_every_seconds`
   to force every iteration to be printed. Defaults to `false`.

 - `run_numerical_stability_report::Bool`: generate (and print) a numerical
   stability report prior to solve. Defaults to `true`.

 - `risk_measure`: the risk measure to use at each node. Defaults to
   [`Expectation`](@ref).

 - `sampling_scheme`: a sampling scheme to use on the forward pass of the
    algorithm. Defaults to [`InSampleMonteCarlo`](@ref).

 - `backward_sampling_scheme`: a backward pass sampling scheme to use on the
    backward pass of the algorithm. Defaults to `CompleteSampler`.

 - `forward_pass::AbstractForwardPass`: specify a scheme to use for the forward
   passes.

 - `add_to_existing_cuts::Bool`: set to `true` to allow training a model that
   was previously trained. Defaults to `false`.

 - `duality_handler::AbstractDualityHandler`: specify a duality handler to use
   when creating cuts.

 - `post_iteration_callback::Function`: a callback with the signature
   `post_iteration_callback(::IterationResult)` that is evaluated after each
   iteration of the algorithm.

The two choices that make up a method of the computational experiments:

 - `shift_function`: the shift-selection rule. Defaults to
   [`RVSDDP.no_shift`](@ref), which reduces RV-SDDP to cyclic SDDP; use
   [`RVSDDP.random_shift`](@ref) for RV-SDDP proper.

 - `refine_scheme`: which trial states of the forward pass receive a cut.
   Defaults to [`RVSDDP.refine_all`](@ref), i.e. every visited state (scheme
   `A`); use [`RVSDDP.refine_periodic`](@ref) for one random block of `T`
   consecutive stages per iteration (scheme `B`).

[`RVSDDP.method_label`](@ref) turns that pair into the name the results are
stored under.

"""
function train(
    model::PolicyGraph;
    cut_limit::Union{Int,Nothing} = nothing,
    iteration_limit::Union{Int,Nothing} = nothing,
    time_limit::Union{Real,Nothing} = nothing,
    print_level::Int = 0,
    log_file::String = "RVSDDP.log",
    log_frequency::Int = 1,
    log_every_seconds::Float64 = log_frequency == 1 ? -1.0 : 0.0,
    log_every_iteration::Bool = false,
    run_numerical_stability_report::Bool = true,
    stopping_rules = AbstractStoppingRule[],
    risk_measure = RVSDDP.Expectation(),
    sampling_scheme = RVSDDP.InSampleMonteCarlo(),
    backward_sampling_scheme::AbstractBackwardSamplingScheme = RVSDDP.CompleteSampler(),
    forward_pass::AbstractForwardPass = DefaultForwardPass(),
    add_to_existing_cuts::Bool = false,
    duality_handler::AbstractDualityHandler = RVSDDP.ContinuousConicDuality(),
    forward_pass_callback::Function = (x) -> nothing,
    post_iteration_callback = result -> nothing,
    infinite::Bool=false,
    discount_factor::Float64=0.1,
    shift_function::Function=RVSDDP.no_shift,
    parallel::Int64=1,
    refine_scheme::Function = RVSDDP.refine_all,
)
    if log_frequency <= 0
        msg = "`log_frequency` must be at least `1`. Got $log_frequency."
        throw(ArgumentError(msg))
    end
    if log_every_iteration
        log_frequency = 1
        log_every_seconds = 0.0
    end
    function log_frequency_f(log::Vector{Log})
        if mod(length(log), log_frequency) != 0
            return false
        end
        last = options.last_log_iteration[]
        if last == 0
            return true
        elseif last == length(log)
            return false
        end
        seconds = log_every_seconds
        if log_every_seconds < 0.0
            if log[end].time <= 10
                seconds = 1.0
            elseif log[end].time <= 120
                seconds = 5.0
            else
                seconds = 30.0
            end
        end
        return log[end].time - log[last].time >= seconds
    end

    if !add_to_existing_cuts && model.most_recent_training_results !== nothing
        @warn("""
        Re-training a model with existing cuts!

        Are you sure you want to do this? The output from this training may be
        misleading because the policy is already partially trained.

        If you meant to train a new policy with different settings, you must
        build a new model.

        If you meant to refine a previously trained policy, turn off this
        warning by passing `add_to_existing_cuts = true` as a keyword argument
        to `RVSDDP.train`.

        In a future release, this warning may turn into an error.
        """)
    end
    # Reset the TimerOutput.
    TimerOutputs.reset_timer!(model.timer_output)
    log_file_handle = open(log_file, "a")
    log = Log[]

    if print_level > 0
        print_helper(print_banner, log_file_handle)
        print_helper(
            print_problem_statistics,
            log_file_handle,
            model,
            model.most_recent_training_results !== nothing,
            risk_measure,
            sampling_scheme,
        )
    end
    if run_numerical_stability_report
        report = sprint(
            io -> numerical_stability_report(
                io,
                model;
                print = print_level > 0,
            ),
        )
        print_helper(print, log_file_handle, report)
    end
    if print_level > 0
        print_helper(print_iteration_header, log_file_handle)
    end
    # Convert the vector to an AbstractStoppingRule. Otherwise if the user gives
    # something like stopping_rules = [RVSDDP.IterationLimit(100)], the vector
    # will be concretely typed and we can't add a TimeLimit.
    stopping_rules = convert(Vector{AbstractStoppingRule}, stopping_rules)
    # Add the limits as stopping rules. An IterationLimit or TimeLimit may
    # already exist in stopping_rules, but that doesn't matter.
    if iteration_limit !== nothing
        push!(stopping_rules, IterationLimit(iteration_limit))
    end
    if time_limit !== nothing
        push!(stopping_rules, TimeLimit(time_limit))
    end
    if cut_limit !== nothing
        push!(stopping_rules, CutLimit(cut_limit))
    end
    # There is no default stopping rule: the caller must specify at least one
    # of `iteration_limit`, `time_limit`, `cut_limit`, or `stopping_rules`.
    if isempty(stopping_rules)
        error(
            "No stopping rule specified. Pass `iteration_limit`, " *
            "`time_limit`, `cut_limit`, or `stopping_rules` to `train`.",
        )
    end
    # `parallel` trajectories are simulated, and their children solved, at the
    # same time. That needs `parallel - 1` extra copies of every subproblem, so
    # that no two of them share a JuMP model; build any that are missing (this
    # is a no-op when the graph was created with `max_parallel >= parallel`).
    if parallel > 1
        _build_replicas!(model, parallel)
        if Threads.nthreads() < parallel
            @warn(
                "`parallel = $(parallel)` but Julia was started with only " *
                "$(Threads.nthreads()) thread(s), so the batch cannot use " *
                "$(parallel) cores. Start Julia with `julia -t $(parallel)` " *
                "(or set `JULIA_NUM_THREADS=$(parallel)`); with `Distributed`, " *
                "pass `addprocs(n; exeflags = \"-t $(parallel)\")`.",
                maxlog = 1,
            )
        end
    end
    dashboard_callback = (::Any, ::Any) -> nothing
    options = Options(
        model,
        model.initial_root_state;
        sampling_scheme,
        backward_sampling_scheme,
        risk_measures = risk_measure,
        stopping_rules,
        dashboard_callback,
        print_level,
        start_time = time(),
        log,
        log_file_handle,
        log_frequency = log_frequency_f,
        forward_pass,
        duality_handler,
        forward_pass_callback,
        post_iteration_callback,
        infinite,
        shift_function,
        parallel,
        refine_scheme,
    )
    status = :not_solved
    try
        status = _training_loop(model, options)
    catch ex
        # Unwrap exceptions from tasks. If there are multiple exceptions,
        # rethrow only the last one.
        if ex isa CompositeException
            ex = last(ex.exceptions)
        end
        if ex isa TaskFailedException
            ex = ex.task.exception
        end
        if ex isa InterruptException
            status = :interrupted
        else
            close(log_file_handle)
            throw(ex)
        end
    finally
        # And close the dashboard callback if necessary.
        dashboard_callback(nothing, true)
    end
    training_results = TrainingResults(status, log)
    model.most_recent_training_results = training_results
    if print_level > 0
        log_iteration(options; force_if_needed = true)
        print_helper(print_footer, log_file_handle, training_results)
        if print_level > 1
            print_helper(
                TimerOutputs.print_timer,
                log_file_handle,
                model.timer_output,
            )
            # Annoyingly, TimerOutputs doesn't end the print section with `\n`,
            # so we do it here.
            print_helper(println, log_file_handle)
        end
    end
    close(log_file_handle)
    return log
end
