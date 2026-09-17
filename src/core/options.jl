#  Copyright (c) 2017-25, Oscar Dowson and RVSDDP.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

# `Options` gathers everything a run needs that is not part of the model
# itself: the algorithmic choices made at the `train` call and the mutable
# bookkeeping carried across iterations (log, starting states, timings).

# Internal struct: storage for RVSDDP options and cached data. Users shouldn't
# interact with this directly.
struct Options
    # The initial state to start from the root node.
    initial_state::Dict{Symbol,Float64}
    # The sampling scheme to use on the forward pass.
    sampling_scheme::AbstractSamplingScheme
    backward_sampling_scheme::AbstractBackwardSamplingScheme
    # Risk measure applied at every node. `Expectation` is the only one the
    # paper uses.
    risk_measure::AbstractRiskMeasure
    stopping_rules::Vector{AbstractStoppingRule}
    dashboard_callback::Function
    print_level::Int
    start_time::Float64
    log::Vector{Log}
    log_file_handle::Any
    log_frequency::Union{Int,Function}
    forward_pass::AbstractForwardPass
    duality_handler::AbstractDualityHandler
    # A callback called after the forward pass.
    forward_pass_callback::Any
    post_iteration_callback::Any
    last_log_iteration::Ref{Int}
    # For threading
    lock::ReentrantLock
    infinite::Bool
    shift_function::Function
    parallel::Int64
    start::Float64
    refine_scheme::Function
    # Internal function: users should never construct this themselves.
    function Options(
        initial_state::Dict{Symbol,Float64};
        sampling_scheme::AbstractSamplingScheme = InSampleMonteCarlo(),
        backward_sampling_scheme::AbstractBackwardSamplingScheme = CompleteSampler(),
        risk_measure::AbstractRiskMeasure = Expectation(),
        stopping_rules::Vector{AbstractStoppingRule} = RVSDDP.AbstractStoppingRule[],
        dashboard_callback::Function = (a, b) -> nothing,
        print_level::Int = 0,
        start_time::Float64 = 0.0,
        log::Vector{Log} = Log[],
        log_file_handle = IOBuffer(),
        log_frequency::Union{Int,Function} = 1,
        forward_pass::AbstractForwardPass = DefaultForwardPass(),
        duality_handler::AbstractDualityHandler = ContinuousConicDuality(),
        forward_pass_callback = x -> nothing,
        post_iteration_callback = result -> nothing,
        infinite::Bool = false,
        shift_function::Function = RVSDDP.no_shift,
        parallel::Int64 = 1,
        refine_scheme::Function = RVSDDP.refine_all,
    )
        return new(
            initial_state,
            sampling_scheme,
            backward_sampling_scheme,
            risk_measure,
            stopping_rules,
            dashboard_callback,
            print_level,
            start_time,
            log,
            log_file_handle,
            log_frequency,
            forward_pass,
            duality_handler,
            forward_pass_callback,
            post_iteration_callback,
            Ref{Int}(0),  # last_log_iteration
            ReentrantLock(),
            infinite,
            shift_function,
            parallel,
            time(),
            refine_scheme,
        )
    end
end
