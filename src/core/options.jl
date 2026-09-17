#  Copyright (c) 2017-25, Oscar Dowson and RVSDDP.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

# `Options` gathers everything a run needs that is not part of the model
# itself: the algorithmic choices made at the `train` call, the mutable
# bookkeeping carried across iterations (log, starting states, timings), and
# the helpers that turn a user-supplied argument into its per-node form.

# to_nodal_form is an internal helper function so users can pass arguments like:
# risk_measure = RVSDDP.Expectation(),
# risk_measure = Dict(1=>Expectation(), 2=>WorstCase())
# risk_measure = (node_index) -> node_index == 1 ? Expectation() : WorstCase()
# It will return a dictionary with a key for each node_index in the policy
# graph, and a corresponding value of whatever the user provided.
function to_nodal_form(model::PolicyGraph{T}, element) where {T}
    # Note: we don't copy element here, so if element is mutable, you should use
    # to_nodal_form(model, x -> new_element()) instead. A good example is
    # Vector{T}; use to_nodal_form(model, i -> T[]).
    store = Dict{T,typeof(element)}()
    for node_index in keys(model.nodes)
        store[node_index] = element
    end
    return store
end

function to_nodal_form(model::PolicyGraph{T}, builder::Function) where {T}
    store = Dict{T,Any}()
    for node_index in keys(model.nodes)
        store[node_index] = builder(node_index)
    end
    V = typeof(first(values(store)))
    for val in values(store)
        V = promote_type(V, typeof(val))
    end
    return Dict{T,V}(key => val for (key, val) in store)
end

function to_nodal_form(model::PolicyGraph{T}, dict::Dict{T,V}) where {T,V}
    for key in keys(model.nodes)
        if !haskey(dict, key)
            error("Missing key: $(key).")
        end
    end
    return dict
end

# Internal struct: storage for RVSDDP options and cached data. Users shouldn't
# interact with this directly.
struct Options{T}
    # The initial state to start from the root node.
    initial_state::Dict{Symbol,Float64}
    # The sampling scheme to use on the forward pass.
    sampling_scheme::AbstractSamplingScheme
    backward_sampling_scheme::AbstractBackwardSamplingScheme
    # Risk measure to use at each node.
    risk_measures::Dict{T,AbstractRiskMeasure}
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
        model::PolicyGraph{T},
        initial_state::Dict{Symbol,Float64};
        sampling_scheme::AbstractSamplingScheme = InSampleMonteCarlo(),
        backward_sampling_scheme::AbstractBackwardSamplingScheme = CompleteSampler(),
        risk_measures = Expectation(),
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
    ) where {T}
        return new{T}(
            initial_state,
            sampling_scheme,
            backward_sampling_scheme,
            to_nodal_form(model, risk_measures),
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
