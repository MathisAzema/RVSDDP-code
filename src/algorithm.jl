#  Copyright (c) 2017-25, Oscar Dowson and RVSDDP.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

macro _timeit_threadsafe(timer, label, block)
    code = quote
        # TimerOutputs is not thread-safe, so run it only if there is a single
        # thread.
        if Threads.nthreads() == 1
            TimerOutputs.@timeit $timer $label $block
        else
            $block
        end
    end
    return esc(code)
end

"""
    _node(model::PolicyGraph{T}, index::T, replica::Int) where {T}

The copy of node `index` that worker `replica` of a `parallel > 1` batch owns.

Worker `1` works on the node itself; worker `r > 1` works on `replicas[r-1]`,
an independent JuMP model carrying exactly the same cuts (see
`_build_replicas!`). Two workers therefore never touch the same subproblem.
"""
function _node(model::PolicyGraph{T}, index::T, replica::Int) where {T}
    node = model[index]
    return replica == 1 ? node : node.replicas[replica-1]
end

"""
    _parallel_foreach(f::Function, n::Int)

Run `f(1), ..., f(n)`, concurrently when there is something to gain.

This is how the `options.parallel` trajectories of a batch are actually spread
over cores: one Julia task per trajectory, each working on its own replica of
every subproblem. Start Julia with `julia -t n` (or `JULIA_NUM_THREADS=n`) to
give those tasks `n` cores; with a single thread this degrades to a plain loop.
"""
function _parallel_foreach(f::Function, n::Int)
    if n <= 1 || Threads.nthreads() == 1
        for i in 1:n
            f(i)
        end
        return
    end
    @sync for i in 1:n
        Threads.@spawn f(i)
    end
    return
end

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

# Internal function: returns a dictionary with a key for each node, where the
# value is a list of other nodes that contain the same children. This is useful
# because on the backward pass we can add cuts to nodes with the same children
# without having to re-solve the children.
function get_same_children(model::PolicyGraph{T}) where {T}
    tmp = Dict{Set{T},Set{T}}()
    for (key, node) in model.nodes
        children = Set(child.term for child in node.children)
        if length(children) == 0
            continue
        elseif haskey(tmp, children)
            push!(tmp[children], key)
        else
            tmp[children] = Set{T}([key])
        end
    end
    same_children = Dict{T,Vector{T}}(key => T[] for key in keys(model.nodes))
    for set in values(tmp)
        for v in set
            same_children[v] = collect(setdiff(set, Ref(v)))
        end
    end
    return same_children
end

# Internal struct: storage for RVSDDP options and cached data. Users shouldn't
# interact with this directly.
struct Options{T}
    # The initial state to start from the root node.
    initial_state::Dict{Symbol,Float64}
    # The sampling scheme to use on the forward pass.
    sampling_scheme::AbstractSamplingScheme
    backward_sampling_scheme::AbstractBackwardSamplingScheme
    # Storage for the set of possible sampling states at each node. We only use
    # this if there is a cycle in the policy graph.
    starting_states::Dict{T,Vector{Dict{Symbol,Float64}}}
    # Risk measure to use at each node.
    risk_measures::Dict{T,AbstractRiskMeasure}
    # The delta by which to check if a state is close to a previously sampled
    # state.
    cycle_discretization_delta::Float64
    # Flag to add cuts to similar nodes.
    refine_at_similar_nodes::Bool
    # A list of nodes that contain a subset of the children of node i.
    similar_children::Dict{T,Vector{T}}
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
    root_node_risk_measure::AbstractRiskMeasure
    #Mathis
    infinite::Bool
    shift_function::Function
    parallel::Int64
    start::Float64
    refine_mode::Int64
    # Internal function: users should never construct this themselves.
    function Options(
        model::PolicyGraph{T},
        initial_state::Dict{Symbol,Float64};
        sampling_scheme::AbstractSamplingScheme = InSampleMonteCarlo(),
        backward_sampling_scheme::AbstractBackwardSamplingScheme = CompleteSampler(),
        risk_measures = Expectation(),
        cycle_discretization_delta::Float64 = 0.0,
        refine_at_similar_nodes::Bool = true,
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
        root_node_risk_measure::AbstractRiskMeasure = Expectation(),
        infinite::Bool = false,
        shift_function::Function = RVSDDP.no_shift,
        parallel::Int64 = 1,
        refine_mode::Int64 = 0,
    ) where {T}
        return new{T}(
            initial_state,
            sampling_scheme,
            backward_sampling_scheme,
            to_nodal_form(model, x -> Dict{Symbol,Float64}[]),
            to_nodal_form(model, risk_measures),
            cycle_discretization_delta,
            refine_at_similar_nodes,
            get_same_children(model),
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
            root_node_risk_measure,
            infinite,
            shift_function,
            parallel,
            time(),
            refine_mode,
        )
    end
end

# Internal function: set the incoming state variables of node to the values
# contained in state.
function set_incoming_state(node::Node, state::Dict{Symbol,Float64})
    for (state_name, value) in state
        JuMP.fix(node.states[state_name].in, value)
    end
    return
end

# Internal function: get the values of the outgoing state variables in node.
# Requires node.subproblem to have been solved with PrimalStatus ==
# FeasiblePoint.
function get_outgoing_state(node::Node)
    values = Dict{Symbol,Float64}()
    for (name, state) in node.states
        # To fix some cases of numerical infeasiblities, if the outgoing value
        # is outside its bounds, project the value back onto the bounds. There
        # is a pretty large (×5) penalty associated with this check because it
        # typically requires a call to the solver. It is worth reducing
        # infeasibilities though.
        outgoing_value = JuMP.value(state.out)
        if JuMP.has_upper_bound(state.out)
            current_bound = JuMP.upper_bound(state.out)
            if current_bound < outgoing_value
                outgoing_value = current_bound
            end
        end
        if JuMP.has_lower_bound(state.out)
            current_bound = JuMP.lower_bound(state.out)
            if current_bound > outgoing_value
                outgoing_value = current_bound
            end
        end
        values[name] = outgoing_value
    end
    return values
end

# Internal function: set the objective of node to the stage objective, plus the
# cost/value-to-go term.
function set_objective(node::Node{T}) where {T}
    if !node.stage_objective_set
        JuMP.set_objective(
            node.subproblem,
            JuMP.objective_sense(node.subproblem),
            @expression(
                node.subproblem,
                node.stage_objective +
                node.discount_factor*bellman_term(node.bellman_function)
            )
        )
    end
    node.stage_objective_set = true
    return
end

# Internal function: overload for the case where JuMP.value fails on a
# Real number.
_value(x::Real) = x
_value(x) = JuMP.value(x)

stage_objective_value(::Node, x::Union{GenericVariableRef,Real}) = _value(x)

function stage_objective_value(node::Node, stage_objective)
    theta = bellman_term(node.bellman_function)
    return JuMP.objective_value(node.subproblem) - node.discount_factor*JuMP.value(theta)
end

"""
    write_subproblem_to_file(
        node::Node,
        filename::String;
        throw_error::Bool = false,
    )

Write the subproblem contained in `node` to the file `filename`.

The `throw_error` is an argument used internally by RVSDDP.jl. If set, an error
will be thrown.

## Example

```julia
RVSDDP.write_subproblem_to_file(model[1], "subproblem_1.lp")
```
"""
function write_subproblem_to_file(
    node::Node,
    filename::String;
    throw_error::Bool = false,
)
    model = MOI.FileFormats.Model(; filename = filename)
    MOI.copy_to(model, JuMP.backend(node.subproblem))
    MOI.write_to_file(model, filename)
    if throw_error
        error(
            "Unable to retrieve solution from node $(node.index).\n\n",
            "  Termination status : $(JuMP.termination_status(node.subproblem))\n",
            "  Primal status      : $(JuMP.primal_status(node.subproblem))\n",
            "  Dual status        : $(JuMP.dual_status(node.subproblem)).\n\n",
            "The current subproblem was written to `$(filename)`.\n\n",
            "There are two common causes of this error:\n",
            "  1) you have a mistake in your formulation, or you violated\n",
            "     the assumption of relatively complete recourse\n",
            "  2) the solver encountered numerical issues\n\n",
            "See https://odow.github.io/RVSDDP.jl/stable/tutorial/warnings/ for more information.",
        )
    end
    return
end

"""
    parameterize(node::Node, noise)

Parameterize node `node` with the noise `noise`.
"""
function parameterize(node::Node, noise)
    node.parameterize(noise)
    set_objective(node)
    return
end

function _has_primal_solution(node::Node)
    status = JuMP.primal_status(node.subproblem)
    return status in (JuMP.FEASIBLE_POINT, JuMP.NEARLY_FEASIBLE_POINT)
end

function _has_dual_solution(node::Node)
    status = JuMP.dual_status(node.subproblem)
    return status in (JuMP.FEASIBLE_POINT, JuMP.NEARLY_FEASIBLE_POINT)
end

"""
    set_numerical_difficulty_callback(
        model::PolicyGraph,
        callback::Function,
    )

Set a callback function `callback(::PolicyGraph, ::Node; require_dual::Bool)`
that is run when the optimizer terminates without finding a primal solution (and
dual solution if `require_dual` is `true`).

## Default callback

The default callback is a small variation of:
```julia
function callback(::PolicyGraph, node::Node; require_dual::Bool)
    MOI.Utilities.reset_optimizer(node.subproblem)
    optimize!(node.subproblem)
    return
end
```
This callback is the default because a common issue is solvers declaring the
infeasible because of numerical issues related to the large number of cutting
planes. Resetting the subproblem---and therefore starting from a fresh problem
instead of warm-starting from the previous solution---is often enough to fix the
problem and allow more iterations.

## Other callbacks

In cases where the problem is truely infeasible (not because of numerical issues
), it may be helpful to write out the irreducible infeasible subsystem (IIS) for
debugging. For this use-case, use a callback as follows:
```julia
function callback(::PolicyGraph, node::Node; require_dual::Bool)
    JuMP.compute_conflict!(node.suprobblem)
    status = JuMP.get_attribute(node.subproblem, MOI.ConflictStatus())
    if status == MOI.CONFLICT_FOUND
        iis_model, _ = JuMP.copy_conflict(node.subproblem)
        print(iis_model)
    end
    return
end
RVSDDP.set_numerical_difficulty_callback(model, callback)
```
"""
function set_numerical_difficulty_callback(
    model::PolicyGraph,
    callback::Function,
)
    model.ext[:numerical_difficulty_callback] = callback
    return
end

function attempt_numerical_recovery(
    model::PolicyGraph,
    node::Node;
    require_dual::Bool = false,
)
    model.ext[:numerical_issue] = true
    callback = get(
        model.ext,
        :numerical_difficulty_callback,
        default_numerical_difficulty_callback,
    )
    callback(model, node; require_dual)
    missing_dual_solution = require_dual && !_has_dual_solution(node)
    if !_has_primal_solution(node) || missing_dual_solution
        # We use the `node.index` in the filename because two threads could
        # both hit an infeasibility at the same time.
        write_subproblem_to_file(
            node,
            "subproblem_$(node.index).mof.json";
            throw_error = true,
        )
    end
    return
end

function default_numerical_difficulty_callback(
    model::PolicyGraph,
    node::Node;
    kwargs...,
)
    if JuMP.mode(node.subproblem) == JuMP.DIRECT
        @warn(
            "Unable to recover in direct mode! Remove `direct = true` when " *
            "creating the policy graph."
        )
        return
    end
    MOI.Utilities.reset_optimizer(node.subproblem)
    optimize!(node.subproblem)
    return
end

"""
    _initialize_solver(node::Node; throw_error::Bool)

After passing a model to a different process, we need to set the optimizer
again.

If `throw_error`, throw an error if the model is in direct mode.

See also: [`_uninitialize_solver`](@ref).
"""
function _initialize_solver(node::Node; throw_error::Bool)
    if mode(node.subproblem) == DIRECT
        if throw_error
            error(
                "Cannot use asynchronous solver with optimizers in direct mode.",
            )
        end
    elseif MOI.Utilities.state(backend(node.subproblem)) == MOIU.NO_OPTIMIZER
        if node.optimizer === nothing
            error(
                """
          You must supply an optimizer for the policy graph, either by passing
          one to the `optimizer` keyword argument to `PolicyGraph`, or by
          using `JuMP.set_optimizer(model, optimizer)`.
          """,
            )
        end
        set_optimizer(node.subproblem, node.optimizer)
        set_silent(node.subproblem)
    end
    return
end

"""
    _initialize_solver(model::PolicyGraph; throw_error::Bool)

After passing a model to a different process, we need to set the optimizer
again.

If `throw_error`, throw an error if the model is in direct mode.

See also: [`_uninitialize_solver`](@ref).
"""
function _initialize_solver(model::PolicyGraph; throw_error::Bool)
    for (_, node) in model.nodes
        _initialize_solver(node; throw_error = throw_error)
    end
    return
end

"""
    _uninitialize_solver(model; throw_error::Bool)

Before passing a model to a different process, we need to drop the inner solver
in case it has some C pointers that we cannot serialize (e.g., HiGHS).

If `throw_error`, throw an error if the model is in direct mode.

See also: [`_initialize_solver`](@ref).
"""
function _uninitialize_solver(model::PolicyGraph; throw_error::Bool)
    for (_, node) in model.nodes
        if mode(node.subproblem) == DIRECT
            if throw_error
                error(
                    "Cannot use asynchronous solver with optimizers in direct mode.",
                )
            end
        elseif MOI.Utilities.state(backend(node.subproblem)) !=
               MOIU.NO_OPTIMIZER
            MOI.Utilities.drop_optimizer(node.subproblem)
        end
    end
    return
end

# Internal function: solve the subproblem associated with node given the
# incoming state variables state and realization of the stagewise-independent
# noise term noise.
function solve_subproblem(
    model::PolicyGraph{T},
    node::Node{T},
    state::Dict{Symbol,Float64},
    noise;
    duality_handler::Union{Nothing,AbstractDualityHandler},
) where {T}
    _initialize_solver(node; throw_error = false)
    # Parameterize the model. First, fix the value of the incoming state
    # variables. Then parameterize the model depending on `noise`. Finally,
    # set the objective.
    set_incoming_state(node, state)
    parameterize(node, noise)
    JuMP.optimize!(node.subproblem)
    lock(model.lock) do
        model.ext[:total_solves] = get(model.ext, :total_solves, 0) + 1
        return
    end
    if JuMP.primal_status(node.subproblem) == JuMP.MOI.INTERRUPTED
        # If the solver was interrupted, the user probably hit CTRL+C but the
        # solver gracefully exited. Since we're in the middle of training or
        # simulation, we need to throw an interrupt exception to keep the
        # interrupt percolating up to the user.
        throw(InterruptException())
    end
    if !_has_primal_solution(node)
        attempt_numerical_recovery(model, node)
    end
    state = get_outgoing_state(node)
    stage_objective = stage_objective_value(node, node.stage_objective)
    @_timeit_threadsafe model.timer_output "get_dual_solution" begin
        objective, dual_values = get_dual_solution(node, duality_handler)
    end
    return (
        state = state,
        duals = dual_values,
        objective = objective,
        stage_objective = stage_objective,
    )
end




# Internal function: calculate the minimum distance between the state `state`
# and the list of states in `starting_states` using the distance measure `norm`.
function distance(
    starting_states::Vector{Dict{Symbol,Float64}},
    state::Dict{Symbol,Float64},
    norm::Function = inf_norm,
)
    if length(starting_states) == 0
        return Inf
    end
    return minimum(norm.(starting_states, Ref(state)); init = Inf)
end

# Internal function: the norm to use when checking the distance between two
# possible starting states. We're going to use: d(x, y) = |x - y| / (1 + |y|).
function inf_norm(x::Dict{Symbol,Float64}, y::Dict{Symbol,Float64})
    norm = 0.0
    for (key, value) in y
        if abs(x[key] - value) > norm
            norm = abs(x[key] - value) / (1 + abs(value))
        end
    end
    return norm
end


mutable struct Trajectory{T}
    scenario_path::Vector{Tuple{T, Any}}
    sampled_states::Vector{Dict{Symbol,Float64}}
    cumulative_value::Float64
end

function _update_delta(
    node::Node{T}, 
    incoming_state::Dict{Symbol,Float64}, 
    risk_adjusted_probability::Vector{Float64}, 
    objective_realizations::Vector{Float64},
) where {T}
    TVᵏ = 0.0
    for i in 1:length(objective_realizations)
        p = risk_adjusted_probability[i]
        TVᵏ += p * objective_realizations[i]
    end
    Vᵏ=compute_V(node.value_function, incoming_state)
    deltaᵏ = TVᵏ - Vᵏ
    push!(node.delta, deltaᵏ)
end

function _refine_at_initial_point(
    model::PolicyGraph{T},
    options::Options,

) where {T}
    if options.infinite
        node_index = length(model.nodes)
        node =  model[node_index]
        next_node = model[node.children[1].term]
        items = BackwardPassItems(T, Noise)
        outgoing_state = model.initial_root_state

        solve_all_children(
            model,
            node,
            items,
            outgoing_state,
            options.backward_sampling_scheme,
            options.duality_handler,
            options,
        )
        shift=options.shift_function(model, next_node, [items], [outgoing_state])

        _update_delta(next_node, outgoing_state, items.probability, items.objectives)

        new_cuts = refine_bellman_function(
            model,
            node,
            node.bellman_function,
            options.risk_measures[node_index],
            outgoing_state,
            items.duals,
            items.supports,
            items.probability,
            items.objectives,
            shift,
            length(options.log)+1,
            time()-options.start,
        )

    else
        node_index = 1
        node =  model[node_index]
        items = BackwardPassItems(T, Noise)
        incoming_state = model.initial_root_state

        solve_one_children(
            model,
            node,
            items,
            incoming_state,
            options.backward_sampling_scheme,
            options.duality_handler,
            options,
        )
        shift=options.shift_function(model, node, [items], [incoming_state])

        πᵏ = Dict(key => 0.0 for key in keys(incoming_state))
        θᵏ = 0.0
        objective_realizations = items.objectives
        risk_adjusted_probability = items.probability
        dual_variables = items.duals
        for i in 1:length(objective_realizations)
            p = risk_adjusted_probability[i]
            θᵏ += p * objective_realizations[i]
            for (key, dual) in dual_variables[i]
                πᵏ[key] += p * dual
            end
        end

        for (key, x) in incoming_state
            θᵏ -= πᵏ[key] * x
        end

        iteration = length(options.log)+1

        cut = Cut(iteration, time() - options.start, θᵏ, πᵏ, incoming_state)

        _update_value_function(node, cut, shift, nothing)
        _update_delta(node, incoming_state, items.probability, items.objectives)
        return []
    end
    # println(new_cuts)

    return new_cuts
end

# Internal function: perform a backward pass of the RVSDDP algorithm along the
# scenario_path, refining the bellman function at sampled_states. Assumes that
# scenario_path does not end in a leaf node (i.e., the forward pass was solved
# with include_last_node = false)
function backward_pass(
    model::PolicyGraph{T},
    options::Options,
    trajectory::Vector{Trajectory{T}},
) where {T}
    scenario_length = length(trajectory[1].scenario_path)
    period= length(model.nodes)

    if options.refine_mode == 0
        index_to_refine = 0:scenario_length-1
    else
        index_rand = rand(1:Int(round(scenario_length/period)))
        index_to_refine = (index_rand-1)*period:index_rand*period-1
    end
    # TODO(odow): improve storage type.
    cuts = Dict{T,Vector{Any}}(index => Any[] for index in keys(model.nodes))
    for index in scenario_length:-1:1
        node_index, _ = trajectory[1].scenario_path[index]
        node =  model[node_index]
        if length(node.children) == 0
            continue
        end
        if index in index_to_refine
            items_traj = [BackwardPassItems(T, Noise) for _ in trajectory]
            outgoing_states = [traj.sampled_states[index] for traj in trajectory]
            # Solve the children for every trajectory of the batch in parallel:
            # trajectory `j` solves replica `j` of each child and fills its own
            # `items_traj[j]`, so the tasks share nothing. They are all joined
            # before the common shift below is computed.
            _parallel_foreach(length(trajectory)) do index_traj
                solve_all_children(
                    model,
                    node,
                    items_traj[index_traj],
                    outgoing_states[index_traj],
                    options.backward_sampling_scheme,
                    options.duality_handler,
                    options;
                    replica = index_traj,
                )
            end

            # From here on we are back on a single task: the shift is computed
            # once from the whole batch, and the cuts it produces mutate the
            # master model (and, through `_add_cut_constraint_to_model`, every
            # replica), so they have to be added one at a time.
            next_node = model[node.children[1].term]
            shift=options.shift_function(model, next_node, items_traj, outgoing_states)
            if index <= length(model.nodes)-1 || options.refine_mode == 1
                outgoing_state = outgoing_states[1]
                items = items_traj[1]
                _update_delta(next_node, outgoing_state, items.probability, items.objectives)
            end
            for (index_traj, traj) in enumerate(trajectory)
                outgoing_state = outgoing_states[index_traj]
                items = items_traj[index_traj]
                new_cuts = refine_bellman_function(
                    model,
                    node,
                    node.bellman_function,
                    options.risk_measures[node_index],
                    outgoing_state,
                    items.duals,
                    items.supports,
                    items.probability,
                    items.objectives,
                    shift,
                    length(options.log)+1,
                    time()-options.start,
                )

                push!(cuts[node_index], new_cuts)
            end
        end
    end
    if 0 in index_to_refine
        new_cuts_0 = _refine_at_initial_point(model, options)
        push!(cuts[length(model.nodes)], new_cuts_0)
    end
    push!(model.approx_value, (time()-options.start, compute_approx_value(model)))
    return cuts
end

struct BackwardPassItems{T,U}
    "Given a (node, noise) tuple, index the element in the array."
    cached_solutions::Dict{Tuple{T,Any},Int}
    duals::Vector{Dict{Symbol,Float64}}
    supports::Vector{U}
    nodes::Vector{T}
    probability::Vector{Float64}
    objectives::Vector{Float64}
    function BackwardPassItems(T, U)
        return new{T,U}(
            Dict{Tuple{T,Any},Int}(),
            Dict{Symbol,Float64}[],
            U[],
            T[],
            Float64[],
            Float64[],
        )
    end
end

function solve_one_children(
    model::PolicyGraph{T},
    node::Node{T},
    items::BackwardPassItems,
    incoming_state::Dict{Symbol,Float64},
    backward_sampling_scheme::AbstractBackwardSamplingScheme,
    duality_handler::Union{Nothing,AbstractDualityHandler},
    options,
) where {T}
    lock(node.lock)
    try
        @_timeit_threadsafe model.timer_output "prepare_backward_pass" begin
            restore_duality = prepare_backward_pass(
                node,
                options.duality_handler,
                options,
            )
        end
        for noise in sample_backward_noise_terms_with_state(
            backward_sampling_scheme,
            node,
            incoming_state,
        )
            if haskey(items.cached_solutions, (node.index, noise.term))
                sol_index = items.cached_solutions[(node.index, noise.term)]
                push!(items.duals, items.duals[sol_index])
                push!(items.supports, items.supports[sol_index])
                push!(items.nodes, node.index)
                push!(items.probability, items.probability[sol_index])
                push!(items.objectives, items.objectives[sol_index])
            else
                @_timeit_threadsafe model.timer_output "solve_subproblem" begin
                    subproblem_results = solve_subproblem(
                        model,
                        node,
                        incoming_state,
                        noise.term;
                        duality_handler = duality_handler,
                    )
                end
                push!(items.duals, subproblem_results.duals)
                push!(items.supports, noise)
                push!(items.nodes, node.index)
                push!(
                    items.probability,
                    noise.probability,
                )
                push!(items.objectives, subproblem_results.objective)
                items.cached_solutions[(node.index, noise.term)] =
                    length(items.duals)
            end
        end
        @_timeit_threadsafe model.timer_output "prepare_backward_pass" begin
            restore_duality()
        end
    finally
        unlock(node.lock)
    end
    return
end

"""
    solve_all_children(model, node, items, outgoing_state, ...; replica)

Solve every child of `node` at `outgoing_state` and accumulate the results in
`items`.

`replica` selects which copy of each child subproblem to solve, so that the
`options.parallel` calls made at a given stage of the backward pass can run
concurrently without sharing a JuMP model. See `_node`.
"""
function solve_all_children(
    model::PolicyGraph{T},
    node::Node{T},
    items::BackwardPassItems,
    outgoing_state::Dict{Symbol,Float64},
    backward_sampling_scheme::AbstractBackwardSamplingScheme,
    duality_handler::Union{Nothing,AbstractDualityHandler},
    options;
    replica::Int = 1,
) where {T}
    for child in node.children
        child_node = _node(model, child.term, replica)
        solve_one_children(
            model,
            child_node,
            items,
            outgoing_state,
            backward_sampling_scheme,
            duality_handler,
            options,
        )
    end
    return
end

"""
    RVSDDP.calculate_bound(
        model::PolicyGraph,
        state::Dict{Symbol,Float64} = model.initial_root_state;
        risk_measure::AbstractRiskMeasure = Expectation(),
    )

Calculate the lower bound (if minimizing, otherwise upper bound) of the problem
model at the point state, assuming the risk measure at the root node is
risk_measure.
"""
function calculate_bound(
    model::PolicyGraph{T},
    root_state::Dict{Symbol,Float64} = model.initial_root_state;
    risk_measure::AbstractRiskMeasure = Expectation(),
) where {T}
    # Initialization.
    noise_supports = Any[]
    probabilities = Float64[]
    objectives = Float64[]
    # Solve all problems that are children of the root node.
    for child in model.root_children
        # It's okay to skip nodes with zero probability.
        #
        # See RVSDDP.jl#796 and RVSDDP.jl#797 for more discussion.
        if isapprox(child.probability, 0.0; atol = 1e-6)
            continue
        end
        node = model[child.term]
        lock(node.lock)
        try
            for noise in node.noise_terms
                subproblem_results = solve_subproblem(
                    model,
                    node,
                    root_state,
                    noise.term;
                    duality_handler = nothing,
                )
                push!(objectives, subproblem_results.objective)
                push!(probabilities, child.probability * noise.probability)
                push!(noise_supports, noise.term)
            end
        finally
            unlock(node.lock)
        end
    end
    # Now compute the risk-adjusted probability measure:
    risk_adjusted_probability = similar(probabilities)
    offset = adjust_probability(
        risk_measure,
        risk_adjusted_probability,
        probabilities,
        noise_supports,
        objectives,
        model.objective_sense == MOI.MIN_SENSE,
    )
    # Finally, calculate the risk-adjusted value.
    return sum(
        obj * prob for (obj, prob) in zip(objectives, risk_adjusted_probability)
    ) + offset
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
    @_timeit_threadsafe model.timer_output "calculate_bound" begin
        bound = calculate_bound(
            model;
            risk_measure = options.root_node_risk_measure,
        )
    end
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

# Internal: run `number_replications` independent simulations of the policy.
function _simulate(
    model::PolicyGraph,
    number_replications::Int,
    variables::Vector{Symbol};
    kwargs...,
)
    _initialize_solver(model; throw_error = false)
    return map(_ -> _simulate(model, variables; kwargs...), 1:number_replications)
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

 - `refine_at_similar_nodes::Bool`: if RVSDDP can detect that two nodes have the
    same children, it can cheaply add a cut discovered at one to the other. In
    almost all cases this should be set to `true`.

 - `risk_measure`: the risk measure to use at each node. Defaults to
   [`Expectation`](@ref).

 -  `root_node_risk_measure::AbstractRiskMeasure`: the risk measure to use at
    the root node when computing the `Bound` column. Note that the choice of
    this option does not change the primal policy, and it applies only if the
    transition from the root node to the first stage is stochastic. Defaults to
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

There is also a special option for infinite horizon problems

 - `cycle_discretization_delta`: the maximum distance between states allowed on
    the forward pass. This is for advanced users only and needs to be used in
    conjunction with a different `sampling_scheme`.
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
    root_node_risk_measure::AbstractRiskMeasure = Expectation(),
    sampling_scheme = RVSDDP.InSampleMonteCarlo(),
    cycle_discretization_delta::Float64 = 0.0,
    refine_at_similar_nodes::Bool = true,
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
    refine_mode::Int64=0,
)
    #Mathis
    # if infinite
    #     sampling_scheme = RVSDDP.InSampleMonteCarlo(max_depth=5*length(keys(model.nodes)))
    # end
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
        cycle_discretization_delta,
        refine_at_similar_nodes,
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
        root_node_risk_measure,
        infinite,
        shift_function,
        parallel,
        refine_mode,
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

# Internal function: helper to conduct a single simulation. Users should use the
# documented, user-facing function RVSDDP.simulate instead.
function _simulate(
    model::PolicyGraph{T},
    variables::Vector{Symbol};
    infinite::Bool=true,
    sampling_scheme::AbstractSamplingScheme,
    custom_recorders::Dict{Symbol,Function},
    duality_handler::Union{Nothing,AbstractDualityHandler},
    skip_undefined_variables::Bool,
    incoming_state::Dict{Symbol,Float64},
) where {T}
    # Sample a scenario path.
    scenario_path, _ = sample_scenario(model, sampling_scheme)

    # Storage for the simulation results.
    simulation = Dict{Symbol,Any}[]
    # A cumulator for the stage-objectives.
    cumulative_value = 0.0
    length_scenario_path = length(scenario_path)
    for (depth, (node_index, noise)) in enumerate(scenario_path)
        node = model[node_index]
        lock(node.lock)
        try
            # Solve the subproblem.
            subproblem_results = solve_subproblem(
                model,
                node,
                incoming_state,
                noise;
                duality_handler = duality_handler,
            )
            # Add the stage-objective
            cumulative_value += subproblem_results.stage_objective
            # Record useful variables from the solve.
            store = Dict{Symbol,Any}(
                :node_index => node_index,
                :noise_term => noise,
                :stage_objective => subproblem_results.stage_objective,
                :bellman_term =>
                    subproblem_results.objective -
                    subproblem_results.stage_objective,
                :outgoing_state => subproblem_results.state,
            )
            if depth == length_scenario_path && infinite
                store[:cost_end_of_horizon] = compute_cost_end_of_horizon(model, length_scenario_path+1, subproblem_results.state)
            end
            # Loop through the primal variable values that the user wants.
            for variable in variables
                if haskey(node.subproblem.obj_dict, variable)
                    # Note: we broadcast the call to value for variables which are
                    # containers (like Array, Containers.DenseAxisArray, etc). If
                    # the variable is a scalar (e.g. just a plain VariableRef), the
                    # broadcast preseves the scalar shape.
                    # TODO: what if the variable container is a dictionary? They
                    # should be using Containers.SparseAxisArray, but this might not
                    # always be the case...
                    # println("variable: $(variable)")
                    # println(JuMP.value.(node.subproblem[variable]))
                    store[variable] = JuMP.value.(node.subproblem[variable])
                elseif skip_undefined_variables
                    store[variable] = NaN
                else
                    error(
                        "No variable named $(variable) exists in the subproblem.",
                        " If you want to simulate the value of a variable, make ",
                        "sure it is defined in _all_ subproblems, or pass ",
                        "`skip_undefined_variables=true` to `simulate`.",
                    )
                end
            end
            # Loop through any custom recorders that the user provided.
            for (sym, recorder) in custom_recorders
                store[sym] = recorder(node.subproblem)
            end
            # Add the store to our list.
            push!(simulation, store)
            # Set outgoing state as the incoming state for the next node.
            incoming_state = copy(subproblem_results.state)
        finally
            unlock(node.lock)
        end
    end
    return simulation
end

function _initial_state(model::PolicyGraph)
    return Dict(String(k) => v for (k, v) in model.initial_root_state)
end

"""
    simulate(
        model::PolicyGraph,
        number_replications::Int = 1,
        variables::Vector{Symbol} = Symbol[];
        sampling_scheme::AbstractSamplingScheme =
            InSampleMonteCarlo(),
        custom_recorders = Dict{Symbol, Function}(),
        duality_handler::Union{Nothing,AbstractDualityHandler} = nothing,
        skip_undefined_variables::Bool = false,
        incoming_state::Dict{String,Float64} = _initial_state(model),
     )::Vector{Vector{Dict{Symbol,Any}}}

Perform a simulation of the policy model with `number_replications` replications.

## Return data structure

Returns a vector with one element for each replication. Each element is a vector
with one-element for each node in the scenario that was sampled. Each element in
that vector is a dictionary containing information about the subproblem that was
solved.

In that dictionary there are four special keys:

 - `:node_index`, which records the index of the sampled node in the policy model

 - `:noise_term`, which records the noise observed at the node

 - `:stage_objective`, which records the stage-objective of the subproblem

 - `:bellman_term`, which records the cost/value-to-go of the node.

The sum of `:stage_objective + :bellman_term` will equal the objective value of
the solved subproblem.

In addition to the special keys, the dictionary will contain the result of
`key => JuMP.value(subproblem[key])` for each `key` in `variables`. This is
useful to obtain the primal value of the state and control variables.

## Positonal arguments

 - `model`: the model to simulate

 - `number_replications::Int = 1`: the number of simulation replications to
   conduct, that is, the length of the simulation vector that is returned by
   this function. If omitted, this defaults to `1`.`

 - `variables::Vector{Symbol} = Symbol[]`: a list of the variable names to
   record the value of in each stage.

## Keyword arguments

 - `sampling_scheme`: the sampling scheme used when simulating.

 - `custom_recorders`: see `Custom recorders` section below.

 - `duality_handler`: the [`RVSDDP.AbstractDualityHandler`](@ref) used to compute
   dual variables. If you do not require dual variables (or if they are not
   available), pass `duality_handler = nothing`.

 - `skip_undefined_variables`: If you attempt to simulate the value of a
   variable that is only defined in some of the stage problems, an error will be
   thrown. To over-ride this (and return a `NaN` instead), pass
   `skip_undefined_variables = true`.

 - `initial_state`: Use `incoming_state` to pass an initial value of the state
   variable, if it differs from that at the root node. Each key should be the
   string name of the state variable.

## Custom recorders

For more complicated data, the `custom_recorders` keyword argument can be used.

For example, to record the dual of a constraint named `my_constraint`, pass the
following:

```julia
simulation_results = RVSDDP.simulate(model, 2;
    custom_recorders = Dict{Symbol, Function}(
        :constraint_dual => sp -> JuMP.dual(sp[:my_constraint])
    )
)
```
The value of the dual in the first stage of the second replication can be
accessed as:
```julia
simulation_results[2][1][:constraint_dual]
```
"""
function simulate(
    model::PolicyGraph,
    number_replications::Int = 1,
    variables::Vector{Symbol} = Symbol[];
    infinite::Bool=true,
    sampling_scheme::AbstractSamplingScheme = InSampleMonteCarlo(),
    custom_recorders = Dict{Symbol,Function}(),
    duality_handler::Union{Nothing,AbstractDualityHandler} = nothing,
    skip_undefined_variables::Bool = false,
    incoming_state::Dict{String,Float64} = _initial_state(model),
)
    return _simulate(
        model,
        number_replications,
        variables;
        infinite=infinite,
        sampling_scheme = sampling_scheme,
        custom_recorders = custom_recorders,
        duality_handler = duality_handler,
        skip_undefined_variables = skip_undefined_variables,
        incoming_state = Dict(Symbol(k) => v for (k, v) in incoming_state),
    )
end

