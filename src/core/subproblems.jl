#  Copyright (c) 2017-25, Oscar Dowson and RVSDDP.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Everything that happens inside a single node's JuMP model: moving a state in
# and out of it, setting its objective, fixing the noise, solving it, and
# recovering from a solver that fails to do so. `solve_subproblem` is the one
# entry point the forward and backward passes use.

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
