#  Copyright (c) 2017-25, Oscar Dowson and RVSDDP.jl contributors, Lea Kapelevich.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

function get_dual_solution(node::Node, ::Nothing)
    return JuMP.objective_value(node.subproblem), Dict{Symbol,Float64}()
end

# ========================= Continuous relaxation ============================ #

"""
    ContinuousConicDuality(optimizer = nothing)

Get the duals of the backward pass from conic duality, relaxing any integrality
first. `optimizer`, if given, is set on the subproblem before those solves; it is
only needed when the default optimizer cannot return duals after relaxation.
"""
struct ContinuousConicDuality{O} <: AbstractDualityHandler
    optimizer::O

    function ContinuousConicDuality(optimizer = nothing)
        return new{typeof(optimizer)}(optimizer)
    end
end

function get_dual_solution(node::Node, ::ContinuousConicDuality)
    if !_has_dual_solution(node)
        model = node.subproblem.ext[:RVSDDP_policy_graph]
        attempt_numerical_recovery(model, node; require_dual = true)
    end
    # Note: due to JuMP's dual convention, we need to flip the sign for
    # maximization problems.
    dual_sign = JuMP.objective_sense(node.subproblem) == MOI.MIN_SENSE ? 1 : -1
    λ = Dict{Symbol,Float64}(
        name => dual_sign * JuMP.dual(JuMP.FixRef(state.in)) for
        (name, state) in node.states
    )
    return objective_value(node.subproblem), λ
end

function _relax_integrality(node::Node, optimizer)
    if !node.has_integrality
        return () -> nothing
    elseif optimizer === nothing
        return JuMP.relax_integrality(node.subproblem)
    end
    undo_relax = JuMP.relax_integrality(node.subproblem)
    JuMP.set_optimizer(node.subproblem, optimizer)
    return () -> begin
        JuMP.set_optimizer(node.subproblem, node.optimizer)
        undo_relax()
        return
    end
end

function prepare_backward_pass(
    node::Node,
    handler::ContinuousConicDuality,
    ::Options,
)
    return _relax_integrality(node, handler.optimizer)
end

duality_log_key(::ContinuousConicDuality) = " "
