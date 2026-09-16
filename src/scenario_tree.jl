#  Copyright (c) 2017-25, Oscar Dowson and RVSDDP.jl contributors
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

# This package's problem setting has an infinite number of scenarios (the
# horizon never terminates), so it never forms the full-horizon deterministic
# equivalent that upstream SDDP.jl offers. What is used --- here and by
# `plugins/two_stage.jl`, which builds the *one-stage* deterministic
# equivalent needed to compute a shift --- is only the scenario-tree
# scaffolding: a small tree type and a helper to copy a JuMP expression onto a
# fresh set of variables.

function throw_detequiv_error(msg::String)
    return error("Unable to formulate deterministic equivalent: ", msg)
end

struct ScenarioTreeNode{T}
    node::Node{T}
    noise::Any
    probability::Float64
    children::Vector{ScenarioTreeNode{T}}
    states::Dict{Symbol,State{JuMP.VariableRef}}
end

struct ScenarioTree{T}
    children::Vector{ScenarioTreeNode{T}}
end

function copy_and_replace_variables(
    src::Vector,
    map::Dict{JuMP.VariableRef,JuMP.VariableRef},
)
    return copy_and_replace_variables.(src, Ref(map))
end

function copy_and_replace_variables(
    src::Real,
    ::Dict{JuMP.VariableRef,JuMP.VariableRef},
)
    return src
end

function copy_and_replace_variables(
    src::JuMP.VariableRef,
    src_to_dest_variable::Dict{JuMP.VariableRef,JuMP.VariableRef},
)
    return src_to_dest_variable[src]
end

function copy_and_replace_variables(
    src::JuMP.GenericAffExpr,
    src_to_dest_variable::Dict{JuMP.VariableRef,JuMP.VariableRef},
)
    return JuMP.GenericAffExpr(
        src.constant,
        Pair{VariableRef,Float64}[
            src_to_dest_variable[key] => val for (key, val) in src.terms
        ],
    )
end

function copy_and_replace_variables(
    src::JuMP.GenericQuadExpr,
    src_to_dest_variable::Dict{JuMP.VariableRef,JuMP.VariableRef},
)
    return JuMP.GenericQuadExpr(
        copy_and_replace_variables(src.aff, src_to_dest_variable),
        Pair{UnorderedPair{VariableRef},Float64}[
            UnorderedPair{VariableRef}(
                src_to_dest_variable[pair.a],
                src_to_dest_variable[pair.b],
            ) => coef for (pair, coef) in src.terms
        ],
    )
end

function copy_and_replace_variables(
    src::Any,
    ::Dict{JuMP.VariableRef,JuMP.VariableRef},
)
    return throw_detequiv_error(
        "`copy_and_replace_variables` is not implemented for functions like `$(src)`.",
    )
end
