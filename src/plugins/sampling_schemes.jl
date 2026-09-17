#  Copyright (c) 2017-25, Oscar Dowson and RVSDDP.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

# ========================= Monte Carlo Sampling Scheme ====================== #

struct InSampleMonteCarlo <: AbstractSamplingScheme
    max_depth::Int
    rollout_limit::Function
    initial_node::Any
end

"""
    InSampleMonteCarlo(;
        max_depth::Int = 0,
        rollout_limit::Function = i -> typemax(Int),
        initial_node::Any = nothing,
    )

A Monte Carlo sampling scheme using the in-sample data from the policy graph
definition.

A trajectory stops at a leaf node or at the depth limit, whichever comes first.
The depth limit is `min(max_depth, rollout_limit(iteration))`, where
`max_depth = 0` means "no limit of its own", so the two can be used on their own
or together:

    InSampleMonteCarlo(max_depth = 120)              # same cap every iteration
    InSampleMonteCarlo(rollout_limit = i -> 2 * i)   # grows with the iteration
    InSampleMonteCarlo(max_depth = 120, rollout_limit = i -> 2 * i)  # both

`iteration` is the training iteration the trajectory belongs to, which `train`
supplies; every trajectory of a given iteration is therefore given the same
limit, whatever `parallel` is. Outside training (`simulate`) it is 1.

A cyclic graph such as [`InfiniteLinearGraph`](@ref) has no leaf, so one of the
two limits must be set or the sampling never returns.

Control which node the trajectories start from using `initial_node`. If it is
left as `nothing`, the root node is used as the starting node.
"""
function InSampleMonteCarlo(;
    max_depth::Int = 0,
    rollout_limit::Function = i -> typemax(Int),
    initial_node::Any = nothing,
)
    return InSampleMonteCarlo(max_depth, rollout_limit, initial_node)
end

function get_noise_terms(
    sampling_scheme::InSampleMonteCarlo,
    node::Node{T},
    node_index::T,
) where {T}
    return node.noise_terms
end

function get_children(
    sampling_scheme::InSampleMonteCarlo,
    node::Node{T},
    node_index::T,
) where {T}
    return node.children
end

function get_root_children(
    sampling_scheme::InSampleMonteCarlo,
    graph::PolicyGraph{T},
) where {T}
    return graph.root_children
end

function sample_noise(noise_terms::Vector{<:Noise})
    if length(noise_terms) == 0
        return nothing
    end
    cumulative_probability = sum(noise.probability for noise in noise_terms)
    if cumulative_probability > 1.0 + 1e-6
        error("Cumulative probability cannot be greater than 1.0.")
    end
    rnd = rand() * cumulative_probability
    for noise in noise_terms
        rnd -= noise.probability
        if rnd <= 0.0
            return noise.term
        end
    end
    return error(
        "Internal RVSDDP error: unable to sample noise from $(noise_terms)",
    )
end

function sample_scenario(
    graph::PolicyGraph{T},
    sampling_scheme::InSampleMonteCarlo;
    iteration::Int = 1,
) where {T}
    # `max_depth = 0` means the scheme sets no cap of its own, so the effective
    # limit is whatever `rollout_limit` allows at this iteration (and vice
    # versa: the default `rollout_limit` allows everything).
    cap = sampling_scheme.max_depth == 0 ? typemax(Int) : sampling_scheme.max_depth
    max_depth = min(cap, sampling_scheme.rollout_limit(iteration))
    # Storage for our scenario. Each tuple is (node_index, noise.term).
    scenario_path = Tuple{T,Any}[]
    # Begin by sampling a node from the children of the root node.
    node_index = something(
        sampling_scheme.initial_node,
        sample_noise(get_root_children(sampling_scheme, graph)),
    )::T
    while true
        node = graph[node_index]
        noise_terms = get_noise_terms(sampling_scheme, node, node_index)
        children = get_children(sampling_scheme, node, node_index)
        noise = sample_noise(noise_terms)
        push!(scenario_path, (node_index, noise))
        # Termination conditions:
        if length(children) == 0
            # 1. Our node has no children, i.e., we are at a leaf node.
            return scenario_path
        elseif 0 < max_depth <= length(scenario_path)
            # 2. max_depth > 0 and we have explored max_depth number of nodes.
            return scenario_path
        end
        # Sample a new node to transition to.
        node_index = sample_noise(children)::T
    end
    # Throw an error because we should never end up here.
    return error(
        "Internal RVSDDP error: something went wrong sampling a scenario.",
    )
end
