#  Copyright (c) 2017-25, Oscar Dowson and RVSDDP.jl contributors
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

struct Graph{T}
    # The root node of the policy graph.
    root_node::T
    # nodes[x] returns a vector of the children of node x and their
    # probabilities.
    nodes::Dict{T,Vector{Tuple{T,Float64}}}
    # A partition of the nodes into ambiguity sets.
    belief_partition::Vector{Vector{T}}
    belief_lipschitz::Vector{Vector{Float64}}
end

"""
    Graph(root_node::T) where T

Create an empty graph struture with the root node `root_node`.

## Example

```jldoctest
julia> graph = RVSDDP.Graph(0)
Root
 0
Nodes
 {}
Arcs
 {}

julia> graph = RVSDDP.Graph(:root)
Root
 root
Nodes
 {}
Arcs
 {}

julia> graph = RVSDDP.Graph((0, 0))
Root
 (0, 0)
Nodes
 {}
Arcs
 {}
```
"""
function Graph(root_node::T) where {T}
    return Graph{T}(
        root_node,
        Dict{T,Vector{Tuple{T,Float64}}}(root_node => Tuple{T,Float64}[]),
        Vector{T}[],
        Vector{Float64}[],
    )
end

# Helper utilities to sort the nodes for printing. This helps linear and
# Markovian policy graphs where the nodes might be stored in an unusual ordering
# in the dictionary.
sort_nodes(nodes::Vector{Int}) = sort!(nodes)
sort_nodes(nodes::Vector{Tuple{Int,Int}}) = sort!(nodes)
sort_nodes(nodes::Vector{Tuple{Int,Float64}}) = sort!(nodes)
sort_nodes(nodes::Vector{Symbol}) = sort!(nodes)
sort_nodes(nodes) = nodes

function Base.show(io::IO, graph::Graph)
    println(io, "Root")
    println(io, " ", graph.root_node)
    println(io, "Nodes")
    nodes = sort_nodes(collect(keys(graph.nodes)))
    if first(nodes) != graph.root_node
        splice!(nodes, findfirst(isequal(graph.root_node), nodes))
        prepend!(nodes, [graph.root_node])
    end
    tree_nodes = filter(n -> n != graph.root_node, nodes)
    if isempty(tree_nodes)
        println(io, " {}")
    else
        for node in tree_nodes
            println(io, " ", node)
        end
    end
    print(io, "Arcs")
    has_arc = false
    for node in nodes
        for (child, probability) in graph.nodes[node]
            print(io, "\n ", node, " => ", child, " w.p. ", probability)
            has_arc = true
        end
    end
    if !has_arc
        print(io, "\n {}")
    end
    if length(graph.belief_partition) > 0
        print(io, "\nPartitions")
        for element in graph.belief_partition
            print(io, "\n {", join(string.(sort_nodes(element)), ", "), "}")
        end
    end
    return
end

# Internal function used to validate the structure of a graph
function _validate_graph(graph::Graph)
    for (node, children) in graph.nodes
        if length(children) > 0
            probability = sum(child[2] for child in children)
            if !(-1e-8 <= probability <= 1.0 + 1e-8)
                error(
                    "Probability on edges leaving node $(node) sum to " *
                    "$(probability), but this must be in [0.0, 1.0]",
                )
            end
        end
    end
    if length(graph.belief_partition) > 0
        # The -1 accounts for the root node, which shouldn't be in the
        # partition.
        if graph.root_node in union(graph.belief_partition...)
            error(
                "Belief partition $(graph.belief_partition) cannot contain " *
                "the root node $(graph.root_node).",
            )
        end
        if length(graph.nodes) - 1 != length(union(graph.belief_partition...))
            error(
                "Belief partition $(graph.belief_partition) does not form a" *
                " valid partition of the nodes in the graph.",
            )
        end
    end
    return
end

"""
    add_node(graph::Graph{T}, node::T) where {T}

Add a node to the graph `graph`.

## Examples

```jldoctest
julia> graph = RVSDDP.Graph(:root);

julia> RVSDDP.add_node(graph, :A)

julia> graph
Root
 root
Nodes
 A
Arcs
 {}
```

```jldoctest
julia> graph = RVSDDP.Graph(0);

julia> RVSDDP.add_node(graph, 2)

julia> graph
Root
 0
Nodes
 2
Arcs
 {}
```
"""
function add_node(graph::Graph{T}, node::T) where {T}
    if haskey(graph.nodes, node) || node == graph.root_node
        error("Node $(node) already exists!")
    end
    graph.nodes[node] = Tuple{T,Float64}[]
    return
end

function add_node(graph::Graph{T}, node) where {T}
    return error("Unable to add node $(node). Nodes must be of type $(T).")
end

function _add_node_if_missing(graph::Graph{T}, node::T) where {T}
    if haskey(graph.nodes, node) || node == graph.root_node
        return
    end
    return add_node(graph, node)
end

"""
    add_edge(graph::Graph{T}, edge::Pair{T, T}, probability::Float64) where {T}

Add an edge to the graph `graph`.

## Examples

```jldoctest
julia> graph = RVSDDP.Graph(0);

julia> RVSDDP.add_node(graph, 1)

julia> RVSDDP.add_edge(graph, 0 => 1, 0.9)

julia> graph
Root
 0
Nodes
 1
Arcs
 0 => 1 w.p. 0.9
```

```jldoctest
julia> graph = RVSDDP.Graph(:root);

julia> RVSDDP.add_node(graph, :A)

julia> RVSDDP.add_edge(graph, :root => :A, 1.0)

julia> graph
Root
 root
Nodes
 A
Arcs
 root => A w.p. 1.0
```
"""
function add_edge(
    graph::Graph{T},
    edge::Pair{T,T},
    probability::Float64,
) where {T}
    (parent, child) = edge
    if !(parent == graph.root_node || haskey(graph.nodes, parent))
        error("Node $(parent) does not exist.")
    elseif !haskey(graph.nodes, child)
        error("Node $(child) does not exist.")
    elseif child == graph.root_node
        error("Cannot have an edge entering the root node.")
    else
        push!(graph.nodes[parent], (child, probability))
    end
    return
end

function _add_to_or_create_edge(
    graph::Graph{T},
    edge::Pair{T,T},
    probability::Float64,
) where {T}
    for (i, (child, p)) in enumerate(graph.nodes[edge[1]])
        if child == edge[2]
            graph.nodes[edge[1]][i] = (edge[2], p + probability)
            return
        end
    end
    return add_edge(graph, edge, probability)
end

function Graph(
    root_node::T,
    nodes::Vector{T},
    edges::Vector{Tuple{Pair{T,T},Float64}},
) where {T}
    graph = Graph(root_node)
    add_node.(Ref(graph), nodes)
    for (edge, probability) in edges
        add_edge(graph, edge, probability)
    end
    return graph
end

"""
    LinearGraph(stages::Int)

Create a linear graph with `stages` number of nodes.

## Examples

```jldoctest
julia> graph = RVSDDP.LinearGraph(3)
Root
 0
Nodes
 1
 2
 3
Arcs
 0 => 1 w.p. 1.0
 1 => 2 w.p. 1.0
 2 => 3 w.p. 1.0
```
"""
function LinearGraph(stages::Int)
    edges = Tuple{Pair{Int,Int},Float64}[]
    for t in 1:stages
        push!(edges, (t - 1 => t, 1.0))
    end
    return Graph(0, collect(1:stages), edges)
end

#Mathis
function InfiniteLinearGraph(stages::Int)
    edges = Tuple{Pair{Int,Int},Float64}[]
    for t in 1:stages
        push!(edges, (t - 1 => t, 1.0))
    end
    push!(edges, (stages => 1, 1.0))
    return Graph(0, collect(1:stages), edges)
end

"""
    Noise(support, probability)

An atom of a discrete random variable at the point of support `support` and
associated probability `probability`.
"""
struct Noise{T}
    # The noise term.
    term::T
    # The probability of sampling the noise term.
    probability::Float64
end

struct State{T}
    # The incoming state variable.
    in::T
    # The outgoing state variable.
    out::T
end

mutable struct TwoStage
    model::JuMP.Model
    non_anticipative_variables::Dict{Symbol, VariableRef}
    states::Dict{Symbol, Vector{VariableRef}}
    bellman_variables::Vector{VariableRef}
    lower_bounds::Dict{Symbol, Float64}
    upper_bounds::Dict{Symbol, Float64}
end

mutable struct Cut2
    iteration::Int64
    time::Float64
    intercept::Float64
    coefficients::Dict{Symbol,Float64}
    shift::Vector{Tuple{Float64,Int64}}
    constraint_V::JuMP.ConstraintRef
    constraint_subproblem::Union{Nothing, JuMP.ConstraintRef}
    state::Dict{Symbol,Float64}
    # The same cut constraint as `constraint_subproblem`, but in each parallel
    # replica of the owning node's subproblem. Empty unless the model was built
    # with `max_parallel > 1` (see `_build_replicas!`). Kept in the same order
    # as `node.replicas`, so that a shift applied to `constraint_subproblem`
    # can be mirrored onto every replica.
    constraint_replicas::Vector{JuMP.ConstraintRef}
end

# Backwards-compatible constructor for call sites that predate replicas.
function Cut2(
    iteration::Int64,
    time::Float64,
    intercept::Float64,
    coefficients::Dict{Symbol,Float64},
    shift::Vector{Tuple{Float64,Int64}},
    constraint_V::JuMP.ConstraintRef,
    constraint_subproblem::Union{Nothing,JuMP.ConstraintRef},
    state::Dict{Symbol,Float64},
)
    return Cut2(
        iteration,
        time,
        intercept,
        coefficients,
        shift,
        constraint_V,
        constraint_subproblem,
        state,
        JuMP.ConstraintRef[],
    )
end

mutable struct Cut3
    intercept::Float64
    coefficients::Dict{Symbol,Float64}
end

mutable struct Value_Function
    model::JuMP.Model
    cut_V::Vector{Cut2}
    theta::JuMP.VariableRef
    states::Dict{Symbol,JuMP.VariableRef}
    model_TV::JuMP.Model
    theta_TV::JuMP.VariableRef
    states_TV::Dict{Symbol,JuMP.VariableRef}
    heuristic_state::Dict{Symbol, Float64}
    cut_TV::Vector{Cut3}
end

mutable struct Node{T}
    # The index of the node in the policy graph.
    index::T
    # The JuMP subproblem.
    subproblem::JuMP.Model
    # Mathis.
    two_stage::TwoStage
    value_function::Value_Function
    states_two_stage::Dict{Symbol,VariableRef}
    constraints::Vector{ConstraintRef}
    # A vector of the child nodes.
    children::Vector{Noise{T}}
    # A vector of the discrete stagewise-independent noise terms.
    noise_terms::Vector{Noise}
    # A function parameterize(model::JuMP.Model, noise) that modifies the JuMP
    # model based on the observation of the noise.
    parameterize::Function  # TODO(odow): make this a concrete type?
    # A list of the state variables in the model.
    states::Dict{Symbol,State{JuMP.VariableRef}}
    # Stage objective
    stage_objective::Any  # TODO(odow): make this a concrete type?
    stage_objective_set::Bool
    # Bellman function
    bellman_function::Any  # TODO(odow): make this a concrete type?
    # Objective-state and belief-state interpolation are unused features of
    # upstream SDDP.jl; these fields are always `nothing`.
    objective_state::Nothing
    belief_state::Nothing
    # An over-loadable hook for the JuMP.optimize! function.
    pre_optimize_hook::Union{Nothing,Function}
    post_optimize_hook::Union{Nothing,Function}
    # Approach for handling discrete variables.
    has_integrality::Bool
    # The user's optimizer. We use this in asynchronous mode.
    optimizer::Any
    # An extension dictionary. This is a useful place for packages that extend
    # RVSDDP.jl to stash things.
    ext::Dict{Symbol,Any}
    # Lock for threading
    lock::ReentrantLock
    # (lower, upper, is_integer)
    incoming_state_bounds::Dict{Symbol,Tuple{Float64,Float64,Bool}}
    #Mathis
    discount_factor::Float64
    delta::Vector{Float64}
    # Independent copies of this node, one per extra worker of a
    # `parallel > 1` batch (so `replicas[r]` is used by worker `r + 1`; worker
    # 1 uses the node itself). Each replica owns its own `subproblem`, its own
    # `parameterize` closure and its own cut constraints, so the workers of a
    # batch never touch the same JuMP model. Empty unless the graph was built
    # with `max_parallel > 1`. See `_build_replicas!`.
    replicas::Vector{Node{T}}
end

function Base.show(io::IO, node::Node)
    println(io, "Node $(node.index)")
    println(io, "  # State variables : ", length(node.states))
    println(io, "  # Children        : ", length(node.children))
    println(io, "  # Noise terms     : ", length(node.noise_terms))
    return
end

function pre_optimize_hook(f::Function, node::Node)
    node.pre_optimize_hook = f
    return
end

function post_optimize_hook(f::Function, node::Node)
    node.post_optimize_hook = f
    return
end

struct Log
    iteration::Int
    nb_cuts::Int
    bound::Float64
    simulation_value::Float64
    time::Float64
    pid::Int
    total_solves::Int
    duality_key::String
    serious_numerical_issue::Bool
    iter_cuts
end

struct TrainingResults
    status::Symbol
    log::Vector{Log}
end

mutable struct PolicyGraph{T}
    # Must be MOI.MIN_SENSE or MOI.MAX_SENSE
    objective_sense::MOI.OptimizationSense
    # Index of the root node.
    root_node::T
    # Children of the root node. child => probability.
    root_children::Vector{Noise{T}}
    # Starting value of the state variables.
    initial_root_state::Dict{Symbol,Float64}
    # All nodes in the graph.
    nodes::Dict{T,Node{T}}
    # Belief partition.
    belief_partition::Vector{Set{T}}
    # Storage for the most recent training results.
    most_recent_training_results::Union{Nothing,TrainingResults}
    # An extension dictionary. This is a useful place for packages that extend
    # RVSDDP.jl to stash things.
    ext::Dict{Symbol,Any}
    timer_output::TimerOutputs.TimerOutput
    lock::ReentrantLock
    discount_factor::Float64
    approx_value::Vector{Tuple{Float64, Float64}}

    function PolicyGraph(sense::Symbol, root_node::T, discount_factor::Float64) where {T}
        if sense != :Min && sense != :Max
            error(
                "The optimization sense must be `:Min` or `:Max`. It is $(sense).",
            )
        end
        optimization_sense = sense == :Min ? MOI.MIN_SENSE : MOI.MAX_SENSE
        return new{T}(
            optimization_sense,
            root_node,
            Noise{T}[],
            Dict{Symbol,Float64}(),
            Dict{T,Node{T}}(),
            Set{T}[],
            nothing,
            Dict{Symbol,Any}(),
            TimerOutputs.TimerOutput(),
            ReentrantLock(),
            discount_factor,
            Float64[],
        )
    end
end

function Base.show(io::IO, graph::PolicyGraph)
    N = length(graph.nodes)
    println(io, "A policy graph with $(N) nodes.")
    nodes = sort_nodes(collect(keys(graph.nodes)))
    if N < 10
        println(io, " Node indices: ", join(nodes, ", "))
    else
        println(io, " Node indices: ", nodes[1], ", ..., ", nodes[end])
    end
    return
end

# So we can query nodes in the graph as graph[node].
function Base.getindex(graph::PolicyGraph{T}, index::T) where {T}
    return graph.nodes[index]
end

# Work around different JuMP modes (Automatic / Manual / Direct).
function construct_subproblem(optimizer_factory, direct_mode::Bool)
    if direct_mode
        model = JuMP.direct_model(MOI.instantiate(optimizer_factory))
        set_silent(model)
        return model
    end
    return JuMP.Model()
end

# Work around different JuMP modes (Automatic / Manual / Direct).
function construct_subproblem(::Nothing, direct_mode::Bool)
    if direct_mode
        error(
            "You must specify an optimizer in the form:\n" *
            "    with_optimizer(Module.Opimizer, args...) if " *
            "direct_mode=true.",
        )
    end
    return JuMP.Model()
end

"""
    PolicyGraph(
        builder::Function,
        graph::Graph{T};
        sense::Symbol = :Min,
        lower_bound = -Inf,
        upper_bound = Inf,
        optimizer = nothing,
    ) where {T}

Construct a policy graph based on the graph structure of `graph`. (See
[`RVSDDP.Graph`](@ref) for details.)

## Keyword arguments

 - `sense`: whether we are minimizing (`:Min`) or maximizing (`:Max`).

 - `lower_bound`: if mimimizing, a valid lower bound for the cost to go in all
   subproblems.

 - `upper_bound`: if maximizing, a valid upper bound for the value to go in all
   subproblems.

 - `optimizer`: the optimizer to use for each of the subproblems

## Examples

```julia
function builder(subproblem::JuMP.Model, index)
    # ... subproblem definition ...
end

model = PolicyGraph(
    builder,
    graph;
    lower_bound = 0.0,
    optimizer = HiGHS.Optimizer,
)
```

Or, using the Julia `do ... end` syntax:

```julia
model = PolicyGraph(
    graph;
    lower_bound = 0.0,
    optimizer = HiGHS.Optimizer,
) do subproblem, index
    # ... subproblem definitions ...
end
```
"""

function PolicyGraph(
    builder::Function,
    graph::Graph{T};
    sense::Symbol = :Min,
    lower_bound = -Inf,
    upper_bound = Inf,
    optimizer = nothing,
    # These arguments are deprecated
    bellman_function = nothing,
    direct_mode::Bool = false,
    discount_factor::Float64 = 1.0,
    max_parallel::Int = 1,
) where {T}
    # Spend a one-off cost validating the graph.
    _validate_graph(graph)
    # Construct a basic policy graph. We will add to it in the remainder of this
    # function.
    policy_graph = PolicyGraph(sense, graph.root_node, discount_factor)
    # Create a Bellman function if one is not given.
    if bellman_function === nothing
        if sense == :Min && lower_bound === -Inf
            error(
                "You must specify a finite lower bound on the objective value" *
                " using the `lower_bound = value` keyword argument.",
            )
        elseif sense == :Max && upper_bound === Inf
            error(
                "You must specify a finite upper bound on the objective value" *
                " using the `upper_bound = value` keyword argument.",
            )
        else
            bellman_function = BellmanFunction(;
                lower_bound = lower_bound,
                upper_bound = upper_bound,
            )
        end
    end
    # Initialize nodes.
    for (node_index, children) in graph.nodes
        if node_index == graph.root_node
            continue
        end
        subproblem = construct_subproblem(optimizer, direct_mode)
        twostage=TwoStage(
            construct_subproblem(optimizer, direct_mode),
            Dict{Symbol,VariableRef}(),
            Dict{Symbol, Vector{VariableRef}}(),
            VariableRef[],
            Dict{Symbol, Float64}(),
            Dict{Symbol, Float64}()
        )

        valuefunction = initialize_value_function(sense, optimizer)
        node = Node(
            node_index,
            subproblem,
            twostage,
            valuefunction,
            Dict{Symbol, VariableRef}(),
            JuMP.ConstraintRef[],
            Noise{T}[],
            Noise[],
            (ω) -> nothing,
            Dict{Symbol,State{JuMP.VariableRef}}(),
            0.0,
            false,
            # Delay initializing the bellman function until later so that it can
            # use information about the children and number of
            # stagewise-independent noise realizations.
            nothing,
            # Likewise for the objective states.
            nothing,
            # And for belief states.
            nothing,
            # The optimize hook defaults to nothing.
            nothing,
            nothing,
            false,
            direct_mode ? nothing : optimizer,
            # The extension dictionary.
            Dict{Symbol,Any}(),
            ReentrantLock(),
            Dict{Symbol,Tuple{Float64,Float64,Bool}}(),
            discount_factor,
            Float64[],
            Node{T}[],
        )
        subproblem.ext[:RVSDDP_policy_graph] = policy_graph
        policy_graph.nodes[node_index] = subproblem.ext[:RVSDDP_node] = node
        JuMP.set_objective_sense(subproblem, policy_graph.objective_sense)
        builder(subproblem, node_index, discount_factor)
        # Add a dummy noise here so that all nodes have at least one noise term.
        if length(node.noise_terms) == 0
            push!(node.noise_terms, Noise(nothing, 1.0))
        end
        ctypes = JuMP.list_of_constraint_types(subproblem)
        node.has_integrality =
            (JuMP.VariableRef, MOI.Integer) in ctypes ||
            (JuMP.VariableRef, MOI.ZeroOne) in ctypes

    end
    # Loop back through and add the arcs/children.
    for (node_index, children) in graph.nodes
        if node_index == graph.root_node
            continue
        end
        node = policy_graph.nodes[node_index]
        for (child, probability) in children
            push!(node.children, Noise(child, probability))
        end
        # Intialize the bellman function. (See note in creation of Node above.)
        node.bellman_function =
            initialize_bellman_function(bellman_function, policy_graph, node)
    end
    # Add root nodes
    for (child, probability) in graph.nodes[graph.root_node]
        push!(policy_graph.root_children, Noise(child, probability))
        # We check the feasibility of the initial point here. It is a really
        # tricky feasibility bug to diagnose otherwise. See #387 for details.
        for (k, v) in policy_graph.initial_root_state
            x_out = policy_graph[child].states[k].out
            if JuMP.has_lower_bound(x_out) && JuMP.lower_bound(x_out) > v
                error("Initial point $(v) violates lower bound on state $k")
            elseif JuMP.has_upper_bound(x_out) && JuMP.upper_bound(x_out) < v
                error("Initial point $(v) violates upper bound on state $k")
            end
        end
    end
    domain = _get_incoming_domain(policy_graph)
    for (node_name, node) in policy_graph.nodes
        for (k, v) in domain[node_name]
            node.incoming_state_bounds[k] = something(v, (-Inf, Inf, false))
        end
    end
    _initialize_solver(policy_graph; throw_error = false)

    for (node_index, children) in graph.nodes
        if node_index == graph.root_node
            continue
        end
        node = policy_graph.nodes[node_index]
        initialize_two_stage(policy_graph, node, optimizer)
        add_state_variables_to_value_function(node)
    end
    # Everything `_build_replicas!` needs to rebuild a subproblem from scratch.
    # We keep it on the graph so that `RVSDDP.train(; parallel = k)` can create
    # the replicas itself if the graph was not built with `max_parallel >= k`.
    policy_graph.ext[:replica_factory] = (
        builder = builder,
        sense = sense,
        optimizer = optimizer,
        direct_mode = direct_mode,
        discount_factor = discount_factor,
        bellman_function = bellman_function,
    )
    _build_replicas!(policy_graph, max_parallel)
    return policy_graph
end

"""
    _build_replicas!(model::PolicyGraph, max_parallel::Int)

Make sure every node of `model` owns at least `max_parallel - 1` replicas, so
that a batch of `max_parallel` trajectories can be solved concurrently with one
independent JuMP model per worker.

A replica is built by re-running the user's `builder` on a fresh subproblem, so
its `parameterize` closure, stage objective and cut constraints all refer to its
own variables: two workers never touch the same JuMP model. Replicas only ever
need to be *solved* (`solve_subproblem`), so the expensive per-node extras that
are read on the master only --- the deterministic-equivalent `two_stage` model
and the `value_function` models --- are left empty.

Any cut already present on the master is replayed into the new replicas, so this
is safe to call on a partially trained model.
"""
function _build_replicas!(model::PolicyGraph{T}, max_parallel::Int) where {T}
    n_replicas = max(max_parallel - 1, 0)
    if n_replicas == 0 ||
       all(length(node.replicas) >= n_replicas for (_, node) in model.nodes)
        return model
    end
    factory = get(model.ext, :replica_factory, nothing)
    if factory === nothing
        error(
            "Cannot build the subproblem replicas needed by `parallel = " *
            "$(max_parallel)`: this policy graph was not created by " *
            "`RVSDDP.PolicyGraph(builder, graph; ...)`.",
        )
    end
    for (node_index, node) in model.nodes
        while length(node.replicas) < n_replicas
            push!(node.replicas, _build_replica(model, node, factory))
        end
    end
    # A replica added to an already-trained model starts out with no cuts, so
    # replay the master's cuts into every replica that is missing them.
    for (_, node) in model.nodes
        _replay_cuts_into_replicas!(model, node)
    end
    _initialize_solver(model; throw_error = false)
    return model
end

# Internal: build one independent copy of `node`'s subproblem.
function _build_replica(
    model::PolicyGraph{T},
    node::Node{T},
    factory,
) where {T}
    subproblem = construct_subproblem(factory.optimizer, factory.direct_mode)
    replica = Node(
        node.index,
        subproblem,
        # `two_stage` and `value_function` are master-only: a replica is never
        # passed to `compute_V` / `compute_TV` / `update_shift`, so building the
        # (expensive) deterministic equivalent for it would be pure waste.
        TwoStage(
            JuMP.Model(),
            Dict{Symbol,VariableRef}(),
            Dict{Symbol,Vector{VariableRef}}(),
            VariableRef[],
            Dict{Symbol,Float64}(),
            Dict{Symbol,Float64}(),
        ),
        initialize_value_function(factory.sense, nothing),
        Dict{Symbol,VariableRef}(),
        JuMP.ConstraintRef[],
        Noise{T}[],
        Noise[],
        (ω) -> nothing,
        Dict{Symbol,State{JuMP.VariableRef}}(),
        0.0,
        false,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        false,
        factory.direct_mode ? nothing : factory.optimizer,
        Dict{Symbol,Any}(),
        ReentrantLock(),
        copy(node.incoming_state_bounds),
        node.discount_factor,
        Float64[],
        Node{T}[],
    )
    # The builder resolves the node it is filling through `subproblem.ext`, so
    # pointing that at the replica is enough to keep the master node untouched.
    subproblem.ext[:RVSDDP_policy_graph] = model
    subproblem.ext[:RVSDDP_node] = replica
    JuMP.set_objective_sense(subproblem, model.objective_sense)
    factory.builder(subproblem, node.index, factory.discount_factor)
    if length(replica.noise_terms) == 0
        push!(replica.noise_terms, Noise(nothing, 1.0))
    end
    replica.has_integrality = node.has_integrality
    for child in node.children
        push!(replica.children, Noise(child.term, child.probability))
    end
    replica.bellman_function =
        initialize_bellman_function(factory.bellman_function, model, replica)
    replica.bellman_function.cut_type = node.bellman_function.cut_type
    replica.bellman_function.global_theta.deletion_minimum =
        node.bellman_function.global_theta.deletion_minimum
    # `initialize_bellman_function` also records, in the node's own value
    # function, the initial bound `θ >= lower_bound` it puts in the subproblem.
    # `update_shift` lowers that bound like any other cut, so the replica's copy
    # of it has to be tracked alongside the master's.
    if !isempty(node.value_function.cut_V) &&
       !isempty(replica.value_function.cut_V)
        bound_constraint = replica.value_function.cut_V[1].constraint_subproblem
        if bound_constraint !== nothing
            push!(
                node.value_function.cut_V[1].constraint_replicas,
                bound_constraint,
            )
        end
    end
    _initialize_solver(replica; throw_error = false)
    return replica
end

# Internal: give every replica of `node` the cut constraints the master already
# has, at their current (shifted) right-hand side.
function _replay_cuts_into_replicas!(
    model::PolicyGraph{T},
    node::Node{T},
) where {T}
    if isempty(node.replicas)
        return
    end
    for cut in _owned_cuts(node)
        while length(cut.constraint_replicas) < length(node.replicas)
            replica = node.replicas[length(cut.constraint_replicas)+1]
            push!(
                cut.constraint_replicas,
                _add_cut_constraint_to_subproblem(
                    replica,
                    cut.coefficients,
                    cut.intercept - cut.shift[end][1],
                ),
            )
        end
    end
    return
end

function _get_incoming_domain(model::PolicyGraph{T}) where {T}
    function _bounds(x)
        l, u = -Inf, Inf
        if has_lower_bound(x)
            l = lower_bound(x)
        end
        if has_upper_bound(x)
            u = upper_bound(x)
        end
        if is_fixed(x)
            l = u = fix_value(x)
        end
        is_int = is_integer(x)
        if is_binary(x)
            l, u = max(something(l, 0.0), 0.0), min(something(u, 1.0), 1.0)
            is_int = true
        end
        return l, u, is_int
    end
    outgoing_bounds = Dict{Tuple{T,Symbol},Any}(
        (k, state_name) => nothing for (k, node) in model.nodes for
        (state_name, _) in node.states
    )
    for (k, node) in model.nodes
        for noise in node.noise_terms
            parameterize(node, noise.term)
            for (state_name, state) in node.states
                domain = outgoing_bounds[(k, state_name)]
                l_new, u_new, is_int_new = _bounds(state.out)
                outgoing_bounds[(k, state_name)] = if domain === nothing
                    (l_new, u_new, is_int_new)
                else
                    l, u, is_int = domain::Tuple{Float64,Float64,Bool}
                    (min(l, l_new), max(u, u_new), is_int & is_int_new)
                end
            end
        end
    end
    incoming_bounds = Dict{T,Dict{Symbol,Any}}()
    for (k, node) in model.nodes
        incoming_bounds[k] = Dict{Symbol,Any}(
            state_name => nothing for (state_name, _) in node.states
        )
    end
    for (parent_name, parent) in model.nodes
        for (state_name, state) in parent.states
            domain_new = outgoing_bounds[(parent_name, state_name)]
            l_new, u_new, is_int_new = domain_new::Tuple{Float64,Float64,Bool}
            for child in parent.children
                domain = incoming_bounds[child.term][state_name]
                incoming_bounds[child.term][state_name] = if domain === nothing
                    (l_new, u_new, is_int_new)
                else
                    l, u, is_int = domain::Tuple{Float64,Float64,Bool}
                    (min(l, l_new), max(u, u_new), is_int & is_int_new)
                end
            end
        end
    end
    # The incoming state from the root node can be anything
    for (state_name, value) in model.initial_root_state
        for child in model.root_children
            incoming_bounds[child.term][state_name] = (-Inf, Inf, false)
        end
    end
    return incoming_bounds
end

# Internal function: helper to get the node given a subproblem.
function get_node(subproblem::JuMP.Model)
    return subproblem.ext[:RVSDDP_node]::Node
end

# Internal function: helper to get the policy graph given a subproblem.
function get_policy_graph(subproblem::JuMP.Model)
    return subproblem.ext[:RVSDDP_policy_graph]::PolicyGraph
end

"""
    parameterize(
        modify::Function,
        subproblem::JuMP.Model,
        realizations::Vector{T},
        probability::Vector{Float64} = fill(1.0 / length(realizations))
    ) where {T}

Add a parameterization function `modify` to `subproblem`. The `modify` function
takes one argument and modifies `subproblem` based on the realization of the
noise sampled from `realizations` with corresponding probabilities
`probability`.

In order to conduct an out-of-sample simulation, `modify` should accept
arguments that are not in realizations (but still of type T).

## Examples

```julia
RVSDDP.parameterize(subproblem, [1, 2, 3], [0.4, 0.3, 0.3]) do ω
    JuMP.set_upper_bound(x, ω)
end
```
"""
function parameterize(
    modify::Function,
    subproblem::JuMP.Model,
    realizations::AbstractVector{T},
    probability::AbstractVector{Float64} = fill(
        1.0 / length(realizations),
        length(realizations),
    ),
) where {T}
    node = get_node(subproblem)
    if length(node.noise_terms) != 0
        error("Duplicate calls to RVSDDP.parameterize detected.")
    end
    for (realization, prob) in zip(realizations, probability)
        push!(node.noise_terms, Noise(realization, prob))
    end
    node.parameterize = modify
    return
end

"""
    set_stage_objective(
        subproblem::JuMP.Model,
        stage_objective::Union{Real,JuMP.AbstractJuMPScalar},
    )

Set the stage-objective of `subproblem` to `stage_objective`.

## Examples

```julia
RVSDDP.set_stage_objective(subproblem, 2x + 1)
```
"""
function set_stage_objective(
    subproblem::JuMP.Model,
    stage_objective::Union{Real,JuMP.AbstractJuMPScalar},
)
    node = get_node(subproblem)
    node.stage_objective = stage_objective
    node.stage_objective_set = false
    return
end

function set_stage_objective(::JuMP.Model, f)
    return error(
        "Unable to set the stage-objective of type $(typeof(f)). It must be " *
        "a scalar function.",
    )
end

"""
    @stageobjective(subproblem, expr)

Set the stage-objective of `subproblem` to `expr`.

## Examples

```julia
@stageobjective(subproblem, 2x + y)
```
"""
macro stageobjective(subproblem, expr)
    code = MutableArithmetics.rewrite_and_return(expr)
    return quote
        RVSDDP.set_stage_objective($(esc(subproblem)), $code)
    end
end

# Internal function: calculate <y, μ>.
function get_objective_state_component(node::Node)
    objective_state_component = JuMP.AffExpr(0.0)
    objective_state = node.objective_state
    if objective_state !== nothing
        for (y, μ) in zip(objective_state.state, objective_state.μ)
            JuMP.add_to_expression!(objective_state_component, y, μ)
        end
    end
    return objective_state_component
end

# Internal function: calculate <b, μ>.
function get_belief_state_component(node::Node)
    belief_component = JuMP.AffExpr(0.0)
    if node.belief_state !== nothing
        belief = node.belief_state
        for (key, μ) in belief.μ
            JuMP.add_to_expression!(belief_component, belief.belief[key], μ)
        end
    end
    return belief_component
end
