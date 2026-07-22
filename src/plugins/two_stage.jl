#Mathis

function add_scenario_to_two_stage(
    model::JuMP.Model,
    two_stage::TwoStage,
    child::ScenarioTreeNode,
    s::Int64,
    check_time_limit::Function,
)
    check_time_limit()
    node = child.node
    parameterize(node, child.noise)
    # Add variables:
    src_variables = JuMP.all_variables(node.subproblem)
    x = @variable(model, [1:length(src_variables)])
    var_src_to_dest = Dict{JuMP.VariableRef,JuMP.VariableRef}()
    for (src, dest) in zip(src_variables, x)
        var_src_to_dest[src] = dest
        name = JuMP.name(src)
        if !isempty(name)
            # append node index to original variable name
            JuMP.set_name(dest, string(name, "#", s))
            if name==JuMP.name(node.bellman_function.global_theta.theta)
                two_stage.bellman_variables[s] = dest
            end
        else
            # append node index to original variable index
            var_name = string("_[", index(src).value, "]")
            JuMP.set_name(dest, string(var_name, "#", s))
        end
        for (i,x) in node.states
            if name==JuMP.name(x.out)
                two_stage.states[i][s]=dest
            end
        end
    end
    # Add constraints:
    for (F, S) in JuMP.list_of_constraint_types(node.subproblem)
        for con in JuMP.all_constraints(node.subproblem, F, S)
            obj = JuMP.constraint_object(con)
            new_func = copy_and_replace_variables(obj.func, var_src_to_dest)
            @constraint(model, new_func in obj.set)
        end
    end
    # Add objective:
    current = JuMP.objective_function(model)
    subproblem_objective =
        copy_and_replace_variables(objective_function(node.subproblem), var_src_to_dest)
    JuMP.set_objective_function(
        model,
        current + child.probability * subproblem_objective,
    )
    return
end

function copy_model(
    model_in::JuMP.Model,
    model_out::JuMP.Model
)
    # Add variables:
    src_variables = JuMP.all_variables(model_in)
    x = @variable(model_out, [1:length(src_variables)])
    var_src_to_dest = Dict{JuMP.VariableRef,JuMP.VariableRef}()
    for (src, dest) in zip(src_variables, x)
        var_src_to_dest[src] = dest
        name = JuMP.name(src)
        if !isempty(name)
            # append node index to original variable name
            JuMP.set_name(dest, name)
        else
            # append node index to original variable index
            var_name = string("_[", index(src).value, "]")
            JuMP.set_name(dest, var_name)
        end
    end
    # Add constraints:
    for (F, S) in JuMP.list_of_constraint_types(model_in)
        for con in JuMP.all_constraints(model_in, F, S)
            obj = JuMP.constraint_object(con)
            new_func = copy_and_replace_variables(obj.func, var_src_to_dest)
            @constraint(model_out, new_func in obj.set)
        end
    end
    # Add objective:
    @variable(model_out, theta)
    current = JuMP.objective_function(model_out)
    subproblem_objective =
        copy_and_replace_variables(objective_function(model_in), var_src_to_dest)
    JuMP.set_objective_function(
        model_out,
        theta
    )
    @constraint(model_out, theta >= subproblem_objective)
    @constraint(model_out, theta >= current)
    return
end

function add_next_node_to_scenario_tree(
    parent::Vector{ScenarioTreeNode{T}},
    pg::PolicyGraph{T},
    node::Node{T},
    check_time_limit::Function,
) where {T}
    if node.objective_state !== nothing
        throw_detequiv_error("Objective states detected!")
    elseif node.belief_state !== nothing
        throw_detequiv_error("Belief states detected!")
    elseif length(node.bellman_function.global_theta.cuts) > 0
        throw_detequiv_error(
            "Model has been used for training. Can only form deterministic " *
            "equivalent on a fresh model.",
        )
    else
        check_time_limit()
    end
    for noise in node.noise_terms
        scenario_node = ScenarioTreeNode(
            node,
            noise.term,
            noise.probability,
            ScenarioTreeNode{T}[],
            Dict{Symbol,State{JuMP.VariableRef}}(),
        )
        push!(parent, scenario_node)
    end
    return
end

function initialize_value_function(
    sense::Symbol,
    optimizer = nothing
)
    model = optimizer === nothing ? JuMP.Model() : JuMP.Model(optimizer)
    set_silent(model)
    theta = @variable(model, V)
    if sense == :Min
        @objective(model, Min, theta)
    else
        @objective(model, Max, theta)
    end

    #TV
    model_TV = optimizer === nothing ? JuMP.Model() : JuMP.Model(optimizer)
    set_silent(model_TV)
    theta_TV = @variable(model_TV, TV)
    if sense == :Min
        @objective(model_TV, Min, theta_TV)
    else
        @objective(model_TV, Max, theta_TV)
    end

    return Value_Function(
        model,
        Cut2[],
        theta,
        Dict{Symbol,JuMP.VariableRef}(),
        model_TV,
        theta_TV,
        Dict{Symbol,JuMP.VariableRef}(),
        Dict{Symbol,JuMP.Float64}(),
        Cut3[]
    )
end

function test_delta(
    node::Node{T},
    optimizer = nothing
) where {T}
    parameterize(node, sum(noise.term*noise.probability for noise in node.noise_terms))
    model_in = node.subproblem
    model_out = JuMP.Model(optimizer)
    set_silent(model_out)
    # Add variables:
    src_variables = JuMP.all_variables(model_in)
    x = @variable(model_out, [1:length(src_variables)])
    var_src_to_dest = Dict{JuMP.VariableRef,JuMP.VariableRef}()
    states = Dict{Symbol,JuMP.VariableRef}()
    name_states = [JuMP.name(x) for (i,x) in node.value_function.states]
    for (src, dest) in zip(src_variables, x)
        var_src_to_dest[src] = dest
        name = JuMP.name(src)
        if name in name_states
            states[Symbol(name)] = dest
        end
        if !isempty(name)
            # append node index to original variable name
            JuMP.set_name(dest, name)
        else
            # append node index to original variable index
            var_name = string("_[", index(src).value, "]")
            JuMP.set_name(dest, var_name)
        end
    end
    # Add constraints:
    for (F, S) in JuMP.list_of_constraint_types(model_in)
        for con in JuMP.all_constraints(model_in, F, S)
            obj = JuMP.constraint_object(con)
            new_func = copy_and_replace_variables(obj.func, var_src_to_dest)
            @constraint(model_out, new_func in obj.set)
        end
    end
    # Add objective:
    @variable(model_out, theta)
    subproblem_objective =
        copy_and_replace_variables(objective_function(model_in), var_src_to_dest)
    @objective(model_out, Min, theta)
    @constraint(model_out, theta >= subproblem_objective)
    for cut in node.value_function.cut_V
        @constraint(model_out, theta >= cut.intercept + sum(cut.coefficients[i] * states[Symbol(JuMP.name(x))] for (i,x) in node.value_function.states))
    end

    res = Inf
    vf=node.value_function
    N = length(vf.cut_V)
    LB = [0.0 for k in 1:N]
    sol = []
    for (k,cut) in enumerate(vf.cut_V)
        @objective(model_out, Min, cut.shift[end][1] + theta - cut.intercept - sum(cut.coefficients[i] * states[Symbol(JuMP.name(x))] for (i,x) in node.value_function.states))
        JuMP.optimize!(model_out)
        res = min(res, objective_value(model_out))
        LB[k] = objective_value(model_out)
        push!(sol, Dict(i => value(states[Symbol(JuMP.name(x))]) for (i,x) in node.value_function.states))
    end
    return LB, sol
end

function add_state_variables_to_value_function(
    node::Node
)
    value_function=node.value_function

    y = @variable(value_function.model, [1:length(node.states)])
    state_variables=[(key,var.in) for (key,var) in node.states]
    for (src, dest) in zip(state_variables, y)
        name = JuMP.name(src[2]) #MATHIS: checker si name exist
        JuMP.set_name(dest, name)
        value_function.states[src[1]]=dest
        lb=node.two_stage.lower_bounds[src[1]]
        ub=node.two_stage.upper_bounds[src[1]]
        value_function.heuristic_state[src[1]]=(ub+lb)/2
        @constraint(value_function.model, dest >= lb)
        @constraint(value_function.model, dest <= ub)
    end

    #TV
    y_TV = @variable(value_function.model_TV, [1:length(node.states)])
    state_variables=[(key,var.in) for (key,var) in node.states]
    for (src, dest) in zip(state_variables, y_TV)
        name = JuMP.name(src[2]) #MATHIS: checker si name exist
        JuMP.set_name(dest, name)
        value_function.states_TV[src[1]]=dest
        lb=node.two_stage.lower_bounds[src[1]]
        ub=node.two_stage.upper_bounds[src[1]]
        @constraint(value_function.model_TV, dest >= lb)
        @constraint(value_function.model_TV, dest <= ub)
    end

end

function initialize_two_stage(
    pg::PolicyGraph{T},
    node::Node,
    optimizer = nothing;
    time_limit::Union{Real,Nothing} = 60.0,
) where {T}
    two_stage=node.two_stage
    start_time = time()
    time_limit = time_limit === nothing ? typemax(Float64) : Float64(time_limit)
    function check_time_limit()
        if time() - start_time > time_limit::Float64
            throw_detequiv_error("Time limit exceeded!")
        end
    end
    tree = ScenarioTree{T}(ScenarioTreeNode{T}[])
    add_next_node_to_scenario_tree(
        tree.children,
        pg,
        node,
        check_time_limit,
    )
    for (i,x) in node.states
        two_stage.states[i]=Vector{VariableRef}(undef, length(tree.children))
    end
    two_stage.bellman_variables=Vector{VariableRef}(undef, length(tree.children))
    
    model = optimizer === nothing ? JuMP.Model() : JuMP.Model(optimizer)
    set_silent(model)
    set_objective_sense(model, pg.objective_sense)

    for (s, child) in enumerate(tree.children)
        add_scenario_to_two_stage(model, two_stage, child, s, check_time_limit)
    end

    y = @variable(model, [1:length(node.states)])
    incoming_variables=[(key,var) for (key,var) in node.states]
    for (src, dest) in zip(incoming_variables, y)
        name = JuMP.name(src[2].in) #MATHIS: checker si name exist
        JuMP.set_name(dest, string("non_anticipative_", name))
        two_stage.non_anticipative_variables[src[1]]=dest
        lb=lower_bound(src[2].out)
        ub=upper_bound(src[2].out)
        @constraint(model, dest >= lb)
        @constraint(model, dest <= ub) # Mathis : borne in et out sont les mêmes ?
        two_stage.lower_bounds[src[1]]=lb
        two_stage.upper_bounds[src[1]]=ub
        for (s, child) in enumerate(tree.children)
            var_s=variable_by_name(model, string(name, "#", s))
            expr=JuMP.GenericAffExpr(
                0.0,
                Pair{VariableRef,Float64}[
                    dest => 1.0,
                    var_s => -1.0,
                ],
            )
            @constraint(model, expr==0.0)
        end
    end
    two_stage.model=model
    return 
end

function compute_inf_TV(
    two_stage::TwoStage,
)
    mod=two_stage.model
    JuMP.optimize!(mod)
    inf_TV=JuMP.objective_value(mod)
    solution=Dict(i => value.(x) for (i,x) in two_stage.non_anticipative_variables)
    return (inf_TV, solution)
end

# function compute_TV(
#     two_stage::TwoStage,
#     incoming_state::Dict{Symbol,Float64}
# )
#     model=two_stage.model
#     for (i, value) in incoming_state
#         JuMP.fix(two_stage.non_anticipative_variables[i], value; force=true)
#     end
#     JuMP.optimize!(model)
#     obj=JuMP.objective_value(model)
#     for (i, value) in incoming_state
#         JuMP.unfix(two_stage.non_anticipative_variables[i])
#     end
#     return obj
# end

function compute_TV(
    node::Node,
    incoming_state::Dict{Symbol,Float64}
)
    TVx=0.0
    model=node.subproblem
    set_incoming_state(node, incoming_state)
    for noise in node.noise_terms
        parameterize(node, noise.term)
        JuMP.optimize!(model)
        TVx += noise.probability * JuMP.objective_value(model)
    end
    return TVx
end

function compute_jensen_TV(
    node::Node,
    incoming_state::Dict{Symbol,Float64}
)
    TVx=0.0
    model=node.subproblem
    set_incoming_state(node, incoming_state)
    parameterize(node, sum(noise.term*noise.probability for noise in node.noise_terms))
    JuMP.optimize!(model)
    return JuMP.objective_value(model)
end

function compute_TV2(
    node::Node,
    incoming_state::Dict{Symbol,Float64}
)
    TVx=0.0
    λ = Dict{Symbol,Float64}(
        name => 0.0 for
        (name, state) in node.states)
    model=node.subproblem
    set_incoming_state(node, incoming_state)
    for noise in node.noise_terms
        parameterize(node, noise.term)
        JuMP.optimize!(model)
        TVx += noise.probability * JuMP.objective_value(model)
        for (name, state) in node.states
            λ[name]+=noise.probability*JuMP.dual(JuMP.FixRef(state.in))
        end
    end
    return TVx, λ
end

function compute_inf_V(
    vf::Value_Function
)
    model=vf.model
    JuMP.optimize!(model)
    obj=JuMP.objective_value(model)
    return (obj, Dict(i => round.(value.(x)) for (i,x) in vf.states))
end

function compute_V(
    vf::Value_Function,
    incoming_state::Dict{Symbol,Float64}
)
    val = maximum([cut.intercept - cut.shift[end][1] + sum(cut.coefficients[i] * x for (i,x) in incoming_state) for cut in vf.cut_V])
    return val
end


function compute_approx_TV(
    vf::Value_Function,
    incoming_state::Dict{Symbol,Float64}
)
    val = maximum([cut.intercept + sum(cut.coefficients[i] * x for (i,x) in incoming_state) for cut in vf.cut_V])
    return val
end

function compute_V2(
    vf::Value_Function,
    incoming_state::Dict{Symbol,Float64}
)
    val, idx = findmax([cut.intercept + sum(cut.coefficients[i] * x for (i,x) in incoming_state) for cut in vf.cut_V])
    return val, vf.cut_V[idx].coefficients
end

function DCA(
    vf::Value_Function,
    incoming_state::Dict{Symbol,Float64}
)
    xk=incoming_state
    vk, gh_k = compute_V2(vf, xk)
    tvk = RVSDDP.compute_approx_TV(vf, xk)
    k=0
    model = vf.model_TV
    primal_obj = JuMP.objective_function(model)
    while true
        JuMP.set_objective_function(
            model,
            @expression(model, primal_obj - sum(gh_k[name] * state for (name, state) in vf.states_TV)),
        )
        JuMP.optimize!(model)
        tvk=JuMP.objective_value(model)
        xk1=Dict(i => value.(x) for (i,x) in vf.states_TV)
        vk, gh_k = compute_V2(vf, xk1)
        tvk = RVSDDP.compute_approx_TV(vf, xk1)
        k+=1
        if sum(abs.(xk1[i] - xk[i]) for i in keys(xk)) < 1e-4 || k > 0
            # if k>2
            #     println(k)
            # end
            JuMP.set_objective_function(
                model,
                @expression(model, primal_obj),
            )
            return xk
        else
            xk = xk1
        end
    end
end