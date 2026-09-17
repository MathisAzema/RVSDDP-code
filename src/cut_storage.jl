#  Copyright (c) 2017-25, Oscar Dowson and RVSDDP.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

# How a run's cuts are written to disk and read back.
#
# The cuts are by far the bulkiest thing a run produces, so the layout matters.
# Three properties of the data drive it:
#
#  * every cut carries one coefficient and one trial-state coordinate per state
#    variable, so those belong in fixed columns rather than in a per-row JSON
#    object that repeats the variable names on every line;
#  * a cut's shift history is not its own. `apply_shift!` lowers *every* cut of
#    the node to the same value, so each history is the running minimum of a
#    single per-node sequence of events, from the cut's creation on
#    (Proposition 8). Only that sequence needs storing;
#  * the columns are numeric and highly repetitive, which a compressed columnar
#    format exploits and a text format cannot.
#
# Hence `cuts.arrow` (one row per cut, fixed numeric columns) plus
# `shift_events.arrow` (one row per shift event). On a 146 571-cut run of the
# Brazilian model this is 8.8 MB where the previous `cuts.csv` was 63.9 MB.
#
# `cut_source` still reads that previous `cuts.csv` when it is the only thing a
# folder holds, so runs saved before this layout replay unchanged.

# `DataFrame` below is used unqualified, as in `value_functions.jl`.
using DataFrames

const CUTS_ARROW = "cuts.arrow"
const SHIFT_EVENTS_ARROW = "shift_events.arrow"
const CUTS_CSV = "cuts.csv"

# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

# The union of the state-variable keys of every node, in a fixed order. Nodes
# normally share them; a node missing one gets `missing` in that column.
function _state_keys(model::PolicyGraph)
    keys_ = Set{Symbol}()
    for (_, node) in model.nodes
        union!(keys_, keys(node.states))
    end
    return sort!(collect(keys_))
end

# The distinct shift events of `node`, oldest first. An event is an entry a
# cut's history received *after* its creation, and `apply_shift!` writes the
# same one into every cut it lowers --- hence the deduplication.
function _shift_events(node::Node)
    events = Set{Tuple{Float64,Int64}}()
    for cut in node.value_function.cut_V
        for k in 2:length(cut.shift)
            push!(events, cut.shift[k])
        end
    end
    # Ascending stamp; within a stamp, the larger shift first, so that replaying
    # them in this order reproduces the running minimum.
    return sort!(collect(events); by = e -> (e[2], -e[1]))
end

"""
    save_cuts(model::PolicyGraph, folder::String)

Write the cuts of every node of `model` under `folder`, as `cuts.arrow` plus
`shift_events.arrow`.

`cuts.arrow` holds one row per cut: `node`, `iteration`, `time`, `intercept`,
the cut's shift at creation (`shift0`, stamped `stamp0`), and one `coef_<v>`
and one `state_<v>` column per state variable `v`. Every later shift the cut
received lives in `shift_events.arrow` instead, once for the whole node rather
than once per cut; `cut_source` replays them back into per-cut histories.
"""
function save_cuts(model::PolicyGraph, folder::String)
    mkpath(folder)
    state_keys = _state_keys(model)
    n = sum(length(node.value_function.cut_V) for (_, node) in model.nodes)

    node_col = Vector{Int32}(undef, n)
    iteration = Vector{Int32}(undef, n)
    time = Vector{Float64}(undef, n)
    intercept = Vector{Float64}(undef, n)
    shift0 = Vector{Float64}(undef, n)
    stamp0 = Vector{Int32}(undef, n)
    coef = Dict(k => Vector{Union{Float64,Missing}}(missing, n) for k in state_keys)
    state = Dict(k => Vector{Union{Float64,Missing}}(missing, n) for k in state_keys)

    i = 0
    for (index, node) in model.nodes
        for cut in node.value_function.cut_V
            i += 1
            node_col[i] = index
            iteration[i] = cut.iteration
            time[i] = cut.time
            intercept[i] = cut.intercept
            shift0[i] = cut.shift[1][1]
            stamp0[i] = cut.shift[1][2]
            for (k, v) in cut.coefficients
                coef[k][i] = v
            end
            for (k, v) in cut.state
                state[k][i] = v
            end
        end
    end

    df = DataFrame(
        node = node_col,
        iteration = iteration,
        time = time,
        intercept = intercept,
        shift0 = shift0,
        stamp0 = stamp0,
    )
    for k in state_keys
        df[!, Symbol("coef_", k)] = coef[k]
        df[!, Symbol("state_", k)] = state[k]
    end
    Arrow.write(joinpath(folder, CUTS_ARROW), df; compress = :zstd)

    ev_node, ev_shift, ev_from = Int32[], Float64[], Int32[]
    for (index, node) in model.nodes
        for (value, from) in _shift_events(node)
            push!(ev_node, index)
            push!(ev_shift, value)
            push!(ev_from, from)
        end
    end
    Arrow.write(
        joinpath(folder, SHIFT_EVENTS_ARROW),
        DataFrame(node = ev_node, shift = ev_shift, effective_from = ev_from);
        compress = :zstd,
    )
    return
end

# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

"""
    CutSource

A saved run's cuts, read one row at a time.

`node`, `iteration` and `time` are the whole columns, which a caller filters on
cheaply. `row(i)` then materialises only the cuts it keeps: it is what rebuilds
a row's coefficient and state dictionaries, and its shift history, so a replay
that discards most of a run never pays for the cuts it discards.
"""
struct CutSource
    n::Int
    node::AbstractVector
    iteration::AbstractVector
    time::AbstractVector
    row::Function
end

"""
    cut_source(folder::String)

The cuts saved under `folder`, from `cuts.arrow` when it is there and from a
pre-Arrow `cuts.csv` otherwise.
"""
function cut_source(folder::String)
    if isfile(joinpath(folder, CUTS_ARROW))
        return _arrow_cut_source(folder)
    elseif isfile(joinpath(folder, CUTS_CSV))
        return _csv_cut_source(folder)
    end
    # Returning an empty source would leave the model without a single cut, and
    # the caller would go on to simulate and write results for an empty policy.
    # Under `pmap` the message alone would be lost on a worker's stdout.
    return error(
        "Nothing to replay: neither $(CUTS_ARROW) nor $(CUTS_CSV) under $(folder).",
    )
end

# Replay the node's shift events onto the cut created as its `cut_index`-th one.
# An event stamped `effective_from` was applied to the cuts that already
# existed, that is to the ones of strictly smaller index, and only where it
# lowered the shift they carried --- exactly what `apply_shift!` does.
function _rebuild_shift(
    shift0::Float64,
    stamp0::Int64,
    cut_index::Int,
    events::Vector{Tuple{Float64,Int64}},
)
    history = Tuple{Float64,Int64}[(shift0, stamp0)]
    for (value, from) in events
        if from > cut_index && value < history[end][1]
            push!(history, (value, from))
        end
    end
    return history
end

function _load_shift_events(folder::String)
    events = Dict{Int,Vector{Tuple{Float64,Int64}}}()
    path = joinpath(folder, SHIFT_EVENTS_ARROW)
    if !isfile(path)
        return events
    end
    tbl = Arrow.Table(path)
    for i in 1:length(tbl.node)
        push!(
            get!(events, Int(tbl.node[i]), Tuple{Float64,Int64}[]),
            (Float64(tbl.shift[i]), Int64(tbl.effective_from[i])),
        )
    end
    return events
end

# Position of each cut among those of its own node, counted over the whole
# file. This is the index the shift stamps are expressed in, so it must not
# depend on the filter a caller later applies.
function _cut_indices(node_column)
    indices = Vector{Int}(undef, length(node_column))
    seen = Dict{Int,Int}()
    for i in eachindex(indices)
        node = Int(node_column[i])
        indices[i] = seen[node] = get(seen, node, 0) + 1
    end
    return indices
end

function _arrow_cut_source(folder::String)
    tbl = Arrow.Table(joinpath(folder, CUTS_ARROW))
    columns = propertynames(tbl)
    prefixed(p) = [
        (Symbol(chop(String(c); head = length(p), tail = 0)), getproperty(tbl, c))
        for c in columns if startswith(String(c), p)
    ]
    coef_columns = prefixed("coef_")
    state_columns = prefixed("state_")
    indices = _cut_indices(tbl.node)
    events = _load_shift_events(folder)
    no_event = Tuple{Float64,Int64}[]

    function row(i::Int)
        node = Int(tbl.node[i])
        dict(cols) = Dict{Symbol,Float64}(
            k => Float64(c[i]) for (k, c) in cols if c[i] !== missing
        )
        return (
            iteration = Int(tbl.iteration[i]),
            time = Float64(tbl.time[i]),
            node = node,
            intercept = Float64(tbl.intercept[i]),
            coefficients = dict(coef_columns),
            shift = _rebuild_shift(
                Float64(tbl.shift0[i]),
                Int64(tbl.stamp0[i]),
                indices[i],
                get(events, node, no_event),
            ),
            state = dict(state_columns),
        )
    end
    return CutSource(length(tbl.node), tbl.node, tbl.iteration, tbl.time, row)
end

# The pre-Arrow layout: one JSON object per dictionary, and a shift history
# spelled out on every row. Read row by row all the same, so that replaying an
# old run no longer materialises every dictionary of the file at once.
function _csv_cut_source(folder::String)
    df = CSV.read(joinpath(folder, CUTS_CSV), DataFrame)
    parse_dict(s) = Dict{Symbol,Float64}(Symbol(k) => Float64(v) for (k, v) in JSON.parse(s))
    function row(i::Int)
        return (
            iteration = df.iteration[i],
            time = df.time[i],
            node = df.node[i],
            intercept = df.intercept[i],
            coefficients = parse_dict(df.coefficients[i]),
            shift = Tuple{Float64,Int64}[(s[1], s[2]) for s in JSON.parse(df.shift[i])],
            state = parse_dict(df.state[i]),
        )
    end
    return CutSource(DataFrames.nrow(df), df.node, df.iteration, df.time, row)
end
