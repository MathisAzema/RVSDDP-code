#  Copyright (c) 2017-25, Oscar Dowson and RVSDDP.jl contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Plumbing shared by every part of the algorithm: thread-safe timing, the
# per-trajectory subproblem replicas, and the task fan-out that runs a batch
# concurrently. Included first because `@_timeit_threadsafe` has to be defined
# before the files that use it are read.

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

"""
    _parallel_foreach_timed(f::Function, n::Int)

`_parallel_foreach(f, n)`, timing every task. Returns one named tuple per task:
the thread it finished on, and three durations in seconds measured from the
start of the step to the moment the last task is joined --- how long the task
waited before starting (`wait_start`, nonzero only when there are fewer threads
than tasks), how long it ran (`busy`), and how long it then waited for the
slowest task of the step (`wait_end`), i.e. the time its core sat idle before
the algorithm could move on.
"""
function _parallel_foreach_timed(f::Function, n::Int)
    starts = zeros(Int64, n)
    stops = zeros(Int64, n)
    threads = zeros(Int, n)
    t0 = Int64(time_ns())
    _parallel_foreach(n) do i
        starts[i] = Int64(time_ns())
        f(i)
        stops[i] = Int64(time_ns())
        threads[i] = Threads.threadid()
        return
    end
    t1 = Int64(time_ns())
    return [
        (
            thread = threads[i],
            wait_start = (starts[i] - t0) / 1e9,
            busy = (stops[i] - starts[i]) / 1e9,
            wait_end = (t1 - stops[i]) / 1e9,
        ) for i in 1:n
    ]
end
