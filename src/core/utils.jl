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
