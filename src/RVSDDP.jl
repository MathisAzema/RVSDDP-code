module RVSDDP

import Reexport
Reexport.@reexport using JuMP

import Distributed
import JSON
import MutableArithmetics
import Printf
import Random
import Statistics
import TimerOutputs
import Arrow
import CSV
import DataFrames

export @stageobjective

# ============================ Model and interfaces ==========================
# The objects a user builds (graph, nodes, subproblems) and the abstract types
# every swappable component of the algorithm is declared against.

include("user_interface.jl")
include("interfaces.jl")
include("JuMP.jl")

# ================================= The core =================================
# RV-SDDP itself. Included in dependency order: `Options` and the subproblem
# layer first, then the cut machinery, then the training loop, then the two
# passes and the shift rule they feed.

include("core/utils.jl")
include("core/options.jl")
include("core/subproblems.jl")
include("core/cuts.jl")
include("core/algorithm.jl")
include("core/forward_passes.jl")
include("core/backward_passes.jl")
include("core/shifts.jl")
include("core/refinement_schemes.jl")
include("core/simulate.jl")

# ================================== Plugins =================================
# Interchangeable implementations of the interfaces declared in
# `interfaces.jl`, selected through keyword arguments of `train`.

include("plugins/risk_measures.jl")
include("plugins/sampling_schemes.jl")
include("plugins/stopping_rules.jl")
include("plugins/duality_handlers.jl")
include("plugins/backward_sampling_schemes.jl")

# =========================== Results and reporting ==========================
# The training log and the numerical-stability report, then the tools that
# build, evaluate and replay the nodes' value functions once a run is over.

include("print.jl")
include("cut_storage.jl")
include("value_functions.jl")

end
