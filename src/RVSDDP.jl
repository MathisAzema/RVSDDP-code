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
import CSV
import DataFrames

export @stageobjective

# Modelling interface.
include("user_interface.jl")

# Default definitions for RVSDDP related modular utilities.
include("plugins/headers.jl")

# Tools for overloading JuMP functions
include("JuMP.jl")

# Printing utilities.
include("print.jl")

# The core RVSDDP code.
include("algorithm.jl")

# Specific plugins.
include("plugins/risk_measures.jl")
include("plugins/sampling_schemes.jl")
include("plugins/bellman_functions.jl")
include("plugins/shifts.jl")
include("plugins/stopping_rules.jl")
include("plugins/duality_handlers.jl")
include("plugins/backward_sampling_schemes.jl")
include("plugins/forward_passes.jl")

# Building, evaluating and replaying the nodes' value functions.
include("value_functions.jl")

end
