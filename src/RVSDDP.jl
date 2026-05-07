module RVSDDP

import Reexport
Reexport.@reexport using JuMP

import Distributed
import HTTP
import JSON
import MutableArithmetics
import Printf
import Random
import SHA
import Statistics
import TimerOutputs
import CSV
import DataFrames
# import Serialization

# Work-around for https://github.com/JuliaPlots/RecipesBase.jl/pull/55
# Change this back to `import RecipesBase` once the fix is tagged.
using RecipesBase

export @stageobjective

# Modelling interface.
include("user_interface.jl")
include("modeling_aids.jl")

# Default definitions for RVSDDP related modular utilities.
include("plugins/headers.jl")

# Tools for overloading JuMP functions
include("binary_expansion.jl")
include("JuMP.jl")

# Printing utilities.
include("cyclic.jl")
include("print.jl")

# The core RVSDDP code.
include("algorithm.jl")

# Specific plugins.
include("plugins/risk_measures.jl")
include("plugins/sampling_schemes.jl")
include("plugins/bellman_functions.jl")
include("plugins/stopping_rules.jl")
include("plugins/local_improvement_search.jl")
include("plugins/duality_handlers.jl")
include("plugins/parallel_schemes.jl")
include("plugins/backward_sampling_schemes.jl")
include("plugins/forward_passes.jl")

# Visualization related code.
include("visualization/publication_plot.jl")
include("visualization/spaghetti_plot.jl")
include("visualization/dashboard.jl")
include("visualization/value_functions.jl")

# Other solvers.
include("deterministic_equivalent.jl")
include("biobjective.jl")
include("alternative_forward.jl")

include("Experimental.jl")
include("MSPFormat.jl")

#Mathis
include("plugins/two_stage.jl")

end
