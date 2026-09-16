#  Copyright (c) 2017-25, Oscar Dowson and RVSDDP.jl contributors and contributors.
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
     CompleteSampler()

Backward sampler that returns all noises of the corresponding node.
"""
struct CompleteSampler <: AbstractBackwardSamplingScheme end

sample_backward_noise_terms(::CompleteSampler, node) = node.noise_terms
