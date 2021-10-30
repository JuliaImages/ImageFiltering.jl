# This file is maintained in a way to support CUDA-only test via
# `julia --project=test/cuda -e 'include("runtests.jl")'`
using ImageFiltering
using CUDA
using TestImages
using ImageBase
using ImageQualityIndexes
using Test
using Random

CUDA.allowscalar(false)

@testset "ImageFiltering" begin
    if CUDA.functional()
        include("models.jl")
    end
end
