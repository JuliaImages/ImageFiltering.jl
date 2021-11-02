# This file is maintained in a way to support CUDA-only test via
# `julia --project=test/cuda -e 'include("runtests.jl")'`
using ImageFiltering
using CUDA
using TestImages
using ImageBase
using ImageQualityIndexes
using Test
using Random
using FFTW
using OffsetArrays
using CUDA.Adapt

CUDA.allowscalar(false)

@testset "ImageFiltering" begin
    if CUDA.functional()
        include("models.jl")
        include("gaborkernels.jl")
    end
end
