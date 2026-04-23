# This file is maintained in a way to support CUDA-only test via
# `julia --project=test/cuda -e 'include("runtests.jl")'`
using ImageFiltering
using CUDA
using TestImages
using ImageBase
using ImageQualityIndexes
using Test
using Random

@testset "extrema CUDA" begin
    @testset "local extrema" begin
        a = zeros(Int, 9, 9)
        a[[1:2; 5], 5] .= 1
        ca = cu(a)

        @test findlocalmaxima(ca) == [CartesianIndex((5, 5))]
        @test findlocalmaxima(ca; window=(1, 3)) ==
              [CartesianIndex((1, 5)), CartesianIndex((2, 5)), CartesianIndex((5, 5))]
        @test findlocalmaxima(ca; window=(1, 3), edges=false) ==
              [CartesianIndex((2, 5)), CartesianIndex((5, 5))]

        a = zeros(Int, 9, 9, 9)
        a[[1:2; 5], 5, 5] .= 1
        ca = cu(a)

        @test findlocalmaxima(ca) == [CartesianIndex((5, 5, 5))]
        @test findlocalmaxima(ca; window=(1, 3, 1)) ==
              [CartesianIndex((1, 5, 5)), CartesianIndex((2, 5, 5)), CartesianIndex((5, 5, 5))]
        @test findlocalmaxima(ca; window=(1, 3, 1), edges=false) ==
              [CartesianIndex((2, 5, 5)), CartesianIndex((5, 5, 5))]

        a = zeros(Int, 9, 9)
        a[[1:2; 5], 5] .= -1
        ca = cu(a)

        @test findlocalminima(ca) == [CartesianIndex((5, 5))]
    end
end

CUDA.allowscalar(false)

@testset "ImageFiltering" begin
    if CUDA.functional()
        include("models.jl")
    end
end
