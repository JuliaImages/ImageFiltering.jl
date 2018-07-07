using ImageFiltering, OffsetArrays, Logging, Test

@testset "basic" begin
    v = OffsetArray([1,2,3], -1:1)
    @test reflect(v) == OffsetArray([3,2,1], -1:1)

    rv = KernelFactors.ReshapedOneD{3,1}(v)
    @test ndims(rv) == 3
    @test rv[0] == 2
    @test rv == KernelFactors.ReshapedOneD{3,1}(Float64.(v))
    @test rv != KernelFactors.ReshapedOneD{3,1}([1,2,3])
    @test reshape(v, 0:0, -1:1, 0:0) == rv
    k3 = (KernelFactors.ReshapedOneD{3,0}(v), rv, KernelFactors.ReshapedOneD{3,2}(v))
    @test ImageFiltering.isseparable(k3)
    @test .*(k3...) == v .* reshape(v, 0:0, -1:1) .* reshape(v, 0:0, 0:0, -1:1)

    kern = OffsetArray([1 4 7; 2 5 8; 3 6 9], 0:2, -1:1)
    @test reflect(kern) == OffsetArray([9 6 3; 8 5 2; 7 4 1], -2:0, -1:1)

    @test eltype(KernelFactors.IIRGaussian(3)) == Float64
    @test eltype(KernelFactors.IIRGaussian(Float32, 3)) == Float32
    kern = KernelFactors.IIRGaussian([1,2.0f0])
    @test isa(kern, Tuple{KernelFactors.ReshapedOneD{Float32},KernelFactors.ReshapedOneD{Float32}})
    @test isa(ImageFiltering.filter_algorithm([1],[1],kern), Algorithm.IIR)
    @test isa(ImageFiltering.filter_algorithm([1],[1],(kern...,Kernel.Laplacian())), Algorithm.Mixed)
    kern = KernelFactors.IIRGaussian(Float64, [1,2.0f0])
    @test isa(kern, Tuple{KernelFactors.ReshapedOneD{Float64},KernelFactors.ReshapedOneD{Float64}})

    @test ndims(Pad(:replicate, (3,3))) == 2

    @test KernelFactors.kernelfactors(([0,3], [1,7]))  == (reshape([0,3], 1:2, 0:0), reshape([1,7], 0:0, 1:2))
    @test KernelFactors.kernelfactors(([0,3], [1,7]')) == (reshape([0,3], 2, 1), reshape([1,7], 1, 2))

    tiles = ImageFiltering.tile_allocate(Float32, (rand(3),rand(3)'))
    @test isa(tiles, Vector{Matrix{Float32}})
    @test length(tiles) == 1

    @test length(ImageFiltering.safetail(CartesianIndices(()))) == 1
    @test ImageFiltering.safetail(CartesianIndex(())) == CartesianIndex(())
    @test length(ImageFiltering.safehead(CartesianIndices(()))) == 1
    @test ImageFiltering.safehead(CartesianIndex(())) == CartesianIndex(())

    # Warnings
    fname = tempname()
    open(fname, "w") do f
        logger = SimpleLogger(f)
        with_logger(logger) do
            KernelFactors.IIRGaussian(0.5, emit_warning=false)
        end
    end
    @test isempty(chomp(read(fname, String)))
    open(fname, "w") do f
        logger = SimpleLogger(f)
        with_logger(logger) do
            KernelFactors.IIRGaussian(0.5)
        end
    end
    @test occursin("too small for accuracy", read(fname, String))
    rm(fname)
end

nothing
