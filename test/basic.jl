using ImageFiltering, ImageCore, OffsetArrays, Logging, ImageMetadata, Test
import AxisArrays
using AxisArrays: AxisArray, Axis

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
    @test length(tiles) == Threads.nthreads()

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

@testset "centered" begin
    check_range(r, f, l) = (@test first(r) == f; @test last(r) == l)
    check_range_axes(r, f, l) = check_range(axes(r)[1], f, l)

    check_range(axes(centered(1:3))[1], -1, 1)
    a = AxisArray(rand(3, 3), Axis{:y}(0.1:0.1:0.3), Axis{:x}(1:3))
    ca = centered(a)
    axs = axes(ca)
    check_range(axs[1], -1, 1)
    check_range(axs[2], -1, 1)
    axs = AxisArrays.axes(ca)
    check_range(axs[1].val, 0.1, 0.3)
    check_range(axs[2].val, 1, 3)
    check_range_axes(axs[1].val, -1, 1)
    check_range_axes(axs[1].val, -1, 1)
    am = ImageMeta(a; prop1="simple")
    ca = centered(am)
    axs = axes(ca)
    check_range(axs[1], -1, 1)
    check_range(axs[2], -1, 1)
    axs = AxisArrays.axes(ca)
    check_range(axs[1].val, 0.1, 0.3)
    check_range(axs[2].val, 1, 3)
    check_range_axes(axs[1].val, -1, 1)
    check_range_axes(axs[1].val, -1, 1)

    a = rand(3, 3)
    ca = centered(a)
    cca = centered(ca)
    c_view_a = centered(view(ca, :, 0:0))
    c_slice_a = centered(a[:, 2:2])
    @test a[2, 2] == ca[0, 0]
    @test a[2, 2] == cca[0, 0]
    @test a[2, 2] == c_view_a[0, 0]
    @test a[2, 2] == c_slice_a[0, 0]
end

@testset "freqkernel/spacekernel" begin
    k = centered(reshape([1,-1], 2, 1))
    kfft = freqkernel(k, (31, 31))
    @test size(kfft) == (31, 31)
    k2 = real.(spacekernel(kfft, axes(k)))
    @test k2 ≈ k
    kfft = freqkernel(k, (31, 31); rfft=true)
    @test size(kfft) == (16, 31)
    k2 = spacekernel(kfft, axes(k); rfftsz=31)
    @test k2 ≈ k

    k = centered([0, 1, 0] * [0, 1, 0]') # Kronecker impulse
    @test freqkernel(k, (8, 8)) ≈ ones(8,8) # flat frequency response

    @test_throws DimensionMismatch freqkernel(ones(5), (3,)) # kernel too big
    k = OffsetArray(ones(3), (-5,)) # index -4:-2 too far left
    @test_throws DimensionMismatch freqkernel(k, (5,))
    k = OffsetArray(ones(3), (+1,)) # index 2:4 too far right
    @test_throws DimensionMismatch freqkernel(k, (5,))

    for T in (Float64, Float32, Float16, N0f16)
        k = T.(Kernel.gaussian(3))
        kfft = freqkernel(k)
        @test size(kfft) == size(k)
        k2 = real.(spacekernel(kfft))
        @test T.(k2) ≈ k

        kfft = freqkernel(k; rfft=true)
        k2 = spacekernel(kfft; rfftsz=size(k, 1))
        @test T.(k2) ≈ k
    end
end
