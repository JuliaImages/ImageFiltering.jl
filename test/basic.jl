using ImagesFiltering, OffsetArrays, Base.Test

@testset "basic" begin
    @test eltype(KernelFactors.IIRGaussian(3)) == Float64
    @test eltype(KernelFactors.IIRGaussian(Float32, 3)) == Float32
    @test isa(KernelFactors.IIRGaussian([1,2.0f0]), Tuple{KernelFactors.ReshapedVector,KernelFactors.ReshapedVector})

    @test KernelFactors.kernelfactors(([0,3], [1,7]))  == (reshape([0,3], 1:2, 0:0), reshape([1,7], 0:0, 1:2))
    @test KernelFactors.kernelfactors(([0,3], [1,7]')) == (reshape([0,3], 2, 1), reshape([1,7], 1, 2))

    # Warnings
    const OLDERR = STDERR
    fname = tempname()
    open(fname, "w") do f
        redirect_stderr(f)
        try
            KernelFactors.IIRGaussian(0.5, emit_warning=false)
        finally
            redirect_stderr(OLDERR)
        end
    end
    @test isempty(chomp(readstring(fname)))
    open(fname, "w") do f
        redirect_stderr(f)
        try
            KernelFactors.IIRGaussian(0.5)
        finally
            redirect_stderr(OLDERR)
        end
    end
    @test contains(readstring(fname), "too small for accuracy")
end

nothing
