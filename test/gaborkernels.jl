@testset "GaborKernels" begin

function is_symmetric(X)
    @assert all(isodd, size(X))
    rst = map(CartesianIndices(size(X).÷2)) do i
        X[i] ≈ X[-i]
    end
    return all(rst)
end

@testset "Gabor" begin
    @testset "API" begin
        # Normally it just return a ComplexF64 matrix if users are careless
        kern = @inferred Kernel.Gabor((11, 11), 2, 0)
        @test size(kern) == (11, 11)
        @test axes(kern) == (-5:5, -5:5)
        @test eltype(kern) == ComplexF64
        # ensure that this is an efficient lazy array
        @test isbitstype(typeof(kern))

        # but still allow construction of ComplexF32 matrix
        kern = @inferred Kernel.Gabor((11, 11), 2.0f0, 0.0f0)
        @test eltype(kern) == ComplexF32

        # passing axes explicitly allows building a subregion of it
        kern1 = @inferred Kernel.Gabor((1:10, 1:10), 2.0, 0.0)
        kern2 = @inferred Kernel.Gabor((21, 21), 2.0, 0.0)
        @test kern2[1:end, 1:end] == kern1

        # keyword version makes it more explicit
        kern1 = @inferred Kernel.Gabor((11, 11), 2, 0)
        kern2 = @inferred Kernel.Gabor((11, 11); wavelength=2, orientation=0)
        @test kern1 === kern2
        # and currently we can't use both positional and keyword
        @test_throws UndefKeywordError Kernel.Gabor((5, 5))
        @test_throws MethodError Kernel.Gabor((11, 11), 2)
        @test_throws MethodError Kernel.Gabor((11, 11), 2; orientation=0)
        @test_throws MethodError Kernel.Gabor((11, 11), 2; wavelength=3)
        @test_throws MethodError Kernel.Gabor((11, 11), 2, 0; wavelength=3)

        # test default keyword values
        kern1 = @inferred Kernel.Gabor((11, 11), 2, 0)
        kern2 = @inferred Kernel.Gabor((11, 11), 2, 0; bandwidth=1, phase_offset=0, aspect_ratio=0.5)
        @test kern1 === kern2
    end

    @testset "getindex" begin
        kern = @inferred Kernel.Gabor((11, 11), 2, 0)
        @test kern[0, 0] ≈ 1
        @test kern[1] == kern[-5, -5]
        @test kern[end] == kern[5, 5]
    end

    @testset "symmetricity" begin
        # for some special orientation we can easily check the symmetric
        for θ in [0, π/2, -π/2, π, -π]
            kern = Kernel.Gabor((11, 11), 2, θ)
            @test is_symmetric(kern)
        end
        # for other orientations the symmetricity still holds, just that it doesn't
        # align along the x-y axis.
        kern = Kernel.Gabor((11, 11), 2, π/4)
        @test !is_symmetric(kern)
    end

    @testset "invalid inputs" begin
        if VERSION >= v"1.7-rc1"
            msg = "the expected kernel size is expected to be larger than"
            @test_warn msg Kernel.Gabor((11, 11), 5, 0)
            msg = "`wavelength` should be equal to or greater than 2"
            @test_warn msg Kernel.Gabor((11, 11), 1, 0)
        end
        @test_throws ArgumentError Kernel.Gabor((-1, -1), 2, 0)
        @test_throws ArgumentError Kernel.Gabor((11, 1), 2, 0; bandwidth=-1)
        @test_throws ArgumentError Kernel.Gabor((11, 1), 2, 0; aspect_ratio=-1)
    end

    @testset "numeric" begin
        function gabor(size_x, size_y, σ, θ, λ, γ, ψ)
            # plain implementation https://en.wikipedia.org/wiki/Gabor_filter
            # See also: https://github.com/JuliaImages/ImageFiltering.jl/pull/57
            σx, σy = σ, σ/γ
            s, c = sincos(θ)

            xmax = floor(Int, size_x/2)
            ymax = floor(Int, size_y/2)
            xmin = -xmax
            ymin = -ymax

            # The original implementation transposes x-y axis
            # x = [j for i in xmin:xmax,j in ymin:ymax]
            # y = [i for i in xmin:xmax,j in ymin:ymax]
            x = [i for i in xmin:xmax,j in ymin:ymax]
            y = [j for i in xmin:xmax,j in ymin:ymax]
            xr = x*c + y*s
            yr = -x*s + y*c

            kernel_real = (exp.(-0.5*(((xr.*xr)/σx^2) + ((yr.*yr)/σy^2))).*cos.(2*(π/λ)*xr .+ ψ))
            kernel_imag = (exp.(-0.5*(((xr.*xr)/σx^2) + ((yr.*yr)/σy^2))).*sin.(2*(π/λ)*xr .+ ψ))

            kernel = (kernel_real, kernel_imag)
            return kernel
        end

        kern1 = Kernel.Gabor((11, 11), 2, 0)
        σ, θ, λ, γ, ψ = kern1.σ, kern1.θ, kern1.λ, kern1.γ, kern1.ψ
        kern2_real, kern2_imag = gabor(map(length, kern1.ax)..., σ, θ, λ, γ, ψ)
        @test collect(real.(kern1)) ≈ kern2_real
        @test collect(imag.(kern1)) ≈ kern2_imag
    end

    @testset "applications" begin
        img = TestImages.shepp_logan(127)
        kern = Kernel.Gabor((19, 19), 2, 0)
        img_out1 = imfilter(img, real.(kern))

        kern_freq = freqkernel(real.(kern), size(img))
        img_out2 = Gray.(real.(ifft(fft(channelview(img)) .* kern_freq)))

        # Except for the boundary, these two methods produce the same result
        @test img_out1[10:end-10, 10:end-10] ≈ img_out2[10:end-10, 10:end-10]
    end
end


@testset "LogGabor" begin
    @testset "API" begin
        # LogGabor: r * a
        # LogGaborComplex: Complex(r, a)
        kern = @inferred Kernel.LogGabor((11, 11), 1/6, 0)
        kern_c = Kernel.LogGaborComplex((11, 11), 1/6, 0)
        @test kern == @. real(kern_c) * imag(kern_c)

        # Normally it just return a ComplexF64 matrix if users are careless
        kern = @inferred Kernel.LogGabor((11, 11), 2, 0)
        @test size(kern) == (11, 11)
        @test axes(kern) == (-5:5, -5:5)
        @test eltype(kern) == Float64
        # ensure that this is an efficient lazy array
        @test isbitstype(typeof(kern))

        # but still allow construction of ComplexF32 matrix
        kern = @inferred Kernel.LogGabor((11, 11), 1.0f0/6, 0.0f0)
        @test eltype(kern) == Float32

        # passing axes explicitly allows building a subregion of it
        kern1 = @inferred Kernel.LogGabor((1:10, 1:10), 1/6, 0)
        @test axes(kern1) == (1:10, 1:10)
        kern2 = @inferred Kernel.LogGabor((21, 21), 1/6, 0)
        # but they are not the same in normalize=true mode
        @test kern2[1:end, 1:end] != kern1

        # when normalize=false, they're the same
        kern1 = @inferred Kernel.LogGabor((1:10, 1:10), 1/6, 0; normalize=false)
        kern2 = @inferred Kernel.LogGabor((21, 21), 1/6, 0; normalize=false)
        @test kern2[1:end, 1:end] == kern1

        # test default keyword values
        kern1 = @inferred Kernel.LogGabor((11, 11), 2, 0)
        kern2 = @inferred Kernel.LogGabor((11, 11), 2, 0; σω=1, σθ=1, normalize=true)
        @test kern1 === kern2

        @test_throws ArgumentError Kernel.LogGabor((11, 11), 2, 0; σω=-1)
        @test_throws ArgumentError Kernel.LogGabor((11, 11), 2, 0; σθ=-1)
        @test_throws ArgumentError Kernel.LogGaborComplex((11, 11), 2, 0; σω=-1)
        @test_throws ArgumentError Kernel.LogGaborComplex((11, 11), 2, 0; σθ=-1)
    end

    @testset "symmetricity" begin
        kern = Kernel.LogGaborComplex((11, 11), 1/12, 0)
        # the real part is an even-symmetric function
        @test is_symmetric(real.(kern))
        # the imaginary part is a mirror along y-axis
        rows =  [imag.(kern[i, :]) for i in axes(kern, 1)]
        @test all(map(is_symmetric,  rows))

        # NOTE(johnnychen94): I'm not sure the current implementation is the standard or
        # correct. This numerical references are generated only to make sure future changes
        # don't accidentally break it.
        # Use N0f8 to ignore "insignificant" numerical changes to keep unit test happy
        ref_real = N0f8[
            0.741  0.796  0.82   0.796  0.741
            0.796  0.886  0.941  0.886  0.796
            0.82   0.941  0.0    0.941  0.82
            0.796  0.886  0.941  0.886  0.796
            0.741  0.796  0.82   0.796  0.741
        ]
        ref_imag = N0f8[
            0.063  0.027  0.008  0.027  0.063
            0.125  0.063  0.008  0.063  0.125
            0.29   0.29   1.0    0.29   0.29
            0.541  0.733  1.0    0.733  0.541
            0.733  0.898  1.0    0.898  0.733
        ]
        kern = Kernel.LogGaborComplex((5, 5), 1/12, 0)
        @test collect(N0f8.(real(kern))) == ref_real
        @test collect(N0f8.(imag(kern))) == ref_imag
    end

    @testset "applications" begin
        img = TestImages.shepp_logan(127)
        kern = Kernel.LogGabor(size(img), 1/100, π/4)
        out = @test_nowarn ifft(centered(fft(channelview(img))) .* ifftshift(kern))
    end
end

end
