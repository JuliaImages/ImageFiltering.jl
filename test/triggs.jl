using ImageFiltering, ImageCore, ComputationalResources
using LinearAlgebra
using Test

@testset "TriggsSdika" begin
    @testset "1d" begin
        l = 1000
        x = [1, 2, 3, 5, 10, 20, l>>1, l-19, l-9, l-4, l-2, l-1, l]
        σs = [3.1, 5, 10.0, 20, 50.0, 100.0]
        # For plotting:
        # aσ = Array{Any}(undef, length(x))
        # for i = 1:length(x)
        #     aσ[i] = Array{Float64}(undef, l, 2*length(σs))
        # end
        for (n,σ) in enumerate(σs)
            kernel = KernelFactors.IIRGaussian(σ)
            A = [kernel.a'; I zeros(2)]
            B = [kernel.b'; I zeros(2)]
            I1 = zeros(3,3); I1[1,1] = 1
            @test kernel.M ≈ (I1+B*kernel.M*A)  # Triggs & Sdika, Eq. 8
            for (i,c) in enumerate(x)
                a = zeros(l)
                a[c] = 1
                af = exp.(-((1:l).-c).^2/(2*σ^2))/(σ*√(2π))
                @inferred(imfilter!(a, a, (kernel,), Fill(0)))
                @test norm(a-af) < 0.1*norm(af)
                # aσ[i][:,2n-1:2n] = [af a]
            end
        end

        kernel = KernelFactors.IIRGaussian(2.0)
        @test_throws DimensionMismatch imfilter([1.0, 2.0], (kernel,))
    end

    @testset "commutivity" begin
        img = 1:8
        k1 = KernelFactors.IIRGaussian(2)
        k2 = centered(ones(3)/3)
        for border in (Pad(:replicate), Fill(0))
            @test imfilter(img, (k1, k2), border) ≈ imfilter(img, (k2, k1), border)
        end
    end

    @testset "images" begin
        imgf = zeros(5, 7); imgf[3,4] = 1
        imgg = fill(Gray{Float32}(0), 5, 7); imgg[3,4] = 1
        imgc = fill(RGB{Float64}(0,0,0), 5, 7); imgc[3,4] = RGB(1,0,0)
        σ = 5
        x = -2:2
        y = (-3:3)'
        kernel = KernelFactors.IIRGaussian((σ, σ))

        for img in (imgf, imgg, imgc)
            imgcmp = img[3,4] .* exp.(-(x.^2 .+ y.^2)/(2*σ^2))/(σ^2*2*pi)
            border = Fill(zero(eltype(img)))
            img0 = copy(img)
            imgfilt = @inferred(imfilter(img, kernel, border))
            @test sum(_abs2, imgcmp - imgfilt) < 0.2^2*sum(_abs2, imgcmp)
            @test img == img0
            @test imgfilt != img
        end
        ret = imfilter(imgf, KernelFactors.IIRGaussian((σ,0)), "replicate")
        @test ret != imgf
        ret = imfilter(imgf, KernelFactors.IIRGaussian((0,σ)), "replicate")
        @test ret != imgf
        out = similar(ret)
        imfilter!(CPU1(Algorithm.IIR()), out, imgf, KernelFactors.IIRGaussian(σ), 2, "replicate")
        @test out == ret

        # for code coverage
        # TODO: make into an actual test
        kerng = KernelFactors.IIRGaussian(σ)
        kern = (kerng,kerng)
        imfilter(imgf, kern, NA())

        # When the image has NaNs
        imgfnan = copy(imgf)
        imgfnum = copy(imgf)
        imgfnan[1,1] = NaN
        imgfnum[1,1] = 0
        imgfden = fill(1, axes(imgfnum))
        imgfden[1,1] = 0
        retnum, retden = imfilter(imgfnum, kernel, Fill(0.0)), imfilter(imgfden, kernel, Fill(0.0))
        ret = imfilter(imgfnan, kernel, NA())
        ret[1,1] = retnum[1,1] = 0
        @test ret ≈ retnum./retden
    end

    @testset "OffsetArrays" begin
        A = reshape(collect(1:100*100),100,100)
        kern = ntuple(d->KernelFactors.IIRGaussian(5.0), 2)
        B = imfilter(A, kern, NA())
        for i = -5:5
            for j = -5:5
                C = OffsetArray(A,-1,-1)
                D = imfilter(C, kern, NA())
                @test collect(B) == collect(D)
            end
        end
    end
end

nothing
