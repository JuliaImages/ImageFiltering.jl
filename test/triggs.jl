using ImageFiltering, Colors, ComputationalResources, FixedPointNumbers
using Base.Test

# @testset "TriggsSdika" begin
#     @testset "1d" begin
        l = 1000
        x = [1, 2, 3, 5, 10, 20, l>>1, l-19, l-9, l-4, l-2, l-1, l]
        σs = [3.1, 5, 10.0, 20, 50.0, 100.0]
        # For plotting:
        # aσ = Array{Any}(length(x))
        # for i = 1:length(x)
        #     aσ[i] = Array{Float64}(l, 2*length(σs))
        # end
        for (n,σ) in enumerate(σs)
            kernel = KernelFactors.IIRGaussian(σ)
            A = [kernel.a'; eye(2) zeros(2)]
            B = [kernel.b'; eye(2) zeros(2)]
            I1 = zeros(3,3); I1[1,1] = 1
            @test kernel.M ≈ (I1+B*kernel.M*A)  # Triggs & Sdika, Eq. 8
            for (i,c) in enumerate(x)
                a = zeros(l)
                a[c] = 1
                af = exp.(-((1:l)-c).^2/(2*σ^2))/(σ*√(2π))
                @inferred(imfilter!(a, a, (kernel,), Fill(0)))
                @test norm(a-af) < 0.1*norm(af)
                # aσ[i][:,2n-1:2n] = [af a]
            end
        end

        kernel = KernelFactors.IIRGaussian(2.0)
        @test_throws DimensionMismatch imfilter([1.0, 2.0], (kernel,))
    # end

    # @testset "commutivity" begin
        img = 1:8
        k1 = KernelFactors.IIRGaussian(2)
        k2 = centered(ones(3)/3)
        for border in (Pad(:replicate), Fill(0))
            @test imfilter(img, (k1, k2), border) ≈ imfilter(img, (k2, k1), border)
        end
    # end

    # @testset "images" begin
        imgf = zeros(5, 7); imgf[3,4] = 1
        imgg = fill(Gray{Float32}(0), 5, 7); imgg[3,4] = 1
        imgc = fill(RGB{Float64}(0,0,0), 5, 7); imgc[3,4] = RGB(1,0,0)
        σ = 5
        x = -2:2
        y = (-3:3)'
        kernel = KernelFactors.IIRGaussian((σ, σ))
        for img in (imgf, imgg, imgc)
            imgcmp = img[3,4]*exp.(-(x.^2 .+ y.^2)/(2*σ^2))/(σ^2*2*pi)
            border = Fill(zero(eltype(img)))
            img0 = copy(img)
            imgfilt = @inferred(imfilter(img, kernel, border))
            @test sum(abs2, imgcmp - imgfilt) < 0.2^2*sum(abs2, imgcmp)
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

        # When the image has NaNs
        imgfnan = copy(imgf)
        imgfnum = copy(imgf)
        imgfnan[1,1] = NaN
        imgfnum[1,1] = 0
        imgfden = ones(imgfnum)
        imgfden[1,1] = 0
        retnum, retden = imfilter(imgfnum, kernel, Fill(0.0)), imfilter(imgfden, kernel, Fill(0.0))
        ret = imfilter(imgfnan, kernel, NA())
        ret[1,1] = retnum[1,1] = 0
        @test ret ≈ retnum./retden
#     end
# end

nothing
