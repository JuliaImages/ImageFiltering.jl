using ImagesFiltering, ImagesCore, OffsetArrays, Colors, FFTViews, ColorVectorSpace, ComputationalResources
using Base.Test

@testset "FIR/FFT" begin
    f32type(img) = f32type(eltype(img))
    f32type{C<:Colorant}(::Type{C}) = base_colorant_type(C){Float32}
    f32type{T<:Number}(::Type{T}) = Float32
    zerona!(a) = (a[isnan.(a)] = zero(eltype(a)); a)
    zerona!(a, nanflag) = (a[nanflag] = zero(eltype(a)); a)
    ## Images for which the boundary conditions will be irrelevant
    imgf = zeros(5, 7); imgf[3,4] = 1
    imgi = zeros(Int, 5, 7); imgi[3,4] = 1
    imgg = fill(Gray(0), 5, 7); imgg[3,4] = 1
    imgc = fill(RGB(0,0,0), 5, 7); imgc[3,4] = RGB(1,0,0)
    # Dense kernel
    kern = [0.1 0.2; 0.4 0.5]
    kernel = OffsetArray(kern, -1:0, 1:2)
    for img in (imgf, imgi, imgg, imgc)
        targetimg = zeros(typeof(img[1]*kern[1]), size(img))
        targetimg[3:4,2:3] = rot180(kern)*img[3,4]
        @test (ret = @inferred(imfilter(img, kernel))) ≈ targetimg
        @test @inferred(imfilter(img, (kernel,))) ≈ targetimg
        @test @inferred(imfilter(f32type(img), img, kernel)) ≈ float32(targetimg)
        fill!(ret, zero(eltype(ret)))
        @test @inferred(imfilter!(ret, img, kernel)) ≈ targetimg
        fill!(ret, zero(eltype(ret)))
        @test @inferred(imfilter!(CPU1(Algorithm.FIR()), ret, img, kernel)) ≈ targetimg
        for border in (Pad{:replicate}(), Pad{:circular}(), Pad{:symmetric}(), Pad{:reflect}(), Fill(zero(eltype(img))))
            @test (ret = @inferred(imfilter(img, kernel, border))) ≈ targetimg
            @test @inferred(imfilter(f32type(img), img, kernel, border)) ≈ float32(targetimg)
            fill!(ret, zero(eltype(ret)))
            @test @inferred(imfilter!(ret, img, kernel, border)) ≈ targetimg
            for alg in (Algorithm.FIR(), Algorithm.FFT())
                @test @inferred(imfilter(img, kernel, border, alg)) ≈ targetimg
                @test @inferred(imfilter(img, (kernel,), border, alg)) ≈ targetimg
                @test @inferred(imfilter(f32type(img), img, kernel, border, alg)) ≈ float32(targetimg)
                fill!(ret, zero(eltype(ret)))
                @test @inferred(imfilter!(CPU1(alg), ret, img, kernel, border)) ≈ targetimg
            end
            @test_throws MethodError imfilter!(CPU1(Algorithm.FIR()), ret, img, kernel, border, Algorithm.FFT())
        end
        targetimg_inner = OffsetArray(targetimg[2:end, 1:end-2], 2:5, 1:5)
        @test @inferred(imfilter(img, kernel, Inner())) ≈ targetimg_inner
        @test @inferred(imfilter(f32type(img), img, kernel, Inner())) ≈ float32(targetimg_inner)
        for alg in (Algorithm.FIR(), Algorithm.FFT())
            @test @inferred(imfilter(img, kernel, Inner(), alg)) ≈ targetimg_inner
            @test @inferred(imfilter(f32type(img), img, kernel, Inner(), alg)) ≈ float32(targetimg_inner)
            @test @inferred(imfilter(CPU1(alg), img, kernel, Inner())) ≈ targetimg_inner
        end
    end
    # Factored kernel
    kernel = (OffsetArray([0.2,0.8], -1:0), OffsetArray([0.3 0.6], 0:0, 1:2))
    kern = parent(kernel[1]).*parent(kernel[2])
    for img in (imgf, imgi, imgg, imgc)
        targetimg = zeros(typeof(img[1]*kern[1]), size(img))
        targetimg[3:4,2:3] = rot180(kern)*img[3,4]
        @test @inferred(imfilter(img, kernel)) ≈ targetimg
        @test @inferred(imfilter(f32type(img), img, kernel)) ≈ float32(targetimg)
        for border in (Pad{:replicate}(), Pad{:circular}(), Pad{:symmetric}(), Pad{:reflect}(), Fill(zero(eltype(img))))
            @test @inferred(imfilter(img, kernel, border)) ≈ targetimg
            @test @inferred(imfilter(f32type(img), img, kernel, border)) ≈ float32(targetimg)
            for alg in (Algorithm.FIR(), Algorithm.FFT())
                @test @inferred(imfilter(img, kernel, border, alg)) ≈ targetimg
                @test @inferred(imfilter(f32type(img), img, kernel, border, alg)) ≈ float32(targetimg)
            end
            @test_throws MethodError imfilter!(CPU1(Algorithm.FIR()), ret, img, kernel, border, Algorithm.FFT())
        end
        targetimg_inner = OffsetArray(targetimg[2:end, 1:end-2], 2:5, 1:5)
        @test @inferred(imfilter(img, kernel, Inner())) ≈ targetimg_inner
        @test @inferred(imfilter(f32type(img), img, kernel, Inner())) ≈ float32(targetimg_inner)
        for alg in (Algorithm.FIR(), Algorithm.FFT())
            @test @inferred(imfilter(img, kernel, Inner(), alg)) ≈ targetimg_inner
            @test @inferred(imfilter(f32type(img), img, kernel, Inner(), alg)) ≈ float32(targetimg_inner)
        end
    end
    # Rational filter coefficients
    kernel = (centered([1//3, 1//3, 1//3]), centered([1//3, 1//3, 1//3]'))
    kern = kernel[1].*kernel[2]
    for img in (imgf, imgi, imgg, imgc)
        targetimg = zeros(typeof(img[1]*kern[1]), size(img))
        targetimg[2:4,3:5] = img[3,4]*(1//9)
        @test @inferred(imfilter(img, kernel)) ≈ targetimg
        @test @inferred(imfilter(f32type(img), img, kernel)) ≈ float32(targetimg)
        for border in (Pad{:replicate}(), Pad{:circular}(), Pad{:symmetric}(), Pad{:reflect}(), Fill(zero(eltype(img))))
            @test @inferred(imfilter(img, kernel, border)) ≈ targetimg
            @test @inferred(imfilter(f32type(img), img, kernel, border)) ≈ float32(targetimg)
            for alg in (Algorithm.FIR(), Algorithm.FFT())
                if alg == Algorithm.FFT() && eltype(img) == Int
                    @test @inferred(imfilter(Float64, img, kernel, border, alg)) ≈ targetimg
                else
                    @test @inferred(imfilter(img, kernel, border, alg)) ≈ targetimg
                end
                @test @inferred(imfilter(f32type(img), img, kernel, border, alg)) ≈ float32(targetimg)
            end
        end
        targetimg_inner = OffsetArray(targetimg[2:end-1, 2:end-1], 2:4, 2:6)
        @test @inferred(imfilter(img, kernel, Inner())) ≈ targetimg_inner
        @test @inferred(imfilter(f32type(img), img, kernel, Inner())) ≈ float32(targetimg_inner)
        for alg in (Algorithm.FIR(), Algorithm.FFT())
            if alg == Algorithm.FFT() && eltype(img) == Int
                @test @inferred(imfilter(Float64, img, kernel, Inner(), alg)) ≈ targetimg_inner
            else
                @test @inferred(imfilter(img, kernel, Inner(), alg)) ≈ targetimg_inner
            end
            @test @inferred(imfilter(f32type(img), img, kernel, Inner(), alg)) ≈ float32(targetimg_inner)
        end
    end
    ## Images for which the boundary conditions matter
    imgf = zeros(5, 7); imgf[1,2] = 1
    imgi = zeros(Int, 5, 7); imgi[1,2] = 1
    imgg = fill(Gray(0), 5, 7); imgg[1,2] = 1
    imgc = fill(RGB(0,0,0), 5, 7); imgc[1,2] = RGB(1,0,0)
    # Dense kernel
    kern = [0.1 0.2; 0.4 0.5]
    kernel = OffsetArray(kern, -1:0, 1:2)
    function target1{T}(img::AbstractArray{T}, kern, ::Union{Pad{:replicate},Pad{:symmetric}})
        ret = float64(zero(img))
        x = img[1,2]
        ret[1,1] = (kern[1,1]+kern[2,1])*x
        ret[2,1] = kern[1,1]*x
        ret
    end
    function target1{T}(img::AbstractArray{T}, kern, ::Pad{:circular})
        a = FFTView(float64(zero(img)))
        x = img[1,2]
        a[0:1, -1:0] = rot180(kern).*x
        parent(a)
    end
    function target1{T}(img::AbstractArray{T}, kern, ::Union{Pad{:reflect}, Fill})
        ret = float64(zero(img))
        x = img[1,2]
        ret[1,1] = kern[2,1]*x
        ret[2,1] = kern[1,1]*x
        ret
    end
    function target1{T}(img::AbstractArray{T}, kern, ::Pad{:na})
        ret = float64(zero(img))
        x = img[1,2]
        ret[1,1] = kern[2,1]*x/(kern[2,1]+kern[2,2])
        ret[2,1] = kern[1,1]*x/sum(kern)
        ret[:,end] = nan(eltype(ret))  # for the last column, this kernel is entirely in the padding region
        ret
    end
    for img in (imgf, imgi, imgg, imgc)
        for border in (Pad{:replicate}(), Pad{:circular}(), Pad{:symmetric}(), Pad{:reflect}(), Fill(zero(eltype(img))))
            targetimg = target1(img, kern, border)
            @test @inferred(imfilter(img, kernel, border)) ≈ targetimg
            @test @inferred(imfilter(f32type(img), img, kernel, border)) ≈ float32(targetimg)
            for alg in (Algorithm.FIR(), Algorithm.FFT())
                @test @inferred(imfilter(img, kernel, border, alg)) ≈ targetimg
                @test @inferred(imfilter(f32type(img), img, kernel, border, alg)) ≈ float32(targetimg)
            end
        end
        border = Pad{:na}()
        targetimg0 = target1(img, kern, border)
        nanflag = isnan.(targetimg0)
        targetimg = zerona!(copy(targetimg0))
        @test @inferred(zerona!(imfilter(img, kernel, border))) ≈ targetimg
        @test @inferred(zerona!(imfilter(f32type(img), img, kernel, border))) ≈ float32(targetimg)
        for alg in (Algorithm.FIR(), Algorithm.FFT())
            @test @inferred(zerona!(imfilter(img, kernel, border, alg), nanflag)) ≈ targetimg
            @test @inferred(zerona!(imfilter(f32type(img), img, kernel, border, alg), nanflag)) ≈ float32(targetimg)
        end
    end
    # Factored kernel
    kernel = (OffsetArray([0.2,0.8], -1:0), OffsetArray([0.3 0.6], 0:0, 1:2))
    kern = parent(kernel[1]).*parent(kernel[2])
    for img in (imgf, imgi, imgg, imgc)
        for border in (Pad{:replicate}(), Pad{:circular}(), Pad{:symmetric}(), Pad{:reflect}(), Fill(zero(eltype(img))))
            targetimg = target1(img, kern, border)
            @test @inferred(imfilter(img, kernel, border)) ≈ targetimg
            @test @inferred(imfilter(f32type(img), img, kernel, border)) ≈ float32(targetimg)
            for alg in (Algorithm.FIR(), Algorithm.FFT())
                @test @inferred(imfilter(img, kernel, border, alg)) ≈ targetimg
                @test @inferred(imfilter(f32type(img), img, kernel, border, alg)) ≈ float32(targetimg)
            end
        end
        border = Pad{:na}()
        targetimg0 = target1(img, kern, border)
        nanflag = isnan.(targetimg0)
        targetimg = zerona!(copy(targetimg0))
        @test @inferred(zerona!(imfilter(img, kernel, border))) ≈ targetimg
        @test @inferred(zerona!(imfilter(f32type(img), img, kernel, border))) ≈ float32(targetimg)
        for alg in (Algorithm.FIR(), Algorithm.FFT())
            @test @inferred(zerona!(imfilter(img, kernel, border, alg), nanflag)) ≈ targetimg
            @test @inferred(zerona!(imfilter(f32type(img), img, kernel, border, alg), nanflag)) ≈ float32(targetimg)
        end
    end
    # filtering with a 0d kernel
    a = rand(3,3)
    # @test_approx_eq imfilter(a, reshape([2])) 2a
    # imfilter! as a copy
    ret = zeros(3, 3)
    b = @inferred(imfilter!(CPU1(Algorithm.FIR()), ret, a, (), NoPad()))
    @test b == a
    @test !(b === a)
end

@testset "TriggsSdika 1d" begin
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
        @test_approx_eq kernel.M (I1+B*kernel.M*A)  # Triggs & Sdika, Eq. 8
        for (i,c) in enumerate(x)
            a = zeros(l)
            a[c] = 1
            af = exp(-((1:l)-c).^2/(2*σ^2))/(σ*√(2π))
            @inferred(imfilter!(a, a, (kernel,), Fill(0)))
            @test norm(a-af) < 0.1*norm(af)
            # aσ[i][:,2n-1:2n] = [af a]
        end
    end
end

@testset "TriggsSdika images" begin
    imgf = zeros(5, 7); imgf[3,4] = 1
    imgg = fill(Gray{Float32}(0), 5, 7); imgg[3,4] = 1
    imgc = fill(RGB{Float64}(0,0,0), 5, 7); imgc[3,4] = RGB(1,0,0)
    σ = 5
    x = -2:2
    y = (-3:3)'
    kernel = KernelFactors.IIRGaussian((σ, σ))
    for img in (imgf, imgg, imgc)
        imgcmp = img[3,4]*exp(-(x.^2 .+ y.^2)/(2*σ^2))/(σ^2*2*pi)
        border = Fill(zero(eltype(img)))
        img0 = copy(img)
        imgf = @inferred(imfilter(img, kernel, border))
        @test sumabs2(imgcmp - imgf) < 0.2^2*sumabs2(imgcmp)
        @test img == img0
        @test imgf != img
    end
    ret = imfilter(imgf, KernelFactors.IIRGaussian((0,σ)), Pad{:replicate}())
    @test ret != imgf
    out = similar(ret)
    imfilter!(CPU1(Algorithm.IIR()), out, imgf, KernelFactors.IIRGaussian(σ), 2, Pad{:replicate}())
    @test out == ret
end


nothing
