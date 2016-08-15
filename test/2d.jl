using ImagesFiltering, ImagesCore, OffsetArrays, Colors
using Base.Test

@testset "FIR/FFT" begin
    imgf = zeros(5, 7); imgf[3,4] = 1
    imgi = zeros(Int, 5, 7); imgi[3,4] = 1
    imgg = fill(Gray(0), 5, 7); imgg[3,4] = 1
    imgc = fill(RGB(0,0,0), 5, 7); imgc[3,4] = RGB(1,0,0)
    kern = [0.1 0.2; 0.4 0.5]
    kernel = OffsetArray(kern, -1:0, 1:2)
    f32type(img) = f32type(eltype(img))
    f32type{C<:Colorant}(::Type{C}) = base_colorant_type(C){Float32}
    f32type{T<:Number}(::Type{T}) = Float32
    for img in (copy(imgf), copy(imgi), copy(imgg), copy(imgc))
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
        end
        targetimg_inner = OffsetArray(targetimg[2:end, 1:end-2], 2:5, 1:3)
        @test @inferred(imfilter(img, kernel, Inner())) ≈ targetimg_inner
        @test @inferred(imfilter(f32type(img), img, kernel, Inner())) ≈ float32(targetimg_inner)
        for alg in (Algorithm.FIR(), Algorithm.FFT())
            @test @inferred(imfilter(img, kernel, Inner(), alg)) ≈ targetimg_inner
            @test @inferred(imfilter(f32type(img), img, kernel, Inner(), alg)) ≈ float32(targetimg_inner)
        end
    end
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
        kernel = Kernel.IIRGaussian(σ)
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
    kernel = Kernel.IIRGaussian((σ, σ))
    for img in (copy(imgf), copy(imgg), copy(imgc))
        imgcmp = img[3,4]*exp(-(x.^2 .+ y.^2)/(2*σ^2))/(σ^2*2*pi)
        border = Fill(zero(eltype(img)))
        imgf = @inferred(imfilter(img, kernel, border))
        @test sumabs2(imgcmp - imgf) < 0.2^2*sumabs2(imgcmp)
    end
end
