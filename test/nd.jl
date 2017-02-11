using ImageFiltering, OffsetArrays, ComputationalResources
using Base.Test

@testset "1d" begin
    img = 1:8
    # Exercise all the different ways to call imfilter
    kern = centered([1/3,1/3,1/3])
    imgf = imfilter(img, kern)
    r = CPU1(Algorithm.FIR())
    @test imfilter(Float64, img, kern) == imgf
    @test imfilter(Float64, img, (kern,)) == imgf
    @test imfilter(Float64, img, kern, "replicate") == imgf
    @test imfilter(Float64, img, (kern,), "replicate") == imgf
    @test_throws ArgumentError imfilter(Float64, img, (kern,), "inner")
    @test_throws ArgumentError imfilter(Float64, img, (kern,), "nonsense")
    @test imfilter(Float64, img, kern, "replicate", Algorithm.FIR()) == imgf
    @test imfilter(Float64, img, (kern,), "replicate", Algorithm.FIR()) == imgf
    @test imfilter(r, img, kern) == imgf
    @test imfilter(r, img, kern) == imgf
    @test imfilter(r, Float64, img, kern) == imgf
    @test imfilter(r, Float64, img, (kern,)) == imgf
    @test imfilter(r, img, kern, "replicate") == imgf
    @test imfilter(r, Float64, img, kern, "replicate") == imgf
    @test imfilter(r, img, (kern,), "replicate") == imgf
    @test imfilter(r, Float64, img, (kern,), "replicate") == imgf
    @test_throws MethodError imfilter(r, img, (kern,), "replicate", Algorithm.FIR())
    @test_throws MethodError imfilter(r, Float64, img, (kern,), "replicate", Algorithm.FIR())
    out = similar(imgf)
    @test imfilter!(out, img, kern) == imgf
    @test imfilter!(out, img, (kern,)) == imgf
    @test imfilter!(out, img, (kern,), "replicate") == imgf
    @test imfilter!(out, img, (kern,), "replicate", Algorithm.FIR()) == imgf
    @test imfilter!(r, out, img, kern) == imgf
    @test imfilter!(r, out, img, (kern,)) == imgf
    @test imfilter!(r, out, img, kern, "replicate") == imgf
    @test imfilter!(r, out, img, (kern,), "replicate") == imgf
    @test_throws MethodError imfilter!(r, out, img, (kern,), "replicate", Algorithm.FIR())

    # Element-type widening (issue #17)
    v = fill(0xff, 10)
    kern = centered(fill(0xff, 3))
    info("Two warnings are expected")
    @test_throws InexactError imfilter(v, kern)
    vout = imfilter(UInt32, v, kern)
    @test eltype(vout) == UInt32
    @test all(x->x==0x0002fa03, vout)

    # Cascades don't result in out-of-bounds values
    k1 = centered([0.25, 0.5, 0.25])
    k2 = OffsetArray([0.5, 0.5], 1:2)
    casc = imfilter(img, (k1, k2, k1))
    A0 = padarray(img, Pad(:replicate, (2,), (4,)))
    A1 = imfilter(A0, k1, Inner())
    @test A1 ≈ OffsetArray([1.0,1.25,2.0,3.0,4.0,5.0,6.0,7.0,7.75,8.0,8.0,8.0], 0:11)
    A2 = imfilter(A1, k2, Inner())
    @test A2 ≈ OffsetArray([1.625,2.5,3.5,4.5,5.5,6.5,7.375,7.875,8.0,8.0], 0:9)
    A3 = imfilter(A2, k1, Inner())
    @test casc ≈ A3
    @test size(casc) == size(img)
    # copy! kernels, presence/order doesn't matter
    kc = centered([1])
    @test ImageFiltering.iscopy(kc)
    @test imfilter(img, (k1,)) ≈ [1.25,2.0,3.0,4.0,5.0,6.0,7.0,7.75]
    @test imfilter(img, (kc, k1)) ≈ [1.25,2.0,3.0,4.0,5.0,6.0,7.0,7.75]
    @test imfilter(img, (k1, kc)) ≈ [1.25,2.0,3.0,4.0,5.0,6.0,7.0,7.75]

    # FFT without padding
    img = collect(1:8)
    img[1] = img[8] = 0
    out = imfilter!(CPU1(Algorithm.FFT()), similar(img, Float64), img, kern, NoPad())
    @test out ≈ imfilter(img, kern)
end

@testset "2d widening" begin
    # issue #17
    img = fill(typemax(Int16), 10, 10)
    kern = centered(Int16[1 2 2 2 1])
    @test_throws InexactError imfilter(img, kern)
    ret = imfilter(Int32, img, kern)
    @test eltype(ret) == Int32
    @test all(x->x==262136, ret)
end

@testset "3d" begin
    img = trues(10,10,10)
    kernel = centered(trues(3,3,3)/27)
    for border in ("replicate", "circular", "symmetric", "reflect", Fill(true))
        for alg in (Algorithm.FIR(), Algorithm.FIRTiled(), Algorithm.FFT())
            @test imfilter(img, kernel, border) ≈ img
        end
    end
    # Fill(0)
    target = convert(Array{Float64}, img)
    for i in (1,10)
        target[:,:,i] = target[:,i,:] = target[i,:,:] = 2/3
    end
    for i in (1,10)
        for j in (1,10)
            target[:,i,j] = target[i,:,j] = target[i,j,:] = (2/3)^2
        end
    end
    for i in (1,10)
        for j in (1,10)
            for k in (1,10)
                target[i,j,k] = (2/3)^3
            end
        end
    end
    @test imfilter(img, kernel, Fill(0)) ≈ target
    @test imfilter(img, kernel, Fill(0), Algorithm.FFT()) ≈ target
    # Inner
    @test imfilter(img, kernel, Inner()) ≈ OffsetArray(trues(8,8,8), 2:9, 2:9, 2:9)
end

nothing
