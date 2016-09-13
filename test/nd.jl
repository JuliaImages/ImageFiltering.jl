using ImageFiltering, OffsetArrays
using Base.Test

@testset "1d" begin
    # Cascades don't result in out-of-bounds values
    img = 1:8
    k1 = centered([0.25, 0.5, 0.25])
    k2 = OffsetArray([0.5, 0.5], 1:2)
    casc = imfilter(img, (k1, k2, k1))
    A0 = padarray(img, Pad(:replicate, (2,), (4,)))
    A1 = imfilter(A0, k1, Inner())
    @test_approx_eq A1 OffsetArray([1.0,1.25,2.0,3.0,4.0,5.0,6.0,7.0,7.75,8.0,8.0,8.0], 0:11)
    A2 = imfilter(A1, k2, Inner())
    @test_approx_eq A2 OffsetArray([1.625,2.5,3.5,4.5,5.5,6.5,7.375,7.875,8.0,8.0], 0:9)
    A3 = imfilter(A2, k1, Inner())
    @test_approx_eq casc A3
    @test size(casc) == size(img)
    # copy! kernels, presence/order doesn't matter
    kc = centered([1])
    @test ImageFiltering.iscopy(kc)
    @test_approx_eq imfilter(img, (k1,)) [1.25,2.0,3.0,4.0,5.0,6.0,7.0,7.75]
    @test_approx_eq imfilter(img, (kc, k1)) [1.25,2.0,3.0,4.0,5.0,6.0,7.0,7.75]
    @test_approx_eq imfilter(img, (k1, kc)) [1.25,2.0,3.0,4.0,5.0,6.0,7.0,7.75]
end

@testset "3d" begin
    img = trues(10,10,10)
    kernel = centered(trues(3,3,3)/27)
    for border in ("replicate", "circular", "symmetric", "reflect", Fill(true))
        for alg in (Algorithm.FIR(), Algorithm.FFT())
            @test_approx_eq imfilter(img, kernel, border) img
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
    @test_approx_eq imfilter(img, kernel, Fill(0)) target
    @test_approx_eq imfilter(img, kernel, Fill(0), Algorithm.FFT()) target
    # Inner
    @test_approx_eq imfilter(img, kernel, Inner()) OffsetArray(trues(8,8,8), 2:9, 2:9, 2:9)
end

nothing
