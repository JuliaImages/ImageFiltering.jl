using ImageFiltering, ImageCore, OffsetArrays
using Base.Test

@testset "cascade" begin
    a = rand(15)
    kern = OffsetArray(ones(3), -1:1)
    kern2 = OffsetArray(Float64[1,2,3,2,1], -2:2)  # the convolution of the above with self
    for border in ("replicate", "circular", "symmetric", "reflect", Fill(zero(eltype(a))))
        afc = imfilter(a, (kern, kern), border)
        af2 = imfilter(a, kern2, border)
        @test_approx_eq afc af2
    end

    a = rand(15, 15)
    kernx, kerny = OffsetArray(ones(3,1), -1:1, 0:0), OffsetArray(ones(1,3), 0:0, -1:1)
    c = Float64[1,2,3,2,1]
    kern2 = OffsetArray(c.*c', -2:2, -2:2)
    kern2x = OffsetArray(c.*ones(1,3), -2:2, -1:1)
    kern2y = OffsetArray(ones(3).*c',  -1:1, -2:2)
    for border in ("replicate", "circular", "symmetric", "reflect", Fill(zero(eltype(a))))
        afc = imfilter(a, (kernx, kerny, kernx, kerny), border)
        af2 = imfilter(a, kern2, border)
        @test_approx_eq afc af2
        afc = imfilter(a, (kernx, kernx, kerny, kerny), border)
        af2 = imfilter(a, kern2, border)
        @test_approx_eq afc af2
        afc = imfilter(a, (kernx, kernx, kerny), border)
        af2 = imfilter(a, kern2x, border)
        @test_approx_eq afc af2
        afc = imfilter(a, (kerny, kernx, kerny), border)
        af2 = imfilter(a, kern2y, border)
        @test_approx_eq afc af2
    end
end
