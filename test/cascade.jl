using ImageFiltering, ImageCore, OffsetArrays, ComputationalResources
using Test

@testset "cascade" begin
    a = rand(15)
    kern = OffsetArray(ones(3), -1:1)
    kern2 = OffsetArray(Float64[1,2,3,2,1], -2:2)  # the convolution of the above with self
    for r in (CPU1(Algorithm.FIR()), CPUThreads(Algorithm.FIR()))
        for border in ("replicate", "circular", "symmetric", "reflect", Fill(zero(eltype(a))))
            afc = imfilter(r, a, (kern, kern), border)
            af2 = imfilter(r, a, kern2, border)
            @test afc ≈ af2
        end
    end

    a = rand(15, 15)
    kernx, kerny = OffsetArray(ones(3,1), -1:1, 0:0), OffsetArray(ones(1,3), 0:0, -1:1)
    c = Float64[1,2,3,2,1]
    kern2 = OffsetArray(c.*c', -2:2, -2:2)
    kern2x = OffsetArray(c.*ones(1,3), -2:2, -1:1)
    kern2y = OffsetArray(ones(3).*c',  -1:1, -2:2)
    for r in (CPU1(Algorithm.FIR()), CPU1(Algorithm.FIRTiled()),
              CPUThreads(Algorithm.FIR()), CPUThreads(Algorithm.FIRTiled()))
        for border in ("replicate", "circular", "symmetric", "reflect", Fill(zero(eltype(a))))
            afc = imfilter(r, a, (kernx, kerny, kernx, kerny), border)
            af2 = imfilter(r, a, kern2, border)
            @test afc ≈ af2
            afc = imfilter(r, a, (kernx, kernx, kerny, kerny), border)
            af2 = imfilter(r, a, kern2, border)
            @test afc ≈ af2
            afc = imfilter(r, a, (kernx, kernx, kerny), border)
            af2 = imfilter(r, a, kern2x, border)
            @test afc ≈ af2
            afc = imfilter(r, a, (kerny, kernx, kerny), border)
            af2 = imfilter(r, a, kern2y, border)
            @test afc ≈ af2
        end
    end
end

nothing
