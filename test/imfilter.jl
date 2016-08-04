using ImagesFiltering, Colors, OffsetArrays, ComputationalResources, Base.Test

function test_similar(a, b)
    @test indices(a) == indices(b)
    for I in eachindex(a)
        @test isapprox(a[I], b[I])
    end
end

@testset "FIR" begin
    approx_equal(ar, v) = all(abs(ar.-v) .< sqrt(eps(v)))
    r = CPU1(ImagesFiltering.FIR())
    for T in (Float64, Int)
        A = zeros(T,3,3); A[2,2] = 1
        kern = OffsetArray(rand(3,3), -1:1, -1:1)
        @test_approx_eq imfilter(r, A, kern) rot180(kern)
        kern = OffsetArray(rand(2,3),  0:1, -1:1)
        @test_approx_eq imfilter(r, A, kern)[1:2,:] rot180(kern)
        kern = OffsetArray(rand(2,3), -1:0, -1:1)
        @test_approx_eq imfilter(r, A, kern)[2:3,:] rot180(kern)
        kern = OffsetArray(rand(3,2), -1:1,  0:1)
        @test_approx_eq imfilter(r, A, kern)[:,1:2] rot180(kern)
        kern = OffsetArray(rand(3,2), -1:1, -1:0)
        @test_approx_eq imfilter(r, A, kern)[:,2:3] rot180(kern)
    end
    imgcol = rand(RGB{Float64}, 5, 6)
    imgcolf = convert(Array{RGB{U8}}, imgcol)
    k = zeros(3,3); k[2,2] = 1
    kern = OffsetArray(k, -1:1, -1:1)
    test_similar(imgcol, imfilter(r, imgcol, (kern,)))
    test_similar(imgcolf, imfilter(r, imgcolf, (kern,)))
    test_similar(imgcol, imfilter(r, imgcol, kern))
    test_similar(imgcolf, imfilter(r, imgcolf, kern))
    for T in (Float64, Int)
        # Separable kernels
        A = zeros(T,3,3); A[2,2] = 1
        kern = OffsetArray(rand(3).*rand(3)', -1:1, -1:1)
        fackern = ImagesFiltering.factorkernel(kern)
        @test isa(fackern, Tuple) && length(fackern) == 2
        @test_approx_eq fackern[1].*fackern[2] kern
        @test_approx_eq imfilter(r, A, kern) rot180(kern)
        kern = OffsetArray(rand(2).*rand(3)',  0:1, -1:1)
        @test_approx_eq imfilter(r, A, kern)[1:2,:] rot180(kern)
        kern = OffsetArray(rand(2).*rand(3)', -1:0, -1:1)
        @test_approx_eq imfilter(r, A, kern)[2:3,:] rot180(kern)
        kern = OffsetArray(rand(3).*rand(2)', -1:1,  0:1)
        @test_approx_eq imfilter(r, A, kern)[:,1:2] rot180(kern)
        kern = OffsetArray(rand(3).*rand(2)', -1:1, -1:0)
        @test_approx_eq imfilter(r, A, kern)[:,2:3] rot180(kern)
    end
    @test approx_equal(imfilter(ones(4,4), OffsetArray(ones(3,3),-1:1,-1:1)), 9.0)
    @test approx_equal(imfilter(ones(3,3), OffsetArray(ones(3,3),-1:1,-1:1)), 9.0)
    @test approx_equal(imfilter(ones(3,3), OffsetArray([1 1 1;1 0.0 1;1 1 1],-1:1,-1:1)), 8.0)
end

nothing
