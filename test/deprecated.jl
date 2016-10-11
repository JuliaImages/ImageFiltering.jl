using ImageFiltering, ImageCore, Colors, FixedPointNumbers, OffsetArrays, Base.Test

isapprox_const(A::AbstractArray, n::Number) = isapprox(A, fill(n, size(A)))

@testset "Deprecated" begin
    @testset "Array padding" begin
        A = [1 2; 3 4]
        @test padarray(A, (0,0), (0,0), "replicate") == A
        @test padarray(A, (1,2), (2,0), "replicate") == OffsetArray([1 1 1 2; 1 1 1 2; 3 3 3 4; 3 3 3 4; 3 3 3 4], (-1,-2))
        @test padarray(A, [2,1], [0,2], "circular") == OffsetArray([2 1 2 1 2; 4 3 4 3 4; 2 1 2 1 2; 4 3 4 3 4], (-2,-1))
        @test padarray(A, (1,2), (2,0), "symmetric") == OffsetArray([2 1 1 2; 2 1 1 2; 4 3 3 4; 4 3 3 4; 2 1 1 2], (-1,-2))
        @test padarray(A, (1,2), (2,0), "value", -1) == OffsetArray([-1 -1 -1 -1; -1 -1 1 2; -1 -1 3 4; -1 -1 -1 -1; -1 -1 -1 -1], (-1,-2))
        A = [1 2 3; 4 5 6]
        @test padarray(A, (1,2), (2,0), "reflect") == OffsetArray([6 5 4 5 6; 3 2 1 2 3; 6 5 4 5 6; 3 2 1 2 3; 6 5 4 5 6], (-1,-2))
        A = [1 2; 3 4]
        @test padarray(A, (1,1)) == OffsetArray([1 1 2 2; 1 1 2 2; 3 3 4 4; 3 3 4 4], (-1,-1))
        @test padarray(A, (1,1), "replicate", "both") == OffsetArray([1 1 2 2; 1 1 2 2; 3 3 4 4; 3 3 4 4], (-1,-1))
        @test padarray(A, (1,1), "circular", "pre") == OffsetArray([4 3 4; 2 1 2; 4 3 4], (-1,-1))
        @test padarray(A, (1,1), "symmetric", "post") == [1 2 2; 3 4 4; 3 4 4]
        A = ["a" "b"; "c" "d"]
        @test padarray(A, (1,1)) == OffsetArray(["a" "a" "b" "b"; "a" "a" "b" "b"; "c" "c" "d" "d"; "c" "c" "d" "d"], (-1,-1))
        @test_throws ArgumentError padarray(A, (1,1), "unknown")
        # issue #292
        A = trues(3,3)
        @test typeof(parent(padarray(A, (1,2), (2,1), "replicate"))) == BitArray{2}
        @test eltype(padarray(grayim(A), (1,2), (2,1), "replicate")) == Gray{Bool}
        # issue #525
        A = falses(10,10,10)
        B = view(A,1:8,1:8,1:8)
        @test isa(parent(padarray(A, ones(Int,3), ones(Int,3), "replicate")), BitArray{3})
        @test isa(parent(padarray(B, ones(Int,3), ones(Int,3), "replicate")), BitArray{3})
    end

    @testset "Filtering" begin
        EPS = 1e-14
        imgcol = colorim(rand(3,5,6))
        imgcolf = convert(Array{RGB{N0f8}}, imgcol)
        for T in (Float64, Int)
            A = zeros(T,3,3); A[2,2] = 1
            kern = rand(3,3)
            @test maximum(abs(imfilter(A, kern) - rot180(kern))) < EPS
            kern = rand(2,3)
            @test maximum(abs(imfilter(A, kern)[1:2,:] - rot180(kern))) < EPS
            kern = rand(3,2)
            @test maximum(abs(imfilter(A, kern)[:,1:2] - rot180(kern))) < EPS
        end
        kern = zeros(3,3); kern[2,2] = 1
        @test maximum(map(abs, imgcol - imfilter(imgcol, kern))) < EPS
        @test maximum(map(abs, imgcolf - imfilter(imgcolf, kern))) < EPS
        for T in (Float64, Int)
            # Separable kernels
            A = zeros(T,3,3); A[2,2] = 1
            kern = rand(3).*rand(3)'
            @test maximum(abs(imfilter(A, kern) - rot180(kern))) < EPS
            kern = rand(2).*rand(3)'
            @test maximum(abs(imfilter(A, kern)[1:2,:] - rot180(kern))) < EPS
            kern = rand(3).*rand(2)'
            @test maximum(abs(imfilter(A, kern)[:,1:2] - rot180(kern))) < EPS
        end
        A = zeros(3,3); A[2,2] = 1
        kern = rand(3,3)
        @test maximum(abs(imfilter_fft(A, kern) - rot180(kern))) < EPS
        kern = rand(2,3)
        @test maximum(abs(imfilter_fft(A, kern)[1:2,:] - rot180(kern))) < EPS
        kern = rand(3,2)
        @test maximum(abs(imfilter_fft(A, kern)[:,1:2] - rot180(kern))) < EPS
        kern = zeros(3,3); kern[2,2] = 1
        @test maximum(map(abs, imgcol - imfilter_fft(imgcol, kern))) < EPS
        @test maximum(map(abs, imgcolf - imfilter_fft(imgcolf, kern))) < EPS

        @test isapprox_const(imfilter(ones(4,4), ones(3,3)), 9.0)
        @test isapprox_const(imfilter(ones(3,3), ones(3,3)), 9.0)
        @test isapprox_const(imfilter(ones(3,3), [1 1 1;1 0.0 1;1 1 1]), 8.0)
        img = ones(4,4)
        @test isapprox_const(imfilter(img, ones(3,3)), 9.0)
        A0 = zeros(5,5,3); A0[3,3,[1,3]] = 1
        A = colorview(RGB, permuteddimsview(A0, (3,1,2)))
        kern = rand(3,3)
        kernpad = zeros(5,5); kernpad[2:4,2:4] = kern
        Af = permuteddimsview(channelview(imfilter(A, kern)), (2,3,1))

        @test cat(3, rot180(kernpad), zeros(5,5), rot180(kernpad)) ≈ Af
        @test isapprox_const(imfilter(ones(4,4),ones(1,3),"replicate"), 3.0)

        A = zeros(5,5); A[3,3] = 1
        kern = rand(3,3)
        Af = imfilter(A, kern, Inner())
        @test Af == OffsetArray(rot180(kern), (1,1))
        Afft = imfilter_fft(A, kern, "inner")
        @test Af ≈ Afft
        h = [0.24,0.87]
        hfft = imfilter_fft(eye(3), h, "inner")
        hfft[abs(hfft) .< 3eps()] = 0
        @test imfilter(eye(3), h, Inner()) ≈ hfft  # issue #204

        # circular
        A = zeros(3, 3)
        A[3,2] = 1
        kern = rand(3,3)
        @test imfilter_fft(A, kern, "circular") ≈ kern[[1,3,2],[3,2,1]]

        A = zeros(5, 5)
        A[5,3] = 1
        kern = rand(3,3)
        @test imfilter_fft(A, kern, "circular")[[1,4,5],2:4] ≈ kern[[1,3,2],[3,2,1]]

        A = zeros(5, 5)
        A[5,3] = 1
        kern = rand(3,3)
        @test imfilter(A, kern, "circular")[[1,4,5],2:4] ≈ kern[[1,3,2],[3,2,1]]

        @test isapprox_const(imfilter_gaussian(ones(4,4), [5,5]), 1.0)
        A = fill(convert(Float32, NaN), 4, 4)
        A[1:4,1] = 1:4
        @test isequal(imfilter_gaussian(A, [0,0]), A)
        @test_approx_eq imfilter_gaussian(A, [0,3]) A  # NaN pattern prevents ≈
        B = copy(A)
        B[isfinite(B)] = 2.5
        @test_approx_eq imfilter_gaussian(A, [10^3,0]) B  # NaN pattern prevents ≈
        @test maximum(map(abs, imfilter_gaussian(imgcol, [10^3,10^3]) - mean(imgcol))) < 1e-4
        @test maximum(map(abs, imfilter_gaussian(imgcolf, [10^3,10^3]) - mean(imgcolf))) < 1e-4
        A = rand(4,5)
        img = reinterpret(Gray{Float64}, A)
        imgf = imfilter_gaussian(img, [2,2])
        @test reinterpret(Float64, data(imgf)) ≈ imfilter_gaussian(A, [2,2])
        A = rand(3,4,5)
        img = colorim(A)
        imgf = imfilter_gaussian(img, [2,2])
        @test channelview(data(imgf)) ≈ imfilter_gaussian(A, [0,2,2])

        # TODO: uncomment these
        # A = zeros(Int, 9, 9); A[5, 5] = 1
        # @test maximum(abs(imfilter_LoG(A, [1,1]) - imlog(1.0))) < EPS
        # @test maximum(imfilter_LoG([0 0 0 0 1 0 0 0 0], [1,1]) - sum(imlog(1.0),1)) < EPS
        # @test maximum(imfilter_LoG([0 0 0 0 1 0 0 0 0]', [1,1]) - sum(imlog(1.0),2)) < EPS

        # @test imaverage() == fill(1/9, 3, 3)
        # @test imaverage([3,3]) == fill(1/9, 3, 3)
        # @test_throws ErrorException imaverage([5])
    end

    @testset "imgradients" begin
        A = rand(5,7)
        for (methstring, meth) in (("sobel", KernelFactors.sobel),
                                   ("prewitt", KernelFactors.prewitt),
                                   ("ando3", KernelFactors.ando3),
                                   ("ando4", KernelFactors.ando4),
                                   ("ando5", KernelFactors.ando5))
            gy, gx = imgradients(A, methstring)
            @test (gy,gx) == imgradients(A, meth, "replicate")
        end
        gy, gx = imgradients(A)
        @test (gy,gx) == imgradients(A, KernelFactors.ando3, "replicate")
        @test_throws ErrorException imgradients(A, "nonsense")
    end

    @testset "extrema_filter" begin
        A = [0.1,0.3,0.2,0.3,0.4]
        minval, maxval = extrema_filter(A, 3)
        @test minval == [0.1,0.2,0.2]
        @test maxval == [0.3,0.3,0.4]
        A = [1 4 7;
             2 5 8;
             3 6 9]
        minval, maxval = extrema_filter(A, [3,3])
        @test minval == [1]'
        @test maxval == [9]'
    end
end

nothing
