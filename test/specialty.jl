using ImageFiltering, ImageCore, OffsetArrays, Colors, FixedPointNumbers
using Statistics, Test
using ImageFiltering: IdentityUnitRange

@testset "specialty" begin
    @testset "Laplacian" begin
        L1 = OffsetArray([1,-2,1],-1:1)
        L2 = OffsetArray([0 1 0; 1 -4 1; 0 1 0], -1:1, -1:1)
        kern = Kernel.Laplacian((true,))
        @test !isempty(kern)
        @test convert(AbstractArray, kern) == L1
        kern = Kernel.Laplacian()
        @test convert(AbstractArray, kern) == L2
        kern = Kernel.Laplacian((true,false))
        @test convert(AbstractArray, kern) == reshape(L1, -1:1, 0:0)
        kern = Kernel.Laplacian((false,true))
        @test convert(AbstractArray, kern) == reshape(L1, 0:0, -1:1)
        kern = Kernel.Laplacian([1], 2)
        @test convert(AbstractArray, kern) == reshape(L1, -1:1, 0:0)
        kern = Kernel.Laplacian([2], 2)
        @test convert(AbstractArray, kern) == reshape(L1, 0:0, -1:1)
        function makeimpulse(T, sz, x)
            A = zeros(T, sz)
            A[x] = oneunit(T)
            A
        end
        # 1d
        kern = Kernel.Laplacian((true,))
        for a in (makeimpulse(Float64, (5,), 3),
                  makeimpulse(Int, (5,), 3),
                  makeimpulse(UInt32, (5,), 3),
                  makeimpulse(UInt16, (5,), 3),
                  makeimpulse(UInt8, (5,), 3),
                  makeimpulse(Bool, (5,), 3),
                  makeimpulse(Gray{N0f8}, (5,), 3),
                  makeimpulse(RGB{Float32}, (5,), 3),
                  makeimpulse(RGB{N0f8}, (5,), 3))
            af = imfilter(a, kern)
            if eltype(a) == unsigned(Int)
                continue  # the concatenation below fails
            end
            T = eltype(a)
            @test af == [zero(T),a[3],-2a[3],a[3],zero(T)]
            af = imfilter(a, (kern,))
            @test af == [zero(T),a[3],-2a[3],a[3],zero(T)]
        end
        for a in (makeimpulse(Float64, (5,), 1),
                  makeimpulse(Gray{N0f8}, (5,), 1),
                  makeimpulse(RGB{Float32}, (5,), 1))
            for (border, edgecoef) in (("replicate", -1),
                                       (Fill(zero(eltype(a))), -2))
                af = imfilter(a, kern, border)
                T = eltype(a)
                @test af == [edgecoef*a[1],a[1],zero(T),zero(T),zero(T)]
            end
        end
        # 2d
        kern = Kernel.Laplacian((true,true))
        @test convert(AbstractArray, kern) == OffsetArray([0 1 0; 1 -4 1; 0 1 0],-1:1,-1:1)
        for a in (makeimpulse(Float64, (5,5), CartesianIndex((3,3))),
                  makeimpulse(Gray{N0f8}, (5,5), CartesianIndex((3,3))),
                  makeimpulse(RGB{Float32}, (5,5), CartesianIndex((3,3))))
            af = imfilter(a, kern)
            T = eltype(a)
            z = zero(T)
            c = oneunit(T)
            @test af == [z z z z z;
                         z z c z z;
                         z c -4c c z;
                         z z c z z;
                         z z z z z]
        end
        for a in (makeimpulse(Float64, (5,5), CartesianIndex((3,1))),
                  makeimpulse(Gray{N0f8}, (5,5), CartesianIndex((3,1))),
                  makeimpulse(RGB{Float32}, (5,5), CartesianIndex((3,1))))
            for (border, edgecoef) in (("replicate", -3),
                                       (Fill(zero(eltype(a))), -4))
                af = imfilter(a, kern, border)
                T = eltype(a)
                z = zero(T)
                c = oneunit(T)
                @test af == [z z z z z;
                             c z z z z;
                             edgecoef*c c z z z;
                             c z z z z;
                             z z z z z]
            end
        end
        for a in (makeimpulse(Float64, (5,5), CartesianIndex((5,5))),
                  makeimpulse(Gray{N0f8}, (5,5), CartesianIndex((5,5))),
                  makeimpulse(RGB{Float32}, (5,5), CartesianIndex((5,5))))
            for (border, edgecoef) in (("replicate", -2),
                                       (Fill(zero(eltype(a))), -4))
                af = imfilter(a, kern, border)
                T = eltype(a)
                z = zero(T)
                c = oneunit(T)
                @test af == [z z z z z;
                             z z z z z;
                             z z z z z;
                             z z z z c;
                             z z z c edgecoef*c]
            end
        end
        # 2d, but only computing the laplacian along the vertical
        kern = Kernel.Laplacian((true,false))
        @test convert(AbstractArray, kern) == OffsetArray(reshape([1, -2, 1], (3,1)),-1:1,0:0)
        for a in (makeimpulse(Float64, (5,5), CartesianIndex((3,3))),
                  makeimpulse(Gray{N0f8}, (5,5), CartesianIndex((3,3))),
                  makeimpulse(RGB{Float32}, (5,5), CartesianIndex((3,3))))
            af = imfilter(a, kern)
            T = eltype(a)
            z = zero(T)
            c = oneunit(T)
            @test af == [z z z z z;
                         z z c z z;
                         z z -2c z z;
                         z z c z z;
                         z z z z z]
        end
        for a in (makeimpulse(Float64, (5,5), CartesianIndex((3,1))),
                  makeimpulse(Gray{N0f8}, (5,5), CartesianIndex((3,1))),
                  makeimpulse(RGB{Float32}, (5,5), CartesianIndex((3,1))))
            for (border, edgecoef) in (("replicate", -2),
                                       (Fill(zero(eltype(a))), -2))
                af = imfilter(a, kern, border)
                T = eltype(a)
                z = zero(T)
                c = oneunit(T)
                @test af == [z z z z z;
                             c z z z z;
                             edgecoef*c z z z z;
                             c z z z z;
                             z z z z z]
            end
        end
        for a in (makeimpulse(Float64, (5,5), CartesianIndex((5,5))),
                  makeimpulse(Gray{N0f8}, (5,5), CartesianIndex((5,5))),
                  makeimpulse(RGB{Float32}, (5,5), CartesianIndex((5,5))))
            for (border, edgecoef) in (("replicate", -1),
                                       (Fill(zero(eltype(a))), -2))
                af = imfilter(a, kern, border)
                T = eltype(a)
                z = zero(T)
                c = oneunit(T)
                @test af == [z z z z z;
                             z z z z z;
                             z z z z z;
                             z z z z c;
                             z z z z edgecoef*c]
            end
        end
    end

    @testset "gaussian" begin
        function gaussiancmp(σ, xr)
            cmp = [exp(-x^2/(2σ^2)) for x in xr]
            cmp ./ sum(cmp)
        end
        for kern in (Kernel.gaussian(()), Kernel.gaussian((),()))
            @test ndims(kern) == 0
            @test kern[] == 1
            @test ImageFiltering.iscopy(kern)
        end
        for kern in (Kernel.gaussian((1.3,)), Kernel.gaussian((1.3,),(7,)))
            @test kern ≈ gaussiancmp(1.3, axes(kern,1))
        end
        @test KernelFactors.gaussian(2, 9) ≈ gaussiancmp(2, IdentityUnitRange(-4:4))
        k = KernelFactors.gaussian((2,3), (9,7))
        @test vec(k[1]) ≈ gaussiancmp(2, IdentityUnitRange(-4:4))
        @test vec(k[2]) ≈ gaussiancmp(3, IdentityUnitRange(-3:3))
        @test sum(KernelFactors.gaussian(5)) ≈ 1
        for k = (KernelFactors.gaussian((2,3)), KernelFactors.gaussian([2,3]), KernelFactors.gaussian([2,3], [9,7]))
            @test sum(k[1]) ≈ 1
            @test sum(k[2]) ≈ 1
        end
        @test Kernel.gaussian((2,), (9,)) ≈ gaussiancmp(2, IdentityUnitRange(-4:4))
        @test Kernel.gaussian((2,3), (9,7)) ≈ gaussiancmp(2, IdentityUnitRange(-4:4)).*gaussiancmp(3, IdentityUnitRange(-3:3))'
        @test sum(Kernel.gaussian(5)) ≈ 1
        for k = (Kernel.gaussian((2,3)), Kernel.gaussian([2,3]), Kernel.gaussian([2,3], [9,7]))
            @test sum(k) ≈ 1
        end
        # Bug noticed in Images issue #674
        k = KernelFactors.gaussian((3, 3, 0))
        @test k[3].data == OffsetArray([1.0], 0:0)
    end

    @testset "DoG" begin
        function gaussiancmp(σ, xr)
            cmp = [exp(-x^2/(2σ^2)) for x in xr]
            OffsetArray(cmp/sum(cmp), xr)
        end
        @test Kernel.DoG((2,), (3,), (9,)) ≈ gaussiancmp(2, -4:4) - gaussiancmp(3, -4:4)
        k = Kernel.DoG((2,3), (4,3.5), (9,7))
        k1 = gaussiancmp(2, -4:4) .* gaussiancmp(3, -3:3)'
        k2 = gaussiancmp(4, -4:4) .* gaussiancmp(3.5, -3:3)'
        @test k ≈ k1-k2
        @test abs(sum(Kernel.DoG(5))) < 1e-8
        @test Kernel.DoG(5) == Kernel.DoG((5,5))
        @test abs(sum(Kernel.DoG((5,), (7,), (21,)))) < 1e-8
        @test axes(Kernel.DoG((5,), (7,), (21,))) == (-10:10,)
    end

    @testset "LoG" begin
        img = rand(20,21)
        σs = (2.5, 3.2)
        kernel1 = Kernel.LoG(σs)
        kernel2 = (KernelFactors.IIRGaussian(σs)..., Kernel.Laplacian())
        imgf1 = imfilter(img, kernel1)
        imgf2 = imfilter(img, kernel2)
        @test cor(vec(imgf1), vec(imgf2)) > 0.8
        # Ensure that edge-trimming under successive stages of filtering works correctly
        ImageFiltering.fillbuf_nan[] = true
        kernel3 = (Kernel.Laplacian(), KernelFactors.IIRGaussian(σs)...)
        @test !any(isnan, imfilter(img, kernel3))
        ImageFiltering.fillbuf_nan[] = false
        @test Kernel.LoG(2.5) == Kernel.LoG((2.5,2.5))
    end
end

nothing
