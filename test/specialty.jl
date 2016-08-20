using ImagesFiltering, ImagesCore, OffsetArrays, Colors
using Base.Test

@testset "specialty" begin
    @testset "Laplacian" begin
        function makeimpulse(T, sz, x)
            A = zeros(T, sz)
            A[x] = one(T)
            A
        end
        # 1d
        kern = Kernel.Laplacian((true,))
        @test convert(AbstractArray, kern) == OffsetArray([1,-2,1],-1:1)
        for a in (makeimpulse(Float64, (5,), 3),
                  makeimpulse(Gray{U8}, (5,), 3),
                  makeimpulse(RGB{Float32}, (5,), 3))
            af = imfilter(a, kern)
            T = eltype(a)
            @test af == [zero(T),a[3],-2a[3],a[3],zero(T)]
            af = imfilter(a, (kern,))
            @test af == [zero(T),a[3],-2a[3],a[3],zero(T)]
        end
        for a in (makeimpulse(Float64, (5,), 1),
                  makeimpulse(Gray{U8}, (5,), 1),
                  makeimpulse(RGB{Float32}, (5,), 1))
            for (border, edgecoef) in ((Pad{:replicate}(), -1),
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
                  makeimpulse(Gray{U8}, (5,5), CartesianIndex((3,3))),
                  makeimpulse(RGB{Float32}, (5,5), CartesianIndex((3,3))))
            af = imfilter(a, kern)
            T = eltype(a)
            z = zero(T)
            c = one(T)
            @test af == [z z z z z;
                         z z c z z;
                         z c -4c c z;
                         z z c z z;
                         z z z z z]
        end
        for a in (makeimpulse(Float64, (5,5), CartesianIndex((3,1))),
                  makeimpulse(Gray{U8}, (5,5), CartesianIndex((3,1))),
                  makeimpulse(RGB{Float32}, (5,5), CartesianIndex((3,1))))
            for (border, edgecoef) in ((Pad{:replicate}(), -3),
                                       (Fill(zero(eltype(a))), -4))
                af = imfilter(a, kern, border)
                T = eltype(a)
                z = zero(T)
                c = one(T)
                @test af == [z z z z z;
                             c z z z z;
                             edgecoef*c c z z z;
                             c z z z z;
                             z z z z z]
            end
        end
        for a in (makeimpulse(Float64, (5,5), CartesianIndex((5,5))),
                  makeimpulse(Gray{U8}, (5,5), CartesianIndex((5,5))),
                  makeimpulse(RGB{Float32}, (5,5), CartesianIndex((5,5))))
            for (border, edgecoef) in ((Pad{:replicate}(), -2),
                                       (Fill(zero(eltype(a))), -4))
                af = imfilter(a, kern, border)
                T = eltype(a)
                z = zero(T)
                c = one(T)
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
                  makeimpulse(Gray{U8}, (5,5), CartesianIndex((3,3))),
                  makeimpulse(RGB{Float32}, (5,5), CartesianIndex((3,3))))
            af = imfilter(a, kern)
            T = eltype(a)
            z = zero(T)
            c = one(T)
            @test af == [z z z z z;
                         z z c z z;
                         z z -2c z z;
                         z z c z z;
                         z z z z z]
        end
        for a in (makeimpulse(Float64, (5,5), CartesianIndex((3,1))),
                  makeimpulse(Gray{U8}, (5,5), CartesianIndex((3,1))),
                  makeimpulse(RGB{Float32}, (5,5), CartesianIndex((3,1))))
            for (border, edgecoef) in ((Pad{:replicate}(), -2),
                                       (Fill(zero(eltype(a))), -2))
                af = imfilter(a, kern, border)
                T = eltype(a)
                z = zero(T)
                c = one(T)
                @test af == [z z z z z;
                             c z z z z;
                             edgecoef*c z z z z;
                             c z z z z;
                             z z z z z]
            end
        end
        for a in (makeimpulse(Float64, (5,5), CartesianIndex((5,5))),
                  makeimpulse(Gray{U8}, (5,5), CartesianIndex((5,5))),
                  makeimpulse(RGB{Float32}, (5,5), CartesianIndex((5,5))))
            for (border, edgecoef) in ((Pad{:replicate}(), -1),
                                       (Fill(zero(eltype(a))), -2))
                af = imfilter(a, kern, border)
                T = eltype(a)
                z = zero(T)
                c = one(T)
                @test af == [z z z z z;
                             z z z z z;
                             z z z z z;
                             z z z z c;
                             z z z z edgecoef*c]
            end
        end
    end
end

nothing
