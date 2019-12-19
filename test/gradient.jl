using ImageFiltering, ImageCore, ColorVectorSpace
using Test

@testset "gradient" begin
    y, x = 1:5, (1:7)'
    img_grads = ((y .* fill(1, size(x)), 1, 0),
            (fill(1, size(y)) .* x, 0, 1),
            (map(Gray, y.*fill(1.0, size(x))), Gray(1), Gray(0)),
            (map(v->RGB(v,0,0), y.*fill(1.0, size(x))), RGB(1,0,0), RGB(0,0,0)))
    for (img, ey, ex) in img_grads
        for kernelfunc in (KernelFactors.ando3, KernelFactors.sobel, KernelFactors.prewitt,
                           KernelFactors.ando4, KernelFactors.ando5, KernelFactors.bickley,
                           KernelFactors.scharr,
                           Kernel.ando3, Kernel.sobel, Kernel.prewitt,
                           Kernel.ando4, Kernel.ando5, Kernel.scharr, Kernel.bickley)
            gy, gx = @inferred(imgradients(img, kernelfunc, Inner()))
            for val in gy
                @test abs(val - ey) < 1e-4
            end
            for val in gx
                @test abs(val - ex) < 1e-4
            end
            gy, gx = imgradients(img, kernelfunc, Pad(:replicate))
            @test axes(gy) == axes(gx) == axes(img)
        end
        for funcpairs in ((Kernel.ando3, KernelFactors.ando3),
                          (Kernel.sobel, KernelFactors.sobel),
                          (Kernel.prewitt, KernelFactors.prewitt),
                          (Kernel.scharr, KernelFactors.scharr),
                          (Kernel.bickley, KernelFactors.bickley))
            fk, fkf = funcpairs
            ky, kx = fk()
            gmy, gmx = imfilter(img, ky), imfilter(img, kx)
            gy, gx = imgradients(img, fkf)
            @test isapprox(gmy, gy; atol=1e-8)
            @test isapprox(gmx, gx; atol=1e-8)
        end
        for fk in (Kernel.ando4, Kernel.ando5)
            ky, kx = fk()
            gy, gx = imfilter(img, ky, Inner()), imfilter(img, kx, Inner())
            for val in gy
                @test abs(val - ey) < 1e-4
            end
            for val in gx
                @test abs(val - ex) < 1e-4
            end
        end
    end
    # 3d
    y, x, z = 1:5, (1:7)', reshape(1:6, 1, 1, 6)
    for (img, ey, ex, ez) in ((y .* fill(1, size(x)) .* fill(1, size(z)), 1, 0, 0),
                              (fill(1, size(y)) .* x .* fill(1, size(z)), 0, 1, 0),
                              (fill(1, size(y)) .* fill(1, size(x)) .* z, 0, 0, 1))
        for kernelfunc in (KernelFactors.ando3, KernelFactors.sobel, KernelFactors.prewitt,
                           KernelFactors.scharr, KernelFactors.bickley,
                           Kernel.ando3, Kernel.sobel, Kernel.prewitt,
                           Kernel.scharr, Kernel.bickley)
            gy, gx, gz = @inferred(imgradients(img, kernelfunc, Inner()))
            for val in gy
                @test abs(val - ey) < 1e-4
            end
            for val in gx
                @test abs(val - ex) < 1e-4
            end
            for val in gz
                @test abs(val - ez) < 1e-4
            end
        end
    end
end

nothing
