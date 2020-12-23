# This is to be used with `@snoopi_deep` from SnoopCompile

using SnoopCompile
using ImageCore
using Statistics

images2d = [rand(Float32, 100, 100),
            rand(Float64, 100, 100),
            rand(Gray{N0f8}, 100, 100),
            rand(Gray{N0f16}, 100, 100),
            rand(RGB{N0f8}, 100, 100),
]

images3d = [rand(Gray{N0f16}, 100, 100, 10),
]

tinf = @snoopi_deep begin
    using ImageFiltering

    Kernel.gaussian((3,3))
    Kernel.gaussian((3.0,3.0))
    Kernel.DoG(3)
    Kernel.DoG(3.0)
    Kernel.Laplacian()
    Kernel.LoG(3)
    Kernel.LoG(3.0)
    Kernel.sobel()

    KernelFactors.gaussian((3,3))
    KernelFactors.gaussian((3.0,3.0))
    KernelFactors.IIRGaussian((3,3))
    KernelFactors.IIRGaussian(3.0)
    KernelFactors.sobel()

    for img in images2d
        for kern in (Kernel.gaussian((3,3)),
                     KernelFactors.gaussian((3,3)),
                     KernelFactors.IIRGaussian((3.0, 3.0)))
            imfilter(img, kern)
        end
        if eltype(img) <: Union{Number,Gray}
            mapwindow(extrema, img, (3,3))
            mapwindow(median!, img, (3,3))
        end
    end

    for img in images3d
        for kern in (Kernel.gaussian((3,3,3)),
                     KernelFactors.gaussian((3,3,3)),
                     KernelFactors.IIRGaussian((3.0, 3.0, 3.0)))
            imfilter(img, kern)
        end
    end
end
