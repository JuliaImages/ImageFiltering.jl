using PrecompileTools

@setup_workload begin
    images2d = Any[
        rand(Float32, 100, 100),
        rand(Float64, 100, 100),
        rand(Gray{N0f8}, 100, 100),
        rand(Gray{N0f16}, 100, 100),
        rand(RGB{N0f8}, 100, 100),
    ]
    images3d = Any[
        rand(Gray{N0f16}, 100, 100, 10),
    ]
    # We could put the kernels and resources here too, but it makes sense to precompile them
    @compile_workload begin
        kernels2d = Any[
            Kernel.gaussian((3, 3)),
            KernelFactors.gaussian((3, 3)),
            KernelFactors.gaussian((3.0f0, 3.0f0)),
            KernelFactors.IIRGaussian((3.0, 3.0)),
        ]
        kernels3d = Any[
            Kernel.gaussian((3, 3, 3)),
            KernelFactors.gaussian((3, 3, 3)),
            KernelFactors.gaussian((3.0f0, 3.0f0, 3.0f0)),
            KernelFactors.IIRGaussian((3.0, 3.0, 3.0)),
        ]
        resources2d = Any[CPU1(FIR()), CPUThreads(FIRTiled((15,15))), CPU1(FFT())]
        resources3d = Any[CPU1(FIR()), CPUThreads(FIRTiled((15,15,15))), CPU1(FFT())]

        for img in images2d
            for kern in kernels2d
                for r in resources2d
                    imfilter(r, img, kern)
                end
                imfilter(img, kern)
            end
            if eltype(img) <: Union{Number,Gray}
                mapwindow(extrema, img, (3, 3))
                mapwindow(median!, img, (3, 3))
            end
        end
        for img in images3d
            for kern in kernels3d
                for r in resources3d
                    imfilter(r, img, kern)
                end
                imfilter(img, kern)
            end
        end
        # Sliding DCT/DST kernels are long, so they are not suited to the (tiled) resources above
        for img in images2d
            for kern in (KernelFactors.DCTGaussian((3.0, 3.0)),
                         KernelFactors.slidinggaussian((3.0, 3.0); order = (1, 0)),
                         KernelFactors.DCTLoG(3.0),
                         KernelFactors.DCTDoG(3.0))
                imfilter(img, kern)
            end
        end
        for img in images3d
            imfilter(img, KernelFactors.DCTGaussian((3.0, 3.0, 3.0)))
        end
    end
end
