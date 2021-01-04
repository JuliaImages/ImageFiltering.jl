if ccall(:jl_generating_output, Cint, ()) == 1
    let
        images2d = [rand(Float32, 100, 100),
                    rand(Float64, 100, 100),
                    rand(Gray{N0f8}, 100, 100),
                    rand(Gray{N0f16}, 100, 100),
                    rand(RGB{N0f8}, 100, 100),
        ]
        images3d = [rand(Gray{N0f16}, 100, 100, 10),
        ]

        for img in images2d
            for kern in (Kernel.gaussian((3,3)),
                        KernelFactors.gaussian((3,3)),
                        KernelFactors.IIRGaussian((3.0, 3.0)))
                for r in (CPU1(FIR()), CPUThreads(FIRTiled((5,5))), CPU1(FFT()))
                    @assert precompile(imfilter, (typeof(r), typeof(img), typeof(kern)))
                end
                @assert precompile(imfilter, (typeof(img), typeof(kern)))
            end
            if eltype(img) <: Union{Number,Gray}
                @assert precompile(mapwindow, (typeof(extrema), typeof(img), typeof((3,3))))
                @assert precompile(mapwindow, (typeof(median!), typeof(img), typeof((3,3))))
            end
        end
        for img in images3d
            for kern in (Kernel.gaussian((3,3,3)),
                         KernelFactors.gaussian((3,3,3)),
                         KernelFactors.IIRGaussian((3.0, 3.0, 3.0)))
                for r in (CPU1(FIR()), CPUThreads(FIRTiled((5,5))), CPU1(FFT()))
                    @assert precompile(imfilter, (typeof(r), typeof(img), typeof(kern)))
                end
                @assert precompile(imfilter, (typeof(img), typeof(kern)))
            end
        end
    end
end
