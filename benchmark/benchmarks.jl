using ImageFiltering, ImageCore
using PkgBenchmark
using BenchmarkTools
using Statistics: quantile, mean, median!

SUITE = BenchmarkGroup()
SUITE["mapwindow"] = BenchmarkGroup()

let grp = SUITE["mapwindow"]
    img1d = randn(1000)
    img2d = randn(30,30)
    img3d = randn(10,11,12)
    grp["cheap f, tiny window"] = @benchmarkable mapwindow(first, $img1d, (1,))
    grp["extrema"] = @benchmarkable mapwindow(extrema, $img2d, (5,5))
    grp["maximum"] = @benchmarkable mapwindow(maximum, $img2d, (5,5))
    grp["minimum"] = @benchmarkable mapwindow(minimum, $img2d, (5,5))
    grp["median!"] = @benchmarkable mapwindow(median!, $img2d, (5,5))
    grp["mean, small window"] = @benchmarkable mapwindow(mean, $img1d, (3,))
    grp["mean, large window"] = @benchmarkable mapwindow(mean, $img3d, (5,5,5))
    grp["expensive f"] = @benchmarkable mapwindow(x -> quantile(vec(x), 0.7), $img3d, (3,3,3))
end

SUITE["2d"] = BenchmarkGroup()
let grp = SUITE["2d"]
    imgF32      = rand(Float32, 100, 100)
    imgN0f8     = Array(rand(N0f8, 100, 100))
    imggrayF32  = Array(rand(Gray{Float32}, 100, 100))
    imggrayN0f8 = Array(rand(Gray{N0f8}, 100, 100))
    imgrgbF32   = Array(rand(RGB{Float32}, 100, 100))
    imgrgbN0f8  = Array(rand(RGB{N0f8}, 100, 100))
    kern3x3     = (centered([ 1/5  1/4  1/7;
                              1/2  1/3 -1/11;
                             -1/25 1/9 -1/7]),)
    kernsobel   = KernelFactors.sobel()[1]
    kerniir     = KernelFactors.IIRGaussian(Float32.((5.2, 1.3)))
    for (name, img) in zip(("F32", "N0f8", "GrayF32", "GrayN0f8", "RGBF32", "RGBN0f8"),
                            (imgF32, imgN0f8, imggrayF32, imggrayN0f8, imgrgbF32, imgrgbN0f8))
        for (kname, kern) in zip(("dense", "factored"), (kern3x3, kernsobel))
            grp[kname*"_"*name] = @benchmarkable imfilter($img, $kern, "replicate", ImageFiltering.FIR())
        end
    end
    for (name, img) in zip(("F32", "N0f8", "GrayF32", "GrayN0f8", "RGBF32", "RGBN0f8"),
                            (imgF32, imgN0f8, imggrayF32, imggrayN0f8, imgrgbF32, imgrgbN0f8))
        grp["IIRGaussian_"*name] = @benchmarkable imfilter($img, $kerniir, "replicate", ImageFiltering.IIR())
    end
end
