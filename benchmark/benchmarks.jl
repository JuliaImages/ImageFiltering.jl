using ImageFiltering, ImageCore
using PkgBenchmark
using BenchmarkTools
using Statistics: quantile, mean, median!

function makeimages(sz)
    imgF32      = rand(Float32, sz)
    imgN0f8     = Array(rand(N0f8, sz))
    imggrayF32  = Array(rand(Gray{Float32}, sz))
    imggrayN0f8 = Array(rand(Gray{N0f8}, sz))
    imgrgbF32   = Array(rand(RGB{Float32}, sz))
    imgrgbN0f8  = Array(rand(RGB{N0f8}, sz))
    return ("F32"=>imgF32, "N0f8"=>imgN0f8, "GrayF32"=>imggrayF32,
            "GrayN0f8"=>imggrayN0f8, "RGBF32"=>imgrgbF32, "RGBN0f8"=>imgrgbN0f8)
end

sz2str(sz) = join(map(string, sz), '×')

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

SUITE["imfilter"] = BenchmarkGroup()
let grp = SUITE["imfilter"]
    kerninsep   = (centered([-1, 0, 1]),
                   centered([ 1/5  1/4  1/7;
                              1/2  1/3 -1/11;
                             -1/25 1/9 -1/7]),   # has full rank so won't be factored
                   centered(rand(3, 3, 3)))
    for sz in ((100, 100), (2048, 2048), (2048,), (100, 100, 100))
        for (aname, img) in makeimages(sz)
            trues = map(i->true, sz)
            twos  = map(i->2, sz)
            szstr = sz2str(sz)
            kerniir = KernelFactors.IIRGaussian(Float32.(twos))
            for (kname, kern) in zip(("densesmall", "denselarge", "factored"),
                                      (kerninsep[length(sz)], Kernel.DoG(twos), KernelFactors.sobel(trues, 1)[1]))
                grp[kname*"_"*aname*"_"*szstr]  = @benchmarkable imfilter($img, ($kern,), "replicate", ImageFiltering.FIR())
            end
            grp["IIRGaussian_"*aname*"_"*szstr] = @benchmarkable imfilter($img, $kerniir, "replicate", ImageFiltering.IIR())
            grp["FFT_"*aname*"_"*szstr]         = @benchmarkable imfilter($img, $(Kernel.DoG(twos),), "replicate", ImageFiltering.FFT())
        end
    end
end
