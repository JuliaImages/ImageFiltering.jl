using ImageFiltering
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
    grp["median!"] = @benchmarkable mapwindow(median!, $img2d, (5,5))
    grp["mean, small window"] = @benchmarkable mapwindow(mean, $img1d, (3,))
    grp["mean, large window"] = @benchmarkable mapwindow(mean, $img3d, (5,5,5))
    grp["expensive f"] = @benchmarkable mapwindow(x -> quantile(vec(x), 0.7), $img3d, (3,3,3))
end
