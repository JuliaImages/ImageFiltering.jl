using ImageFiltering
using TestImages, ImageTransformations
using BenchmarkTools

const SUITE = BenchmarkGroup(["ImageFIltering"])

include("utils.jl")
include("imfilter.jl")
