module ImagesFiltering

using Colors, FixedPointNumbers, ImagesCore, MappedArrays, FFTViews, StaticArrays, ComputationalResources
using ColorVectorSpace  # in case someone filters RGB arrays

export Kernel, Pad, Fill, Inner, Algorithm, imfilter, imfilter!, padarray, centered

typealias FixedColorant{T<:UFixed} Colorant{T}

module Algorithm
    # deliberately don't export these, but it's expected that they
    # will be used as Algorithm.FFT(), etc.
    abstract Alg
    immutable FFT <: Alg end
    immutable FIR <: Alg end
    immutable IIR <: Alg end
end
using .Algorithm: Alg, FFT, FIR, IIR

Alg{A<:Alg}(r::AbstractResource{A}) = r.settings

include("kernel.jl")
using .Kernel
using .Kernel: TriggsSdika, IIRFilter
include("border.jl")
include("deprecated.jl")
include("imfilter.jl")

function __init__()
    # See ComputationalResources README for explanation
    push!(LOAD_PATH, dirname(@__FILE__))
    # if haveresource(ArrayFireLibs)
    #     @eval using DummyAF
    # end
    pop!(LOAD_PATH)
end

end # module
