module ImagesFiltering

using Colors, FixedPointNumbers, ImagesCore, MappedArrays, FFTViews, OffsetArrays, StaticArrays, ComputationalResources
using ColorVectorSpace  # in case someone filters RGB arrays
using Base: Indices, tail, fill_to_length, @pure

export Kernel, KernelFactors, Pad, Fill, Inner, NoPad, Algorithm, imfilter, imfilter!, padarray, centered

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

include("utils.jl")
include("kernelfactors.jl")
using .KernelFactors: TriggsSdika, IIRFilter, ReshapedVector, iterdims

typealias ArrayLike{T} Union{AbstractArray{T}, IIRFilter{T}, ReshapedVector{T}}
typealias ReshapedTriggsSdika{T,N,Npre,V<:TriggsSdika} ReshapedVector{T,N,Npre,V}
typealias AnyTriggs Union{TriggsSdika, ReshapedTriggsSdika}

include("kernel.jl")
using .Kernel
using .Kernel: Laplacian

typealias NDimKernel{N,K} Union{AbstractArray{K,N},ReshapedVector{K,N},Laplacian{N}}

include("border.jl")

typealias BorderSpec{Style,T} Union{Pad{Style,0}, Fill{T,0}, Inner{0}}
typealias BorderSpecRF{T} Union{Pad{:replicate,0}, Fill{T,0}}
typealias BorderSpecNoNa{T} Union{Pad{:replicate,0}, Pad{:circular,0}, Pad{:symmetric,0},
                                  Pad{:reflect,0}, Fill{T,0}, Inner{0}}
typealias BorderSpecAny Union{BorderSpec,NoPad}

include("deprecated.jl")

typealias ProcessedKernel Tuple

include("imfilter.jl")
include("specialty.jl")

function __init__()
    # See ComputationalResources README for explanation
    push!(LOAD_PATH, dirname(@__FILE__))
    # if haveresource(ArrayFireLibs)
    #     @eval using DummyAF
    # end
    pop!(LOAD_PATH)
end

end # module
