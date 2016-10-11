__precompile__()

module ImageFiltering

using Colors, FixedPointNumbers, ImageCore, MappedArrays, FFTViews, OffsetArrays, StaticArrays, ComputationalResources, TiledIteration
using ColorVectorSpace  # for filtering RGB arrays
using Base: Indices, tail, fill_to_length, @pure, depwarn

export Kernel, KernelFactors, Pad, Fill, Inner, NA, NoPad, Algorithm, imfilter, imfilter!, mapwindow, imgradients, padarray, centered, kernelfactors, reflect

typealias FixedColorant{T<:Normed} Colorant{T}
typealias StaticOffsetArray{T,N,A<:StaticArray} OffsetArray{T,N,A}
typealias OffsetVector{T} OffsetArray{T,1}

# Needed for type-stability
function Base.transpose{T}(A::StaticOffsetArray{T,2})
    inds1, inds2 = indices(A)
    OffsetArray(transpose(parent(A)), inds2, inds1)
end

module Algorithm
    # deliberately don't export these, but it's expected that they
    # will be used as Algorithm.FFT(), etc.
    abstract Alg
    "Filter using the Fast Fourier Transform" immutable FFT <: Alg end
    "Filter using a direct algorithm" immutable FIR <: Alg end
    "Cache-efficient filtering using tiles" immutable FIRTiled{N} <: Alg
        tilesize::Dims{N}
    end
    "Filter with an Infinite Impulse Response filter" immutable IIR <: Alg end
    "Filter with a cascade of mixed types (IIR, FIR)" immutable Mixed <: Alg end

    FIRTiled() = FIRTiled(())
end
using .Algorithm: Alg, FFT, FIR, FIRTiled, IIR, Mixed

Alg{A<:Alg}(r::AbstractResource{A}) = r.settings

include("utils.jl")
include("kernelfactors.jl")
using .KernelFactors: TriggsSdika, IIRFilter, ReshapedOneD, iterdims, kernelfactors

typealias ReshapedVector{T,N,Npre,V<:AbstractVector} ReshapedOneD{T,N,Npre,V}
typealias ArrayType{T} Union{AbstractArray{T}, ReshapedVector{T}}
typealias ReshapedIIR{T,N,Npre,V<:IIRFilter} ReshapedOneD{T,N,Npre,V}
typealias AnyIIR{T} Union{IIRFilter{T}, ReshapedIIR{T}}
typealias ArrayLike{T} Union{ArrayType{T}, AnyIIR{T}}

include("kernel.jl")
using .Kernel
using .Kernel: Laplacian, reflect

typealias NDimKernel{N,K} Union{AbstractArray{K,N},ReshapedOneD{K,N},Laplacian{N}}

include("border.jl")

typealias BorderSpec{T} Union{Pad{0}, Fill{T,0}, Inner{0}}
typealias BorderSpecNoNa{T} Union{Pad{0}, Fill{T,0}, Inner{0}}
typealias BorderSpecAny Union{BorderSpec,NA,NoPad}

include("deprecated.jl")

typealias ProcessedKernel Tuple

include("imfilter.jl")
include("specialty.jl")

include("mapwindow.jl")
using .MapWindow

function __init__()
    # See ComputationalResources README for explanation
    push!(LOAD_PATH, dirname(@__FILE__))
    # if haveresource(ArrayFireLibs)
    #     @eval using DummyAF
    # end
    pop!(LOAD_PATH)
end

end # module
