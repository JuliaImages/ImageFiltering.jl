module ImageFiltering

using FFTW
using ImageCore, FFTViews, OffsetArrays, StaticArrays, ComputationalResources, TiledIteration
# Where possible we avoid a direct dependency to reduce the number of [compat] bounds
# using FixedPointNumbers: Normed, N0f8 # reexported by ImageCore
using ImageCore.MappedArrays
using Statistics, LinearAlgebra
using Base: Indices, tail, fill_to_length, @pure, depwarn, @propagate_inbounds
using OffsetArrays: IdentityUnitRange   # using the one in OffsetArrays makes this work with multiple Julia versions
using SparseArrays   # only needed to fix an ambiguity in borderarray
using Reexport

@reexport using OffsetArrays: centered # this method once lived here

if supertype(TileBuffer) === DenseArray
    # TiledIteration >= 0.4
    tilebuf_parent(tilebuf) = parent(parent(tilebuf))
else
    # TiledIteration < 0.4
    tilebuf_parent(tilebuf) = parent(tilebuf)
end

export Kernel, KernelFactors,
    Pad, Fill, Inner, NA, NoPad,
    BorderArray,
    Algorithm,
    imfilter, imfilter!,
    mapwindow, mapwindow!,
    imgradients, padarray, centered, kernelfactors, reflect,
    freqkernel, spacekernel,
    findlocalminima, findlocalmaxima,
    blob_LoG, BlobLoG

const FixedColorant{T<:Normed} = Colorant{T}
const StaticOffsetArray{T,N,A<:StaticArray} = OffsetArray{T,N,A}
const OffsetVector{T} = OffsetArray{T,1}

# Add a fix that should have been included in julia-1.0.3
if isdefined(Broadcast, :_sametype) && !isdefined(Broadcast, :axistype)
    axistype(a::T, b::T) where T = a
    axistype(a, b) = UnitRange{Int}(a)
    Broadcast._bcs1(a, b) = Broadcast._bcsm(b, a) ? axistype(b, a) : (Broadcast._bcsm(a, b) ? axistype(a, b) : throw(DimensionMismatch("arrays could not be broadcast to a common size")))
end

# Needed for type-stability
function Base.transpose(A::StaticOffsetArray{T,2}) where T
    inds1, inds2 = axes(A)
    OffsetArray(transpose(parent(A)), inds2, inds1)
end

module Algorithm
    # deliberately don't export these, but it's expected that they
    # will be used as Algorithm.FFT(), etc.
    abstract type Alg end
    "Filter using the Fast Fourier Transform" struct FFT <: Alg end
    "Filter using a direct algorithm" struct FIR <: Alg end
    "Cache-efficient filtering using tiles" struct FIRTiled{N} <: Alg
        tilesize::Dims{N}
    end
    "Filter with an Infinite Impulse Response filter" struct IIR <: Alg end
    "Filter with a cascade of mixed types (IIR, FIR)" struct Mixed <: Alg end

    FIRTiled() = FIRTiled(())
end
using .Algorithm: Alg, FFT, FIR, FIRTiled, IIR, Mixed

Algorithm.Alg(r::AbstractResource{A}) where {A<:Alg} = r.settings

include("utils.jl")
include("compat.jl")
include("kernelfactors.jl")
using .KernelFactors: TriggsSdika, IIRFilter, ReshapedOneD, iterdims, kernelfactors

const ReshapedVector{T,N,Npre,V<:AbstractVector} = ReshapedOneD{T,N,Npre,V}
const ArrayType{T} = Union{AbstractArray{T}, ReshapedVector{T}}
const ReshapedIIR{T,N,Npre,V<:IIRFilter} = ReshapedOneD{T,N,Npre,V}
const AnyIIR{T} = Union{IIRFilter{T}, ReshapedIIR{T}}
const ArrayLike{T} = Union{ArrayType{T}, AnyIIR{T}}

include("kernel.jl")
using .Kernel
using .Kernel: Laplacian, reflect, ando3, ando4, ando5, scharr, bickley, prewitt, sobel, gabor, moffat, butterworth

const NDimKernel{N,K} = Union{AbstractArray{K,N},ReshapedOneD{K,N},Laplacian{N}}

include("border.jl")
include("borderarray.jl")

const BorderSpec{T} = Union{Pad{0}, Fill{T,0}, Inner{0}}
const BorderSpecNoNa{T} = Union{Pad{0}, Fill{T,0}, Inner{0}}
const BorderSpecAny = Union{BorderSpec,NA,NoPad}

const ProcessedKernel = Tuple

include("imfilter.jl")
include("specialty.jl")

include("mapwindow.jl")
using .MapWindow
include("extrema.jl")

include("models.jl")

function __init__()
    # See ComputationalResources README for explanation
    push!(LOAD_PATH, dirname(@__FILE__))
    # if haveresource(ArrayFireLibs)
    #     @eval using DummyAF
    # end
    pop!(LOAD_PATH)
end

include("precompile.jl")

end # module
