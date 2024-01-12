module ImageFiltering

using FFTW
using RealFFTs
using ImageCore, FFTViews, OffsetArrays, StaticArrays, ComputationalResources, TiledIteration
# Where possible we avoid a direct dependency to reduce the number of [compat] bounds
# using FixedPointNumbers: Normed, N0f8 # reexported by ImageCore
using ImageCore.MappedArrays
using Statistics, LinearAlgebra
using Base: Indices, tail, fill_to_length, @pure, depwarn, @propagate_inbounds
import Base: copy!
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
    blob_LoG, BlobLoG,
    planned_fft

FixedColorant{T<:Normed} = Colorant{T}
StaticOffsetArray{T,N,A<:StaticArray} = OffsetArray{T,N,A}
OffsetVector{T} = OffsetArray{T,1}

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
    import FFTW
    # deliberately don't export these, but it's expected that they
    # will be used as Algorithm.FFT(), etc.
    abstract type Alg end
    "Filter using the Fast Fourier Transform" struct FFT <: Alg
        plan1::Union{Function,Nothing}
        plan2::Union{Function,Nothing}
        plan3::Union{Function,Nothing}
    end
    FFT() = FFT(nothing, nothing, nothing)
    function Base.show(io::IO, alg::FFT)
        if alg.plan1 === nothing && alg.plan2 === nothing && alg.plan3 === nothing
            print(io, "FFT()")
        else
            print(io, "FFT(planned)")
        end
    end
    "Filter using a direct algorithm" struct FIR <: Alg end
    Base.show(io::IO, alg::FIR) = print(io, "FIR()")
    "Cache-efficient filtering using tiles" struct FIRTiled{N} <: Alg
        tilesize::Dims{N}
    end
    Base.show(io::IO, ::FIRTiled{N}) where N = print(io, "FIRTiled{$N}()")
    "Filter with an Infinite Impulse Response filter" struct IIR <: Alg end
    Base.show(io::IO, alg::IIR) = print(io, "IIR()")
    "Filter with a cascade of mixed types (IIR, FIR)" struct Mixed <: Alg end
    Base.show(io::IO, alg::Mixed) = print(io, "Mixed()")

    FIRTiled() = FIRTiled(())
end
using .Algorithm: Alg, FFT, FIR, FIRTiled, IIR, Mixed

Algorithm.Alg(r::AbstractResource{A}) where {A<:Alg} = r.settings

include("utils.jl")
include("compat.jl")
include("kernelfactors.jl")
using .KernelFactors: TriggsSdika, IIRFilter, ReshapedOneD, iterdims, kernelfactors

ReshapedVector{T,N,Npre,V<:AbstractVector} = ReshapedOneD{T,N,Npre,V}
ArrayType{T} = Union{AbstractArray{T}, ReshapedVector{T}}
ReshapedIIR{T,N,Npre,V<:IIRFilter} = ReshapedOneD{T,N,Npre,V}
AnyIIR{T} = Union{IIRFilter{T}, ReshapedIIR{T}}
ArrayLike{T} = Union{ArrayType{T}, AnyIIR{T}}

include("kernel.jl")
using .Kernel
using .Kernel: Laplacian, reflect, ando3, ando4, ando5, scharr, bickley, prewitt, sobel, gabor, moffat

NDimKernel{N,K} = Union{AbstractArray{K,N},ReshapedOneD{K,N},Laplacian{N}}

include("border.jl")
include("borderarray.jl")

BorderSpec{T} = Union{Pad{0}, Fill{T,0}, Inner{0}}
BorderSpecNoNa{T} = Union{Pad{0}, Fill{T,0}, Inner{0}}
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

if Base.VERSION >= v"1.4.2"
    include("precompile.jl")
end

end # module
