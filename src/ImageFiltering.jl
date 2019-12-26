module ImageFiltering

using FFTW
using ImageCore, MappedArrays, FFTViews, OffsetArrays, StaticArrays, ComputationalResources, TiledIteration
using Statistics, LinearAlgebra
using ColorVectorSpace  # for filtering RGB arrays
using Base: Indices, tail, fill_to_length, @pure, depwarn, @propagate_inbounds
using OffsetArrays: IdentityUnitRange   # using the one in OffsetArrays makes this work with multiple Julia versions
using Requires

export Kernel, KernelFactors,
    Pad, Fill, Inner, NA, NoPad,
    BorderArray,
    Algorithm,
    imfilter, imfilter!,
    mapwindow, mapwindow!,
    imgradients, padarray, centered, kernelfactors, reflect,
    freqkernel, spacekernel

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

Alg(r::AbstractResource{A}) where {A<:Alg} = r.settings

include("utils.jl")
include("kernelfactors.jl")
using .KernelFactors: TriggsSdika, IIRFilter, ReshapedOneD, iterdims, kernelfactors

ReshapedVector{T,N,Npre,V<:AbstractVector} = ReshapedOneD{T,N,Npre,V}
ArrayType{T} = Union{AbstractArray{T}, ReshapedVector{T}}
ReshapedIIR{T,N,Npre,V<:IIRFilter} = ReshapedOneD{T,N,Npre,V}
AnyIIR{T} = Union{IIRFilter{T}, ReshapedIIR{T}}
ArrayLike{T} = Union{ArrayType{T}, AnyIIR{T}}

include("kernel.jl")
using .Kernel
using .Kernel: Laplacian, reflect, ando3, ando4, ando5, scharr, bickley, prewitt, sobel, gabor

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

function __init__()
    # See ComputationalResources README for explanation
    push!(LOAD_PATH, dirname(@__FILE__))
    # if haveresource(ArrayFireLibs)
    #     @eval using DummyAF
    # end
    pop!(LOAD_PATH)
    @require AxisArrays="39de3d68-74b9-583c-8d2d-e117c070f3a9" begin
        centered(ax::AxisArrays.Axis{name}) where name = AxisArrays.Axis{name}(centered(ax.val))
        centered(a::AxisArrays.AxisArray) = AxisArrays.AxisArray(centered(a.data), centered.(AxisArrays.axes(a)))
    end
    @require ImageMetadata="bc367c6b-8a6b-528e-b4bd-a4b897500b49" begin
        centered(a::ImageMetadata.ImageMeta) = ImageMetadata.ImageMeta(centered(a.data), ImageMetadata.properties(a))
    end
end

end # module
