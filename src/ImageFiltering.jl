__precompile__()

module ImageFiltering

importall FFTW
using Colors, FixedPointNumbers, ImageCore, MappedArrays, FFTViews, OffsetArrays, StaticArrays, ComputationalResources, TiledIteration
using ColorVectorSpace  # for filtering RGB arrays
using Compat
using Base: Indices, tail, fill_to_length, @pure, depwarn

export Kernel, KernelFactors, Pad, Fill, Inner, NA, NoPad, Algorithm, imfilter, imfilter!, mapwindow, imgradients, padarray, centered, kernelfactors, reflect

# TODO: just use .+
# See https://github.com/JuliaLang/julia/pull/22932#issuecomment-330711997
if VERSION < v"0.7.0-DEV.1759"
    plus(r::AbstractUnitRange, i::Integer) = r + i
else
    plus(r::AbstractUnitRange, i::Integer) = broadcast(+, r, i)
end
plus(a::AbstractArray, x::Number) = a .+ x

FixedColorant{T<:Normed} = Colorant{T}
StaticOffsetArray{T,N,A<:StaticArray} = OffsetArray{T,N,A}
OffsetVector{T} = OffsetArray{T,1}

# Needed for type-stability
function Base.transpose(A::StaticOffsetArray{T,2}) where T
    inds1, inds2 = indices(A)
    OffsetArray(transpose(parent(A)), inds2, inds1)
end

module Algorithm
    # deliberately don't export these, but it's expected that they
    # will be used as Algorithm.FFT(), etc.
    using Compat
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
using .Kernel: Laplacian, reflect

NDimKernel{N,K} = Union{AbstractArray{K,N},ReshapedOneD{K,N},Laplacian{N}}

include("border.jl")

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
end

"""
`KernelFactors` is a module implementing separable filtering kernels,
each stored in terms of their factors. The following kernels are
supported:

  - `sobel`
  - `prewitt`
  - `ando3`, `ando4`, and `ando5` (the latter in 2d only)
  - `gaussian`
  - `IIRGaussian` (approximate gaussian filtering, fast even for large σ)

See also: [`Kernel`](@ref).
"""
KernelFactors

"""
`Kernel` is a module implementing filtering kernels of full
dimensionality. The following kernels are supported:

  - `sobel`
  - `prewitt`
  - `ando3`, `ando4`, and `ando5`
  - `gaussian`
  - `DoG` (Difference-of-Gaussian)
  - `LoG` (Laplacian-of-Gaussian)
  - `Laplacian`

See also: [`KernelFactors`](@ref).
"""
Kernel

end # module
