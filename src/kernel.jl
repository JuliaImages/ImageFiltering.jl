"""
`Kernel` is a module implementing filtering kernels of full
dimensionality. The following kernels are supported:

  - `sobel`
  - `prewitt`
  - `ando3`, `ando4`, and `ando5`
  - `scharr`
  - `bickley`
  - `gaussian`
  - `DoG` (Difference-of-Gaussian)
  - `LoG` (Laplacian-of-Gaussian)
  - `Laplacian`

See also: [`KernelFactors`](@ref).
"""
module Kernel

using StaticArrays, OffsetArrays
using ..ImageFiltering: centered, KernelFactors

# We would like to do `using ..ImageFiltering.imgradients` so that that
# Documenter.jl (the documentation system) can parse a reference such as `See
# also: [`ImageFiltering.imgradients`](@ref)`. However, imgradients is not yet
# in scope because of the order in which include files are included into
# ImageFiltering.jl. With the more general `using ImageFiltering`, we seem to
# sidestep the scope problem, although I don't actually understand the mechanism
# form why this works. - ZS
using ImageFiltering
import ..ImageFiltering: _reshape

function product2d(kf)
    k1, k2 = kf
    k1[1].*k1[2], k2[1].*k2[2]
end

"""
```julia
    diff1, diff2 = sobel()
```

Return ``3 \\times 3`` kernels for two-dimensional gradient compution using the
Sobel operator. The `diff1` kernel computes the gradient along the y-axis (first
dimension), and the `diff2` kernel computes the gradient along the x-axis
(second dimension).

# Citation
P.-E. Danielsson and O. Seger, "Generalized and separable sobel operators," in  *Machine Vision for Three-Dimensional Scenes*,  H. Freeman, Ed.  Academic Press, 1990,  pp. 347–379. [doi:10.1016/b978-0-12-266722-0.50016-6](http://dx.doi.org/doi:10.1016/b978-0-12-266722-0.50016-6)

See also: [`KernelFactors.sobel`](@ref), [`Kernel.prewitt`](@ref),
[`Kernel.ando3`](@ref), [`Kernel.scharr`](@ref), [`Kernel.bickley`](@ref) and
[`imgradients`](@ref).
"""
sobel() = product2d(KernelFactors.sobel())

sobel(extended, d) = (broadcast(*, KernelFactors.sobel(extended, d)...),)

"""
```julia
    diff1, diff2 = prewitt()
```

Return ``3 \\times 3`` kernels for two-dimensional gradient compution using the
Prewitt operator.  The `diff1` kernel computes the gradient along the y-axis
(first dimension), and the `diff2` kernel computes the gradient along the
x-axis (second dimension).

# Citation
J. M. Prewitt, "Object enhancement and extraction," *Picture processing and Psychopictorics*, vol. 10, no. 1, pp. 15–19, 1970.

See also: [`KernelFactors.prewitt`](@ref), [`Kernel.sobel`](@ref),
[`Kernel.ando3`](@ref), [`Kernel.scharr`](@ref),[`Kernel.bickley`](@ref) and
[`ImageFiltering.imgradients`](@ref).
"""
prewitt() = product2d(KernelFactors.prewitt())

prewitt(extended, d) = (broadcast(*, KernelFactors.prewitt(extended, d)...),)

"""
```julia
    diff1, diff2 = ando3()
```

Return ``3 \\times 3`` for two-dimensional gradient compution using  Ando's
"optimal" filters. The `diff1` kernel computes the gradient along the y-axis
(first dimension), and the `diff2` kernel computes the gradient along the x-axis
(second dimension).

# Citation
S. Ando, "Consistent gradient operators," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 22, no.3, pp. 252–265, 2000. [doi:10.1109/34.841757](http://dx.doi.org/doi:10.1109/34.841757)

See also: [`KernelFactors.ando3`](@ref), [`Kernel.ando4`](@ref),
[`Kernel.ando5`](@ref) and  [`ImageFiltering.imgradients`](@ref).
"""
ando3() = product2d(KernelFactors.ando3())

ando3(extended, d) = (broadcast(*, KernelFactors.ando3(extended, d)...),)

"""
```julia
    diff1, diff2 = ando4()
```

Return ``4 \\times 4`` kernels for two-dimensional gradient compution using
Ando's "optimal" filters.  The `diff1` kernel computes the gradient along the
y-axis (first dimension), and  the `diff2` kernel computes the gradient along
the x-axis (second dimension).

# Citation
S. Ando, "Consistent gradient operators," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 22, no.3, pp. 252–265, 2000. [doi:10.1109/34.841757](http://dx.doi.org/doi:10.1109/34.841757)

See also: [`KernelFactors.ando4`](@ref), [`Kernel.ando3`](@ref),
[`Kernel.ando5`](@ref) and [`ImageFiltering.imgradients`](@ref).
"""
function ando4()
    f = centered(@SMatrix [ -0.022116 -0.025526  0.025526  0.022116
                            -0.098381 -0.112984  0.112984  0.098381
                            -0.098381 -0.112984  0.112984  0.098381
                            -0.022116 -0.025526  0.025526  0.022116 ])
    return f', f
end

function ando4(extended::Tuple{Bool,Bool}, d)
    all(extended) || error("all dimensions must be extended")
    (ando4()[d],)
end

"""
```julia
    diff1, diff2 = ando5()
```

Return ``5 \\times 5`` kernels for two-dimensional gradient compution using
Ando's "optimal" filters. The `diff1` kernel computes the gradient along the
y-axis (first dimension), and the `diff2` kernel computes the gradient along the
x-axis (second dimension).

# Citation
S. Ando, "Consistent gradient operators," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 22, no.3, pp. 252–265, 2000. [doi:10.1109/34.841757](http://dx.doi.org/doi:10.1109/34.841757)

See also: [`KernelFactors.ando5`](@ref), [`Kernel.ando3`](@ref),
[`Kernel.ando4`](@ref) and  [`ImageFiltering.imgradients`](@ref).
"""
function ando5()
    f = centered(@SMatrix [ -0.003776 -0.010199  0.0  0.010199  0.003776
                            -0.026786 -0.070844  0.0  0.070844  0.026786
                            -0.046548 -0.122572  0.0  0.122572  0.046548
                            -0.026786 -0.070844  0.0  0.070844  0.026786
                            -0.003776 -0.010199  0.0  0.010199  0.003776 ])
    return f', f
end

function ando5(extended::Tuple{Bool,Bool}, d)
    all(extended) || error("all dimensions must be extended")
    (ando5()[d],)
end

"""
```julia
    diff1, diff2 = scharr()
```

Return ``3 \\times 3`` kernels for two-dimensional gradient compution using the Scharr
operator. The `diff1` kernel computes the gradient along the y-axis (first dimension),
and the `diff2` kernel  computes the gradient along the x-axis (second dimension).

# Citation
H. Scharr and  J. Weickert, "An anisotropic diffusion algorithm with optimized rotation invariance," *Mustererkennung 2000*, pp. 460–467, 2000. [doi:10.1007/978-3-642-59802-9_58](http://dx.doi.org/doi:10.1007/978-3-642-59802-9_58)

See also: [`KernelFactors.scharr`](@ref), [`Kernel.prewitt`](@ref),
[`Kernel.ando3`](@ref), [`Kernel.bickley`](@ref) and
[`ImageFiltering.imgradients`](@ref).
"""
scharr() = product2d(KernelFactors.scharr())

scharr(extended, d) = (broadcast(*, KernelFactors.scharr(extended, d)...),)

"""
```julia
    diff1, diff2 = bickley()
```

Return ``3 \\times 3`` kernels for two-dimensional gradient compution using the
Bickley operator. The `diff1` kernel computes the gradient along the y-axis
(first dimension), and the `diff2` kernel computes the gradient along the x-axis
(second dimension).

# Citation
W. G. Bickley, "Finite difference formulae for the square lattice," *The Quarterly Journal of Mechanics and Applied Mathematics*, vol. 1, no. 1, pp. 35–42, 1948.  [doi:10.1093/qjmam/1.1.35](http://dx.doi.org/doi:10.1137/12087092x)


See also: [`KernelFactors.bickley`](@ref), [`Kernel.prewitt`](@ref),
[`Kernel.ando3`](@ref),  [`Kernel.scharr`](@ref) and
[`ImageFiltering.imgradients`](@ref).
"""
bickley() = product2d(KernelFactors.bickley())

bickley(extended, d) = (broadcast(*, KernelFactors.bickley(extended, d)...),)

"""
    gaussian((σ1, σ2, ...), [(l1, l2, ...]) -> g
    gaussian(σ)                  -> g

Construct a multidimensional gaussian filter, with standard deviation
`σd` along dimension `d`. Optionally provide the kernel length `l`,
which must be a tuple of the same length.

If `σ` is supplied as a single number, a symmetric 2d kernel is
constructed.

See also: [`KernelFactors.gaussian`](@ref).
"""
@inline gaussian(σs::NTuple{N,Real}, ls::NTuple{N,Integer}) where {N} =
    broadcast(*, KernelFactors.gaussian(σs, ls)...)
gaussian(σ::Tuple{Real}, l::Tuple{Integer}) = KernelFactors.gaussian(σ[1], l[1])
gaussian(σ::Tuple{}, l::Tuple{}) = reshape([1])  # 0d
gaussian(σs::AbstractVector{T}, ls::AbstractVector{I}) where {T<:Real,I<:Integer} =
    gaussian((σs...,), (ls...,))

@inline gaussian(σs::NTuple{N,Real}) where {N} = broadcast(*, KernelFactors.gaussian(σs)...)
gaussian(σs::AbstractVector{T}) where {T<:Real} = gaussian((σs...,))
gaussian(σ::Tuple{Real}) = KernelFactors.gaussian(σ[1])
gaussian(σ::Tuple{}) = reshape([1])

gaussian(σ::Real) = gaussian((σ, σ))

"""
    DoG((σp1, σp2, ...), (σm1, σm2, ...), [l1, l2, ...]) -> k
    DoG((σ1, σ2, ...))                                   -> k
    DoG(σ::Real)                                         -> k

Construct a multidimensional difference-of-gaussian kernel `k`, equal
to `gaussian(σp, l)-gaussian(σm, l)`.  When only a single `σ` is
supplied, the default is to choose `σp = σ, σm = √2 σ`. Optionally
provide the kernel length `l`; the default is to extend by two
`max(σp,σm)` in each direction from the center. `l` must be odd.

If `σ` is provided as a single number, a symmetric 2d DoG kernel is
returned.

See also: [`KernelFactors.IIRGaussian`](@ref).
"""
DoG(σps::NTuple{N,Real}, σms::NTuple{N,Real}, ls::NTuple{N,Integer}) where {N} =
    gaussian(σps, ls) - gaussian(σms, ls)
function DoG(σps::NTuple{N,Real}) where N
    σms = map(s->s*√2, σps)
    neg = gaussian(σms)
    l = map(length, indices(neg))
    gaussian(σps, l) - neg
end
DoG(σ::Real) = DoG((σ,σ))

"""
    LoG((σ1, σ2, ...)) -> k
    LoG(σ)             -> k

Construct a Laplacian-of-Gaussian kernel `k`. `σd` is the gaussian width
along dimension `d`.  If `σ` is supplied as a single number, a
symmetric 2d kernel is returned.

See also: [`KernelFactors.IIRGaussian`](@ref) and [`Kernel.Laplacian`](@ref).
"""
function LoG(σs::NTuple{N}) where N
    w = CartesianIndex(map(n->(ceil(Int,8.5*n)>>1), σs))
    R = CartesianRange(-w, w)
    σ = SVector(σs)
    C = 1/(prod(σ)*(2π)^(N/2))
    σ2 = σ.^2
    σ2i = sum(1./σ2)
    function df(I::CartesianIndex, σ2, σ2i)
        x = SVector(I.I)
        xσ = x.^2./σ2
        (sum(xσ./σ2) - σ2i) * exp(-sum(xσ)/2)
    end
    centered([C*df(I, σ2, σ2i) for I in R])
end
LoG(σ::Real) = LoG((σ,σ))

struct Laplacian{N}
    flags::NTuple{N,Bool}
    offsets::Vector{CartesianIndex{N}}

    function Laplacian{N}(flags::NTuple{N,Bool}) where {N}
        offsets = Array{CartesianIndex{N}}(0)
        for i = 1:N
            if flags[i]
                push!(offsets,
                      CartesianIndex{N}((ntuple(d->0, i-1)..., 1, ntuple(d->0, N-i)...)))
            end
        end
        new{N}(flags, offsets)
    end
end

"""
    Laplacian((true,true,false,...))
    Laplacian(dims, N)
    Lacplacian()

Laplacian kernel in `N` dimensions, taking derivatives along the
directions marked as `true` in the supplied tuple. Alternatively, one
can pass `dims`, a listing of the dimensions for
differentiation. (However, this variant is not inferrable.)

`Laplacian()` is the 2d laplacian, equivalent to `Laplacian((true,true))`.

The kernel is represented as an opaque type, but you can use
`convert(AbstractArray, L)` to convert it into array format.
"""
Laplacian(flags::NTuple{N,Bool}) where {N} = Laplacian{N}(flags)
Laplacian() = Laplacian((true,true))

function Laplacian(dims, N::Int)
    flags = falses(N)
    flags[[dims...]] = true
    Laplacian((flags...,))
end

Base.indices(L::Laplacian) = map(f->f ? (-1:1) : (0:0), L.flags)
Base.isempty(L::Laplacian) = false
function Base.convert(::Type{AbstractArray}, L::Laplacian{N}) where N
    A = fill!(OffsetArray{Int}(indices(L)), 0)
    for I in L.offsets
        A[I] = A[-I] = 1
    end
    A[ntuple(d->0, Val{N})...] = -2*length(L.offsets)
    A
end
_reshape(L::Laplacian{N}, ::Type{Val{N}}) where {N} = L

"""
    reflect(kernel) --> reflectedkernel

Compute the pointwise reflection around 0, 0, ... of the kernel
`kernel`.  Using `imfilter` with a `reflectedkernel` performs convolution,
rather than correlation, with respect to the original `kernel`.
"""
function reflect(kernel::AbstractArray)
    inds = map(reflectind, indices(kernel))
    out = similar(kernel, inds)
    for I in CartesianRange(indices(kernel))
        out[-I] = kernel[I]
    end
    out
end

reflectind(r::AbstractUnitRange) = -last(r):-first(r)

end
