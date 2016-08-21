module KernelFactors

using StaticArrays, OffsetArrays
using ..ImagesFiltering: centered, dummyind, _reshape
using Base: tail

abstract IIRFilter{T}

Base.eltype{T}(kernel::IIRFilter{T}) = T

#### FIR filters

## gradients

"`kern1, kern2 = sobel()` returns factored Sobel filters for dimensions 1 and 2 of your image"
function sobel()
    f1 = centered(SVector( 1.0, 2.0, 1.0))
    f2 = centered(SVector(-1.0, 0.0, 1.0))
    return kernelfactors((f2, f1)), kernelfactors((f1, f2))
end

"`kern1, kern2 = prewitt()` returns factored Prewitt filters for dimensions 1 and 2 of your image"
function prewitt()
    f1 = centered(SVector( 1.0, 1.0, 1.0))
    f2 = centered(SVector(-1.0, 0.0, 1.0))
    return kernelfactors((f2, f1)), kernelfactors((f1, f2))
end

# Consistent Gradient Operators
# Ando Shigeru
# IEEE Trans. Pat. Anal. Mach. Int., vol. 22 no 3, March 2000
#
# TODO: These coefficients were taken from the paper. It would be nice
#       to resolve the optimization problem and use higher precision
#       versions, which might allow better separable approximations of
#       ando4 and ando5.

"""
`kern1, kern2 = ando3()` returns optimal 3x3 gradient filters for dimensions 1 and 2 of your image, as defined in
Ando Shigeru, IEEE Trans. Pat. Anal. Mach. Int., vol. 22 no 3, March 2000.

See also: `ando4`, `ando5`.
"""
function ando3()
    f1 = centered(SVector(0.112737, 0.274526, 0.112737))
    f2 = centered(SVector(-1.0, 0.0, 1.0))
    return kernelfactors((f2, f1)), kernelfactors((f1, f2))
end

"""
`kern1, kern2 = ando4()` returns separable approximations of the
optimal 4x4 filters for dimensions 1 and 2 of your image, as defined
in Ando Shigeru, IEEE Trans. Pat. Anal. Mach. Int., vol. 22 no 3,
March 2000.

See also: `Kernel.ando4`.
"""
function ando4()
    f1 = centered(SVector(0.025473821998749126, 0.11299599504060115, 0.11299599504060115, 0.025473821998749126))
    f2 = centered(SVector(-0.2254400431590164, -1.0, 1.0, 0.2254400431590164))
    return kernelfactors((f2, f1)), kernelfactors((f1, f2))
end

"""
`kern1, kern2 = ando5_sep()` returns separable approximations of the
optimal 5x5 gradient filters for dimensions 1 and 2 of your image, as defined
in Ando Shigeru, IEEE Trans. Pat. Anal. Mach. Int., vol. 22 no 3,
March 2000.

See also: `Kernel.ando5`.
"""
function ando5()
    f1 = centered(SVector(0.03136697678461958, 0.21844976784066258, 0.37816313370270255, 0.21844976784066258, 0.03136697678461958))
    f2 = centered(SVector(-0.12288050911244743, -0.3242040200682835, 0.0, 0.3242040200682835, 0.12288050911244743))
    return kernelfactors((f2, f1)), kernelfactors((f1, f2))
end

## Gaussian

"""
    gaussian(σ::Real, [l]) -> g

Construct a 1d gaussian kernel `g` with standard deviation `σ`, optionally
providing the kernel length `l`. The default is to extend by two `σ`
in each direction from the center. `l` must be odd.
"""
function gaussian(σ::Real, l = 4*ceil(Int,σ)+1)
    isodd(l) || throw(ArgumentError("length must be odd"))
    w = l>>1
    g = [exp(-x^2/(2*σ^2)) for x=-w:w]
    centered(g/sum(g))
end

"""
    gaussian((σ1, σ2, ...), [l]) -> (g1, g2, ...)

Construct a multidimensional gaussian filter as a product of single-dimension
factors, with standard deviation `σd` along dimension `d`. Optionally
provide the kernel length `l`, which must be a tuple of the same
length.
"""
gaussian{N}(σs::NTuple{N,Real}, ls::NTuple{N,Integer}) =
    kernelfactors( map((σ,l)->gaussian(σ,l), σs, ls) )
gaussian{N}(σs::NTuple{N,Real}) = kernelfactors(map(σ->gaussian(σ), σs))

gaussian(σs::AbstractVector, ls::AbstractVector) = gaussian((σs...,), (ls...,))
gaussian(σs::AbstractVector) = gaussian((σs...,))

#### IIR

immutable TriggsSdika{T,k,l,L} <: IIRFilter{T}
    a::SVector{k,T}
    b::SVector{l,T}
    scale::T
    M::SMatrix{l,k,T,L}
    asum::T
    bsum::T

    TriggsSdika(a, b, scale, M) = new(a, b, scale, M, sum(a), sum(b))
end
"""
    TriggsSdika(a, b, scale, M)

Defines a kernel for one-dimensional infinite impulse response (IIR)
filtering. `a` is a "forward" filter, `b` a "backward" filter, `M` is
a matrix for matching boundary conditions at the right edge, and
`scale` is a constant scaling applied to each element at the
conclusion of filtering.

# Citation

B. Triggs and M. Sdika, "Boundary conditions for Young-van Vliet
recursive filtering". IEEE Trans. on Sig. Proc. 54: 2365-2367
(2006).
"""
TriggsSdika{T,k,l,L}(a::SVector{k,T}, b::SVector{l,T}, scale, M::SMatrix{l,k,T,L}) = TriggsSdika{T,k,l,L}(a, b, scale, M)

"""
    TriggsSdika(ab, scale)

Create a symmetric Triggs-Sdika filter (with `a = b = ab`). `M` is
calculated for you. Only length 3 filters are currently supported.
"""
function TriggsSdika{T}(a::SVector{3,T}, scale)
    a1, a2, a3 = a[1], a[2], a[3]
    Mdenom = (1+a1-a2+a3)*(1-a1-a2-a3)*(1+a2+(a1-a3)*a3)
    M = @SMatrix([-a3*a1+1-a3^2-a2     (a3+a1)*(a2+a3*a1)  a3*(a1+a3*a2);
                  a1+a3*a2            -(a2-1)*(a2+a3*a1)  -(a3*a1+a3^2+a2-1)*a3;
                  a3*a1+a2+a1^2-a2^2   a1*a2+a3*a2^2-a1*a3^2-a3^3-a3*a2+a3  a3*(a1+a3*a2)]);
    TriggsSdika(a, a, scale, M/Mdenom)
end
Base.vec(kernel::TriggsSdika) = kernel

# Note that there's a sign reversal between Young & Triggs.
"""
    IIRGaussian([T], σ; emit_warning::Bool=true)

Construct an infinite impulse response (IIR) approximation to a
Gaussian of standard deviation `σ`. `σ` may either be a single real
number or a tuple of numbers; in the latter case, a tuple of such filters
will be created, each for filtering a different dimension of an array.

Optionally specify the type `T` for the filter coefficients; if not
supplied, it will match `σ` (unless `σ` is not floating-point, in
which case `Float64` will be chosen).

# Citation

I. T. Young, L. J. van Vliet, and M. van Ginkel, "Recursive Gabor
Filtering". IEEE Trans. Sig. Proc., 50: 2798-2805 (2002).
"""
function IIRGaussian{T}(::Type{T}, sigma::Real; emit_warning::Bool = true)
    if emit_warning && sigma < 1 && sigma != 0
        warn("sigma is too small for accuracy")
    end
    m0 = convert(T,1.16680)
    m1 = convert(T,1.10783)
    m2 = convert(T,1.40586)
    q = convert(T,1.31564*(sqrt(1+0.490811*sigma*sigma) - 1))
    ascale = (m0+q)*(m1*m1 + m2*m2  + 2m1*q + q*q)
    B = (m0*(m1*m1 + m2*m2)/ascale)^2
    # This is what Young et al call -b, but in filt() notation would be called a
    a1 = q*(2*m0*m1 + m1*m1 + m2*m2 + (2*m0+4*m1)*q + 3*q*q)/ascale
    a2 = -q*q*(m0 + 2m1 + 3q)/ascale
    a3 = q*q*q/ascale
    a = SVector(a1,a2,a3)
    TriggsSdika(a, B)
end
IIRGaussian(sigma::Real; emit_warning::Bool = true) = IIRGaussian(iirgt(sigma), sigma; emit_warning=emit_warning)

function IIRGaussian{T}(::Type{T}, sigma::Tuple; emit_warning::Bool = true)
    map(s->IIRGaussian(T, s; emit_warning=emit_warning), sigma)
end
IIRGaussian(sigma::Tuple; emit_warning::Bool = true) = IIRGaussian(iirgt(sigma), sigma; emit_warning=emit_warning)

IIRGaussian(sigma::AbstractVector; kwargs...) = IIRGaussian((sigma...,); kwargs...)
IIRGaussian{T}(::Type{T}, sigma::AbstractVector; kwargs...) = IIRGaussian(T, (sigma...,); kwargs...)

iirgt(sigma::AbstractFloat) = typeof(sigma)
iirgt(sigma::Real) = Float64
iirgt(sigma::Tuple) = promote_type(map(iirgt, sigma)...)

###### Utilities

"""
    kernelfactors(factors::Tuple)

Prepare a factored kernel for filtering. If passed a 2-tuple of
vectors of lengths `m` and `n`, this will return a 2-tuple of matrices
of sizes `m×1` and `1×n`. In general, each successive `factor` will be
reshaped to extend along the corresponding dimension.

If passed a tuple of general arrays, it is assumed that each is shaped
appropriately along its "leading" dimensions; the dimensionality of each is
"extended" to `N = length(factors)`, appending 1s to the size as needed.
"""
kernelfactors{N}(factors::NTuple{N,AbstractVector}) = _kernelfactors((), factors)

_kernelfactors(out::NTuple, ::Tuple{}) = out
@inline function _kernelfactors{L,M}(out::NTuple{L}, factors::NTuple{M,AbstractVector})
    # L+M=N
    f = factors[1]
    ind1 = indices(f,1)
    ind = dummyind(ind1)
    newf = reshape(f, ntuple(d->ind, Val{L})..., ind1, tail(ntuple(d->ind, Val{M}))...)
    _kernelfactors((out..., newf), tail(factors))
end

# A variant in which we just need to fill out to N dimensions
kernelfactors{N}(factors::NTuple{N,AbstractArray}) = map(f->_reshape(f, Val{N}), factors)

end
