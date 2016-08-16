module Kernel

using StaticArrays

abstract IIRFilter{T}

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

end
