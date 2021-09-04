"""
`Kernel` is a module implementing filtering (correlation) kernels of full
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
  - `gabor`
  - `moffat`

See also: [`KernelFactors`](@ref).
"""
module Kernel

using StaticArrays, OffsetArrays
using ..ImageFiltering
using ..ImageFiltering.KernelFactors
import ..ImageFiltering: _reshape, IdentityUnitRange

# We would like to do `using ..ImageFiltering.imgradients` so that that
# Documenter.jl (the documentation system) can parse a reference such as `See
# also: [`ImageFiltering.imgradients`](@ref)`. However, imgradients is not yet
# in scope because of the order in which include files are included into
# ImageFiltering.jl. With the more general `using ImageFiltering`, we seem to
# sidestep the scope problem, although I don't actually understand the mechanism
# form why this works. - ZS

function product2d(kf)
    k1, k2 = kf
    k1[1].*k1[2], k2[1].*k2[2]
end

"""
    kern = box(m, n)
    kern = box((m, n, ...))

Return a box kernel computing a moving average. `m, n, ...` specify the size of the kernel, which is centered around zero.
"""
box(sz::Dims) = broadcast(*, KernelFactors.box(sz)...)

"""
```julia
    diff1, diff2 = sobel()
```

Return ``3 \\times 3`` correlation kernels for two-dimensional gradient compution
using the Sobel operator. The `diff1` kernel computes the gradient along the
y-axis (first dimension), and the `diff2` kernel computes the gradient along the
x-axis (second dimension). `diff1 == rotr90(diff2)`

```julia
    (diff,) = sobel(extended::NTuple{N,Bool}, d)
```
Return (a tuple of) the N-dimensional correlation kernel for gradient compution
along the dimension `d` using the Sobel operator. If `extended[dim]` is false,
`diff` will have size 1 along that dimension.

# Citation
P.-E. Danielsson and O. Seger, "Generalized and separable sobel operators," in  *Machine Vision for Three-Dimensional Scenes*,  H. Freeman, Ed.  Academic Press, 1990,  pp. 347–379. [doi:10.1016/b978-0-12-266722-0.50016-6](https://doi.org/doi:10.1016/b978-0-12-266722-0.50016-6)

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

Return ``3 \\times 3`` correlation kernels for two-dimensional gradient compution
using the Prewitt operator. The `diff1` kernel computes the gradient along the
y-axis (first dimension), and the `diff2` kernel computes the gradient along the
x-axis (second dimension). `diff1 == rotr90(diff2)`

```julia
    (diff,) = prewitt(extended::NTuple{N,Bool}, d)
```
Return (a tuple of) the N-dimensional correlation kernel for gradient compution
along the dimension `d` using the Prewitt operator. If `extended[dim]` is false,
`diff` will have size 1 along that dimension.

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

Return ``3 \\times 3`` correlation kernels for two-dimensional gradient
compution using Ando's "optimal" filters. The `diff1` kernel computes the
gradient along the y-axis (first dimension), and the `diff2` kernel computes the
gradient along the x-axis (second dimension). `diff1 == rotr90(diff2)`

```julia
    (diff,) = ando3(extended::NTuple{N,Bool}, d)
```
Return (a tuple of) the N-dimensional correlation kernel for gradient compution
along the dimension `d` using Ando's "optimal" filters of size 3. If
`extended[dim]` is false, `diff` will have size 1 along that dimension.

# Citation
S. Ando, "Consistent gradient operators," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 22, no.3, pp. 252–265, 2000. [doi:10.1109/34.841757](https://doi.org/doi:10.1109/34.841757)

See also: [`KernelFactors.ando3`](@ref), [`Kernel.ando4`](@ref),
[`Kernel.ando5`](@ref) and  [`ImageFiltering.imgradients`](@ref).
"""
ando3() = product2d(KernelFactors.ando3())

ando3(extended, d) = (broadcast(*, KernelFactors.ando3(extended, d)...),)

"""
```julia
    diff1, diff2 = ando4()
```

Return ``4 \\times 4`` correlation  kernels for two-dimensional gradient
compution using Ando's "optimal" filters.  The `diff1` kernel computes the
gradient along the y-axis (first dimension), and  the `diff2` kernel computes
the gradient along the x-axis (second dimension). `diff1 == rotr90(diff2)`

```julia
    (diff,) = ando4(extended::NTuple{N,Bool}, d)
```
Return (a tuple of) the N-dimensional correlation kernel for gradient compution
along the dimension `d` using Ando's "optimal" filters of size 4. If
`extended[dim]` is false, `diff` will have size 1 along that dimension.

# Citation
S. Ando, "Consistent gradient operators," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 22, no.3, pp. 252–265, 2000. [doi:10.1109/34.841757](https://doi.org/doi:10.1109/34.841757)

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

Return ``5 \\times 5`` correlation  kernels for two-dimensional gradient
compution using Ando's "optimal" filters.  The `diff1` kernel computes the
gradient along the y-axis (first dimension), and  the `diff2` kernel computes
the gradient along the x-axis (second dimension). `diff1 == rotr90(diff2)`

```julia
    (diff,) = ando5(extended::NTuple{N,Bool}, d)
```
Return (a tuple of) the N-dimensional correlation kernel for gradient compution
along the dimension `d` using Ando's "optimal" filters of size 5. If
`extended[dim]` is false, `diff` will have size 1 along that dimension.

# Citation
S. Ando, "Consistent gradient operators," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 22, no.3, pp. 252–265, 2000. [doi:10.1109/34.841757](https://doi.org/doi:10.1109/34.841757)

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

Return ``3 \\times 3`` correlation kernels for two-dimensional gradient
compution using the Scharr operator. The `diff1` kernel computes the gradient
along the y-axis (first dimension), and the `diff2` kernel  computes the
gradient along the x-axis (second dimension). `diff1 == rotr90(diff2)`

```julia
    (diff,) = scharr(extended::NTuple{N,Bool}, d)
```
Return (a tuple of) the N-dimensional correlation kernel for gradient compution
along the dimension `d` using the Scharr operator. If `extended[dim]` is false,
`diff` will have size 1 along that dimension.

# Citation
H. Scharr and  J. Weickert, "An anisotropic diffusion algorithm with optimized rotation invariance," *Mustererkennung 2000*, pp. 460–467, 2000. [doi:10.1007/978-3-642-59802-9_58](https://doi.org/doi:10.1007/978-3-642-59802-9_58)

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

Return ``3 \\times 3`` correlation kernels for two-dimensional gradient
compution using the Bickley operator. The `diff1` kernel computes the gradient
along the y-axis (first dimension), and the `diff2` kernel computes the gradient
along the x-axis (second dimension). `diff1 == rotr90(diff2)`

```julia
    (diff,) = bickley(extended::NTuple{N,Bool}, d)
```
Return (a tuple of) the N-dimensional correlation kernel for gradient compution
along the dimension `d` using the Bickley operator. If `extended[dim]` is false,
`diff` will have size 1 along that dimension.

# Citation
W. G. Bickley, "Finite difference formulae for the square lattice," *The Quarterly Journal of Mechanics and Applied Mathematics*, vol. 1, no. 1, pp. 35–42, 1948.  [doi:10.1093/qjmam/1.1.35](https://doi.org/doi:10.1137/12087092x)


See also: [`KernelFactors.bickley`](@ref), [`Kernel.prewitt`](@ref),
[`Kernel.ando3`](@ref),  [`Kernel.scharr`](@ref) and
[`ImageFiltering.imgradients`](@ref).
"""
bickley() = product2d(KernelFactors.bickley())

bickley(extended, d) = (broadcast(*, KernelFactors.bickley(extended, d)...),)

"""
    gaussian((σ1, σ2, ...), [(l1, l2, ...)]) -> g
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
    l = map(length, axes(neg))
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
    ws = map(n->(ceil(Int,8.5*n)>>1), σs)
    R = CartesianIndices(map(w->IdentityUnitRange(-w:w), ws))
    σ = SVector(σs)
    C = 1/(prod(σ)*(2π)^(N/2))
    σ2 = σ.^2
    σ2i = sum(1 ./ σ2)
    function df(I::CartesianIndex, σ2, σ2i)
        x = SVector(Tuple(I))
        xσ = x.^2 ./ σ2
        (sum(xσ./σ2) - σ2i) * exp(-sum(xσ)/2)
    end
    [C*df(I, σ2, σ2i) for I in R]
end
LoG(σ::Real) = LoG((σ,σ))

struct Laplacian{N}
    flags::NTuple{N,Bool}
    offsets::Vector{CartesianIndex{N}}

    function Laplacian{N}(flags::NTuple{N,Bool}) where {N}
        offsets = Array{CartesianIndex{N}}(undef, 0)
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
    Laplacian()

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
    flags[[dims...]] .= true
    Laplacian((flags...,))
end

Base.axes(L::Laplacian) = map(f->f ? (-1:1) : (0:0), L.flags)
Base.isempty(L::Laplacian) = false
function Base.convert(::Type{AbstractArray}, L::Laplacian{N}) where N
    A = fill!(OffsetArray{Int}(undef, axes(L)), 0)
    for I in L.offsets
        A[I] = A[-I] = 1
    end
    A[ntuple(d->0, Val(N))...] = -2*length(L.offsets)
    A
end
_reshape(L::Laplacian{N}, ::Val{N}) where {N} = L

"""
    laplacian2d(alpha::Number)

Construct a weighted discrete Laplacian approximation in 2d. `alpha` controls the weighting of the faces
relative to the corners.

# Examples

```jldoctest
julia> Kernel.laplacian2d(0)      # the standard Laplacian
3×3 OffsetArray(::Matrix{Float64}, -1:1, -1:1) with eltype Float64 with indices -1:1×-1:1:
 0.0   1.0  0.0
 1.0  -4.0  1.0
 0.0   1.0  0.0

julia> Kernel.laplacian2d(1)      # a corner-focused Laplacian
3×3 OffsetArray(::Matrix{Float64}, -1:1, -1:1) with eltype Float64 with indices -1:1×-1:1:
 0.5   0.0  0.5
 0.0  -2.0  0.0
 0.5   0.0  0.5

julia> Kernel.laplacian2d(0.5)    # equal weight for face-pixels and corner-pixels.
3×3 OffsetArray(::Matrix{Float64}, -1:1, -1:1) with eltype Float64 with indices -1:1×-1:1:
 0.333333   0.333333  0.333333
 0.333333  -2.66667   0.333333
 0.333333   0.333333  0.333333
```
"""
function laplacian2d(alpha::Number=0)
    lc = alpha/(1 + alpha)
    lb = (1 - alpha)/(1 + alpha)
    lm = -4/(1 + alpha)
    return centered([lc lb lc; lb lm lb; lc lb lc])
end

"""
    gabor(size_x,size_y,σ,θ,λ,γ,ψ) -> (k_real,k_complex)

Returns a 2 Dimensional Complex Gabor kernel contained in a tuple where

  - `size_x`, `size_y` denote the size of the kernel
  - `σ` denotes the standard deviation of the Gaussian envelope
  - `θ` represents the orientation of the normal to the parallel stripes of a Gabor function
  - `λ` represents the wavelength of the sinusoidal factor
  - `γ` is the spatial aspect ratio, and specifies the ellipticity of the support of the Gabor function
  - `ψ` is the phase offset

#Citation
N. Petkov and P. Kruizinga, “Computational models of visual neurons specialised in the detection of periodic and aperiodic oriented visual stimuli: bar and grating cells,” Biological Cybernetics, vol. 76, no. 2, pp. 83–96, Feb. 1997. doi.org/10.1007/s004220050323
"""
function gabor(size_x::Integer, size_y::Integer, σ::Real, θ::Real, λ::Real, γ::Real, ψ::Real)

    σx = σ
    σy = σ/γ
    nstds = 3
    c = cos(θ)
    s = sin(θ)

    validate_gabor(σ,λ,γ)

    if(size_x > 0)
        xmax = floor(Int64,size_x/2)
    else
        @warn "The input parameter size_x should be positive. Using size_x = 6 * σx + 1 (Default value)"
        xmax = round(Int64,max(abs(nstds*σx*c),abs(nstds*σy*s),1))
    end

    if(size_y > 0)
        ymax = floor(Int64,size_y/2)
    else
        @warn "The input parameter size_y should be positive. Using size_y = 6 * σy + 1 (Default value)"
        ymax = round(Int64,max(abs(nstds*σx*s),abs(nstds*σy*c),1))
    end

    xmin = -xmax
    ymin = -ymax

    x = [j for i in xmin:xmax,j in ymin:ymax]
    y = [i for i in xmin:xmax,j in ymin:ymax]
    xr = x*c + y*s
    yr = -x*s + y*c

    kernel_real = (exp.(-0.5*(((xr.*xr)/σx^2) + ((yr.*yr)/σy^2))).*cos.(2*(π/λ)*xr .+ ψ))
    kernel_imag = (exp.(-0.5*(((xr.*xr)/σx^2) + ((yr.*yr)/σy^2))).*sin.(2*(π/λ)*xr .+ ψ))

    kernel = (kernel_real,kernel_imag)
    return kernel
end

function validate_gabor(σ::Real,λ::Real,γ::Real)
    if !(σ>0 && λ>0 && γ>0)
        throw(ArgumentError("The parameters σ, λ and γ must be positive numbers."))
    end
end

"""
    moffat(α, β, ls) -> k

Constructs a 2D, symmetric Moffat kernel `k` with core width, `α`, and power, `β`.
Size of kernel defaults to 4 * full-width-half-max or as specified in `ls`.
See [this notebook](https://nbviewer.jupyter.org/github/ysbach/AO_2017/blob/master/04_Ground_Based_Concept.ipynb#1.2.-Moffat) for details.

# Citation
Moffat, A. F. J. "A theoretical investigation of focal stellar images in the photographic emulsion and application to photographic photometry." Astronomy and Astrophysics 3 (1969): 455.
"""
function moffat(α::Real, β::Real, ls::Tuple{Integer, Integer})
    ws = map(n->(ceil(Int,n)>>1), ls)
    R = CartesianIndices(map(w->IdentityUnitRange(-w:w), ws))
    α2 = α^2
    amp = (β - 1)/(π * α2)
    @. amp*((1+df(R)/α2)^-β)
end
moffat(α::Real, β::Real, ls::Integer)    = moffat(α, β, (ls,ls))
moffat(α::Real, β::Real)                 = moffat(α, β, ceil(Int, (α*2*sqrt(2^(1/β) - 1))*4))

@inline function df(I::CartesianIndex)
    x = SVector(Tuple(I))
    sum(x.^2)
end

"""
    reflect(kernel) --> reflectedkernel

Compute the pointwise reflection around 0, 0, ... of the kernel
`kernel`.  Using `imfilter` with a `reflectedkernel` performs convolution,
rather than correlation, with respect to the original `kernel`.
"""
function reflect(kernel::AbstractArray)
    inds = map(reflectind, axes(kernel))
    out = similar(kernel, inds)
    for I in CartesianIndices(axes(kernel))
        out[-I] = kernel[I]
    end
    out
end

reflectind(r::AbstractUnitRange) = -last(r):-first(r)

if Base.VERSION >= v"1.4.2" && ccall(:jl_generating_output, Cint, ()) == 1
    precompile(Laplacian, ())
    precompile(sobel, ())
    for T in (Int, Float64, Float32)
        precompile(gaussian, (Tuple{T,T},))
        precompile(DoG, (T,))
        precompile(LoG, (T,))
    end
end

end
