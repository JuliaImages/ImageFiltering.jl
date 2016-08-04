module ImagesFiltering

using ComputationalResources
using ColorVectorSpace  # in case someone filters RGB arrays

# include("kernel.jl")
# using .Kernel
include("border.jl")
# using .Border
# using .Border: AbstractBorder, Pad, Inner, Fill

export Kernel, Pad, Fill, Inner, imfilter, imfilter!, padarray

# deliberately don't export these, but it's expected that they will be used
abstract Alg
immutable FFT <: Alg end
immutable FIR <: Alg end

# see below for imfilter docstring
function imfilter(img::AbstractArray, kernel, args...)
    imfilter(filter_type(img, kernel), img, kernel, args...)
end

function imfilter{T}(::Type{T}, img::AbstractArray, kernel, args...)
    imfilter!(similar(img, T), img, kernel, args...)
end

function imfilter(r::AbstractResource, img::AbstractArray, kernel, args...)
    imfilter(r, filter_type(img, kernel), img, kernel, args...)
end

function imfilter{T}(r::AbstractResource, ::Type{T}, img::AbstractArray, kernel::AbstractArray, args...)
    imfilter(r, T, img, factorkernel(kernel), args...)
end

function imfilter{T}(r::AbstractResource, ::Type{T}, img::AbstractArray, kernel::Tuple)
    imfilter(r, T, img, kernel, Pad{:replicate}(kernel))
end

function imfilter{T}(r::AbstractResource, ::Type{T}, img::AbstractArray, kernel::Tuple, border::AbstractBorder)
    imfilter!(r, similar(img, T), img, kernel, Pad{:replicate}(kernel))
end

"""
    imfilter([T], img, kernel, [border=Pad], [alg]) --> imgfilt
    imfilter([r], img, kernel, [border=Pad], [alg]) --> imgfilt
    imfilter(r, T, img, kernel, [border=Pad], [alg]) --> imgfilt

Filter an array `img` with kernel `kernel` by computing their
correlation.

`kernel[0,0,..]` corresponds to the center (zero displacement) of the
kernel; the OffsetArrays package allows you to set `kernel`'s
indices. For example, to filter with a random *centered* 3x3 kernel,
you might use

    kernel = OffsetArray(rand(3,3), -1:1, -1:1)

`kernel` can be specified as an array or as a "factored kernel," a
tuple `(filt1, filt2, ...)` of filters to apply along each axis of the
image. In cases where you know your kernel is separable, this format
can speed processing.  Each of these should have the same
dimensionality as the image itself, and be shaped in a manner that
indicates the filtering axis, e.g., a 3x1 filter for filtering the
first dimension and a 1x3 filter for filtering the second
dimension. In two dimensions, any kernel passed as a single matrix is
checked for separability; if you want to eliminate that check, pass
the kernel as a single-element tuple, `(kernel,)`.

Optionally specify the `border`, as one of `Fill(value)`,
`Pad{:replicate}`, `Pad{:circular}`, `Pad{:symmetric}`, `Pad{:reflect}`,
or `Inner()`. The default is `Pad{:replicate}`. These choices specify
the boundary conditions, and therefore affect the result at the edges
of the image.

`alg` allows you to choose the particular algorithm: `FIR()` (finite
impulse response, aka traditional digital filtering) or `FFT()`
(Fourier-based filtering). If no choice is specified, one will be
chosen based on the size of the image and kernel in a way that
strives to deliver good performance.

Optionally, you can control the element type of the output image by
passing in a type `T` as the first argument.

You can also dispatch to different implementations by passing in a
resource `r` as defined by the ComputationalResources package.  For
example,

    imfilter(ArrayFire(), img, kernel)

would request that the computation be performed on the GPU using the
ArrayFire libraries.

See also `imfilter!`.
"""
imfilter

# see below for imfilter! docstring
function imfilter!(out::AbstractArray, img::AbstractArray, kernel::AbstractArray, args...)
    imfilter!(out, img, factorkernel(kernel), args...)
end

function imfilter!(r::AbstractResource, out::AbstractArray, img::AbstractArray, kernel::AbstractArray, args...)
    imfilter!(r, out, img, factorkernel(kernel), args...)
end

function imfilter!(out::AbstractArray, img::AbstractArray, kernel::Tuple)
    imfilter!(out, img, kernel, Pad{:replicate}(kernel))
end

function imfilter!(r::AbstractResource, out::AbstractArray, img::AbstractArray, kernel::Tuple)
    imfilter!(r, out, img, kernel, Pad{:replicate}(kernel))
end

function imfilter!(out::AbstractArray, img::AbstractArray, kernel::Tuple, border::AbstractBorder)
    imfilter!(out, img, kernel, border, filter_algorithm(out, img, kernel))
end

function imfilter!(out::AbstractArray, img::AbstractArray, kernel::Tuple, alg::Alg)
    imfilter!(out, img, kernel, Pad{:replicate}(kernel), alg)
end

function imfilter!(out::AbstractArray, img::AbstractArray, kernel::Tuple, border::AbstractBorder, alg::Alg)
    imfilter!(CPU1(alg), out, img, kernel, border)
end

"""
    imfilter!(imgfilt, img, kernel, [border=Pad], [alg])
    imfilter!([r], imgfilt, img, kernel, [border=Pad])

Filter an array `img` with kernel `kernel` by computing their
correlation, storing the result in `imgfilt`.

The indices of `imgfilt` determine the region over which the filtered
image is computed---you can use this fact to select a specific region
of interest.

See `imfilter` for information about the arguments.
"""
imfilter!

function imfilter!{S,T,N}(r::AbstractResource,
                          out::AbstractArray{S,N},
                          img::AbstractArray{T,N},
                          kernel::Tuple,
                          border::AbstractBorder)
    A = padarray(S, img, border)
    if length(kernel) == 1
        imfilter!(r, out, A, kernel[1])
    else
        Ac = similar(A)
        for i = 1:length(kernel)-1
            kern = kernel[i]
            imfilter!(r, Ac, A, kern, interior(Ac, kern))
            A, Ac = Ac, A
        end
        imfilter!(r, out, A, kernel[end])
    end
    out
end

function imfilter!{S,T,K,N}(::CPU1{FIR},
                            out::AbstractArray{S,N},
                            A::AbstractArray{T,N},
                            kern::AbstractArray{K,N},
                            R::CartesianRange=CartesianRange(indices(out)))
    indso, indsA, indsk = indices(out), indices(A), indices(kern)
    for i = 1:N
        # Check that R is inbounds for out
        if R.start[i] < first(indso[i]) || R.stop[i] > last(indso[i])
            throw(DimensionMismatch("output indices $indso disagrees with requested range $R"))
        end
        if      first(indsA[i]) > R.start[i] + first(indsk[i]) ||
                last(indsA[i])  < R.stop[i]  + last(indsk[i])
            throw(DimensionMismatch("requested range $R and kernel indices $indsk do not agree with indices of padded input, $indsA"))
        end
    end
    (isempty(A) || isempty(kern)) && return out
    p = first(A) * first(kern)
    TT = typeof(p+p)
    for I in R
        tmp = zero(TT)
        @inbounds for J in CartesianRange(indsk)
            tmp += A[I+J]*kern[J]
        end
        @inbounds out[I] = tmp
    end
    out
end

function interior(A, kern)
    indsA, indsk = indices(A), indices(kern)
    CartesianRange(CartesianIndex(map((ia,ik)->first(ia) + lo(ik), indsA, indsk)),
                   CartesianIndex(map((ia,ik)->last(ia)  - hi(ik), indsA, indsk)))
end

filter_type{S,T}(img::AbstractArray{S}, kernel::AbstractArray{T}) = typeof(zero(S)*zero(T) + zero(S)*zero(T))
filter_type{S,T}(img::AbstractArray{S}, kernel::Tuple{AbstractArray{T},Vararg{AbstractArray{T}}}) = typeof(zero(S)*zero(T) + zero(S)*zero(T))

factorkernel(kernel::AbstractArray) = (copy(kernel),)  # copy to ensure consistency

# Note that this isn't (and can't be) type stable
function factorkernel(kernel::StridedMatrix)
    SVD = svdfact(kernel)
    U, S, Vt = SVD[:U], SVD[:S], SVD[:Vt]
    separable = true
    EPS = sqrt(eps(eltype(S)))
    for i = 2:length(S)
        separable &= (abs(S[i]) < EPS)
    end
    separable || return (copy(kernel),)
    s = S[1]
    u, v = U[:,1:1], Vt[1:1,:]
    ss = sqrt(s)
    (ss*u, ss*v)
end

function factorkernel{T}(kernel::AbstractMatrix{T})
    m, n = length(indices(kernel,1)), length(indices(kernel,2))
    kern = Array{T}(m, n)
    copy!(kern, 1:m, 1:n, kernel, indices(kernel,1), indices(kernel,2))
    _factorkernel(factorkernel(kern), kernel)
end
_factorkernel(fk::Tuple{Matrix}, kernel::AbstractMatrix) = (copy(kernel),)
function _factorkernel(fk::Tuple{Matrix,Matrix}, kernel::AbstractMatrix)
    kern1 = fk[1]
    k1 = similar(kernel, eltype(kern1), (indices(kernel,1), 0:0))
    copy!(k1, indices(k1)..., kern1, indices(kern1)...)
    kern2 = fk[2]
    k2 = similar(kernel, eltype(kern1), (0:0, indices(kernel,2)))
    copy!(k2, indices(k2)..., kern2, indices(kern2)...)
    (k1, k2)
end

filter_algorithm(out, img, kernel) = FIR()

function __init__()
    # See ComputationalResources README for explanation
    push!(LOAD_PATH, dirname(@__FILE__))
    # if haveresource(ArrayFireLibs)
    #     @eval using DummyAF
    # end
    pop!(LOAD_PATH)
end

end # module
