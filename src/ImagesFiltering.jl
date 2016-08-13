module ImagesFiltering

using Colors, ImagesCore, MappedArrays, FFTViews, StaticArrays, ComputationalResources
using ColorVectorSpace  # in case someone filters RGB arrays

# deliberately don't export these, but it's expected that they will be used
abstract Alg
immutable FFT <: Alg end
immutable FIR <: Alg end

Alg{A<:Alg}(r::AbstractResource{A}) = r.settings

include("kernel.jl")
using .Kernel
include("border.jl")
# using .Border
# using .Border: AbstractBorder, Pad, Inner, Fill

export Kernel, Pad, Fill, Inner, imfilter, imfilter!, padarray

typealias BorderSpec{P<:Pad,T} Union{Type{P}, Fill{T,0}}

using .Kernel: TriggsSdika

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

# These next two are explicit rather than using args... as a means to
# prevent specifying both r and an algorithm
function imfilter{T}(r::AbstractResource, ::Type{T}, img::AbstractArray, kernel::Tuple)
    imfilter(r, T, img, kernel, Pad{:replicate})
end

function imfilter{T}(r::AbstractResource, ::Type{T}, img::AbstractArray, kernel::Tuple, border::BorderSpec)
    imfilter!(r, similar(img, T), img, kernel, border)
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

# function imfilter!(out::AbstractArray, img::AbstractArray, kernel::Tuple)
#     imfilter!(out, img, kernel, Pad{:replicate}(kernel))
# end

# function imfilter!(r::AbstractResource, out::AbstractArray, img::AbstractArray, kernel::Tuple)
#     imfilter!(r, out, img, kernel, Pad{:replicate}(kernel))
# end

function imfilter!(out::AbstractArray, img::AbstractArray, kernel::Tuple)
    imfilter!(out, img, kernel, filter_algorithm(out, img, kernel))
end

function imfilter!(r::AbstractResource, out::AbstractArray, img::AbstractArray, kernel::Tuple)
    imfilter!(r, out, img, kernel, Pad{:replicate})
end

function imfilter!(out::AbstractArray, img::AbstractArray, kernel::Tuple, border::BorderSpec)
    imfilter!(out, img, kernel, border, filter_algorithm(out, img, kernel))
end

# function imfilter!(out::AbstractArray, img::AbstractArray, kernel::Tuple, border::BorderSpec)
#     imfilter!(out, img, kernel, border, filter_algorithm(out, img, kernel))
# end

function imfilter!(out::AbstractArray, img::AbstractArray, kernel::Tuple, alg::Alg)
    imfilter!(out, img, kernel, Pad{:replicate}, alg)
end

# function imfilter!{B<:AbstractBorder}(out::AbstractArray, img::AbstractArray, kernel::Tuple, border::Union{Type{B},Fill}, alg::Alg)
#     imfilter!(out, img, kernel, B(kernel, img, alg), alg)
# end

function imfilter!(out::AbstractArray, img::AbstractArray, kernel::Tuple, border::BorderSpec, alg::Alg)
    imfilter!(CPU1(alg), out, img, kernel, border)
end

# function imfilter!{B<:AbstractBorder}(r::AbstractResource, out::AbstractArray, img::AbstractArray, kernel::Tuple, ::Type{B})
#     imfilter!(r, out, img, kernel, B(kernel, img, Alg(r)))
# end

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
                          border::BorderSpec)
    bord = border(kernel, img, Alg(r))
    A = padarray(S, img, bord)
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

## FIR filtering

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
        # Check that input A is big enough not to throw a BoundsError
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

### FFT filtering

function imfilter!{S,T,N}(r::CPU1{FFT},
                          out::AbstractArray{S,N},
                          img::AbstractArray{T,N},
                          kernel::Tuple,
                          border::BorderSpec)
    bord = border(kernel, img, Alg(r))
    A = padarray(S, img, bord)
    kern = prod_kernel(kernel...)
    krn = FFTView(zeros(eltype(kern), map(length, indices(A))))
    for I in CartesianRange(indices(kern))
        krn[I] = kern[I]
    end
    Af = filtfft(A, krn)
    if map(first, indices(out)) == map(first, indices(Af))
        R = CartesianRange(indices(out))
        copy!(out, R, Af, R)
    else
        dest = FFTView(out)
        src = Base.of_indices(view(FFTView(Af), indices(dest)...), indices(dest))
        copy!(dest, src)
    end
end

filtfft(A, krn) = irfft(rfft(A).*conj(rfft(krn)), length(indices(A,1)))
function filtfft{C<:Colorant}(A::AbstractArray{C}, krn)
    Av, dims = channelview_dims(A)
    kernrs = kreshape(C, krn)
    Avf = irfft(rfft(Av, dims).*conj(rfft(kernrs, dims)), length(indices(Av, dims[1])), dims)
    colorview(base_colorant_type(C){eltype(Avf)}, Avf)
end
channelview_dims{C<:Colorant,N}(A::AbstractArray{C,N}) = channelview(A), ntuple(d->d+1, Val{N})
if ImagesCore.squeeze1
    channelview_dims{C<:ImagesCore.Color1,N}(A::AbstractArray{C,N}) = channelview(A), ntuple(identity, Val{N})
end

function kreshape{C<:Colorant}(::Type{C}, krn::FFTView)
    kern = parent(krn)
    kernrs = FFTView(reshape(kern, 1, size(kern)...))
end
if ImagesCore.squeeze1
    kreshape{C<:ImagesCore.Color1}(::Type{C}, krn::FFTView) = krn
end

### Triggs-Sdika (modified Young-van Vliet) recursive filtering
# B. Triggs and M. Sdika, "Boundary conditions for Young-van Vliet
# recursive filtering". IEEE Trans. on Sig. Proc. 54: 2365-2367
# (2006).

function imfilter_inplace!(img, kernel::TriggsSdika, dim::Integer, border::BorderSpec)
    inds = indices(img)
    k, l = length(kernel.a), length(kernel.b)
    if length(inds[dim]) <= max(k, l)
        throw(DimensionMismatch("size of img along dimension $dim $(length(inds[dim])) is too small for filtering with IIR kernel of length $(max(k,l))"))
    end
    Rbegin = CartesianRange(inds[1:dim-1])
    Rend   = CartesianRange(inds[dim+1:end])
    _imfilter_inplace!(img, kernel, Rbegin, inds[dim], Rend, border)
end

@noinline function _imfilter_inplace!{T,k,l}(img, kernel::TriggsSdika{T,k,l},
                                             Rbegin::CartesianRange, ind::AbstractUnitRange,
                                             Rend::CartesianRange, border::BorderSpec)
    for Iend in Rend
        # Initialize the left border
        indleft = ind[1:k]
        for Ibegin in Rbegin
            leftborder!(img, kernel, Ibegin, indleft, Iend, border)
        end
        # Propagate forwards. We omit the final point in case border
        # is Pad{:replicate}, so that the original value is still
        # available. rightborder! will handle that point.
        for i = ind[k+1]:ind[end-1]
            @inbounds for Ibegin in Rbegin
                tmp = one(T)*img[Ibegin, i, Iend]
                for j = 1:k
                    tmp += kernel.a[j]*img[Ibegin, i-j, Iend]
                end
                img[Ibegin, i, Iend] = tmp
            end
        end
        # Initialize the right border
        indright = ind[end-l+1:end]
        for Ibegin in Rbegin
            rightborder!(img, kernel, Ibegin, indright, Iend, border)
        end
        # Propagate backwards
        for i = ind[end-l]:-1:ind[1]
            @inbounds for Ibegin in Rbegin
                tmp = one(T)*img[Ibegin, i, Iend]
                for j = 1:l
                    tmp += kernel.b[j]*img[Ibegin, i+j, Iend]
                end
                img[Ibegin, i, Iend] = tmp
            end
        end
        # Final scaling
        for i in ind
            @inbounds for Ibegin in Rbegin
                img[Ibegin, i, Iend] *= kernel.scale
            end
        end
    end
    img
end

# Implements the initialization in the first paragraph of Triggs & Sdika, section II
function leftborder!(img, kernel, Ibegin, indleft, Iend, border::Fill)
    _leftborder!(img, kernel, Ibegin, indleft, Iend, convert(eltype(img), border.value))
end
function leftborder!(img, kernel, Ibegin, indleft, Iend, border::Pad{:replicate})
    _leftborder!(img, kernel, Ibegin, indleft, Iend, img[Ibegin, indleft[1], Iend])
end
function _leftborder!{T,k,l}(img, kernel::TriggsSdika{T,k,l}, Ibegin, indleft, Iend, iminus)
    uminus = iminus/(1-kernel.asum)
    n = 0
    for i in indleft
        n += 1
        tmp = one(T)*img[Ibegin, i, Iend]
        for j = 1:n-1
            tmp += kernel.a[j]*img[Ibegin, i-j, Iend]
        end
        for j = n:k
            tmp += kernel.a[j]*uminus
        end
        img[Ibegin, i, Iend] = tmp
    end
    img
end

# Implements Triggs & Sdika, Eqs 14-15
function rightborder!(img, kernel, Ibegin, indright, Iend, border::Fill)
    _rightborder!(img, kernel, Ibegin, indright, Iend, convert(eltype(img), border.value))
end
function rightborder!(img, kernel, Ibegin, indright, Iend, border::Pad{:replicate})
    _rightborder!(img, kernel, Ibegin, indright, Iend, img[Ibegin, indright[end], Iend])
end
function _rightborder!{T,k,l}(img, kernel::TriggsSdika{T,k,l}, Ibegin, indright, Iend, iplus)
    # The final value from forward-filtering was not calculated, so do that here
    i = last(indright)
    tmp = one(T)*img[Ibegin, i, Iend]
    for j = 1:k
        tmp += kernel.a[j]*img[Ibegin, i-j, Iend]
    end
    img[Ibegin, i, Iend] = tmp
    # Initialize the v values at and beyond the right edge
    uplus = iplus/(1-kernel.asum)
    vplus = uplus/(1-kernel.bsum)
    vright = kernel.M * rightΔu(img, uplus, Ibegin, last(indright), Iend, kernel) +
             fill(vplus, SVector{l,typeof(vplus)})
    img[Ibegin, last(indright), Iend] = vright[1]
    # Propagate inward
    n = 1
    for i in last(indright)-1:-1:first(indright)
        n += 1
        tmp = one(T)*img[Ibegin, i, Iend]
        for j = 1:n-1
            tmp += kernel.b[j]*img[Ibegin, i+j, Iend]
        end
        for j = n:l
            tmp += kernel.b[j]*vright[j-n+2]
        end
        img[Ibegin, i, Iend] = tmp
    end
    img
end

# Part of Triggs & Sdika, Eq. 14
function rightΔu{T,l}(img, uplus, Ibegin, i, Iend, kernel::TriggsSdika{T,3,l})
    @inbounds ret = SVector(img[Ibegin, i,   Iend]-uplus,
                            img[Ibegin, i-1, Iend]-uplus,
                            img[Ibegin, i-2, Iend]-uplus)
    ret
end

### Utilities

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

prod_kernel(kern) = kern
prod_kernel(kern, kern1, kerns...) = prod_kernel(kern.*kern1, kerns...)

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
