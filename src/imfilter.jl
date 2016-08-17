typealias BorderSpec{Style,T} Union{Pad{Style,0}, Fill{T,0}, Inner{0}}

typealias ProcessedKernel Tuple

# see below for imfilter docstring

# Step 1: if necessary, determine the output's element type
@inline function imfilter(img::AbstractArray, kernel, args...)
    imfilter(filter_type(img, kernel), img, kernel, args...)
end

# Step 2: if necessary, put the kernel into cannonical (factored) form
@inline function imfilter{T}(::Type{T}, img::AbstractArray, kernel::AbstractArray, args...)
    imfilter(T, img, factorkernel(kernel), args...)
end

# Step 3: if necessary, fill in the default border
function imfilter{T}(::Type{T}, img::AbstractArray, kernel::ProcessedKernel, args...)
    imfilter(T, img, kernel, Pad{:replicate}(), args...)
end

# Step 4: if necessary, allocate the ouput
@inline function imfilter{T}(::Type{T}, img::AbstractArray, kernel::ProcessedKernel, border::Inner{0}, args...)
    R = interior(img, kernel)
    inds = to_ranges(R)
    imfilter!(similar(img, T, inds), img, kernel, border, args...)
end
@inline function imfilter{T}(::Type{T}, img::AbstractArray, kernel::ProcessedKernel, border::BorderSpec, args...)
    imfilter!(similar(img, T), img, kernel, border, args...)
end

# Now do the same steps for the case where the user supplies a Resource
@inline function imfilter(r::AbstractResource, img::AbstractArray, kernel, args...)
    imfilter(r, filter_type(img, kernel), img, kernel, args...)
end

@inline function imfilter{T}(r::AbstractResource, ::Type{T}, img::AbstractArray, kernel::AbstractArray, args...)
    imfilter(r, T, img, factorkernel(kernel), args...)
end

# For steps 3 & 4, we make args... explicit as a means to prevent
# specifying both r and an algorithm
function imfilter{T}(r::AbstractResource, ::Type{T}, img::AbstractArray, kernel::ProcessedKernel)
    imfilter(r, T, img, kernel, Pad{:replicate}())  # supply the default border
end
function imfilter{T}(r::AbstractResource, ::Type{T}, img::AbstractArray, kernel::ProcessedKernel, border::Inner{0})
    R = interior(img, kernel)
    inds = to_ranges(R)
    imfilter!(r, similar(img, T, inds), img, kernel, border)
end
function imfilter{T}(r::AbstractResource, ::Type{T}, img::AbstractArray, kernel::ProcessedKernel, border::BorderSpec)
    imfilter!(r, similar(img, T), img, kernel, border)
end

"""
    imfilter([T], img, kernel, [border=Pad{:replicate}()], [alg]) --> imgfilt
    imfilter([r], img, kernel, [border=Pad{:replicate}()], [alg]) --> imgfilt
    imfilter(r, T, img, kernel, [border=Pad{:replicate}()], [alg]) --> imgfilt

Filter an array `img` with kernel `kernel` by computing their
correlation.

`kernel[0,0,..]` corresponds to the origin (zero displacement) of the
kernel; you can use `centered` to place the origin at the array
center, or use the OffsetArrays package to set `kernel`'s indices
manually. For example, to filter with a random *centered* 3x3 kernel,
you could use either of the following:

    kernel = centered(rand(3,3))
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
`Pad{:replicate}()`, `Pad{:circular}()`, `Pad{:symmetric}()`,
`Pad{:reflect}()`, `Pad{:na}()`, or `Inner()`. The default is
`Pad{:replicate}()`. These choices specify the boundary conditions,
and therefore affect the result at the edges of the image. See
`padarray` for more information.

`alg` allows you to choose the particular algorithm: `FIR()` (finite
impulse response, aka traditional digital filtering) or `FFT()`
(Fourier-based filtering). If no choice is specified, one will be
chosen based on the size of the image and kernel in a way that strives
to deliver good performance. Alternatively you can use a custom filter
type, like `IIRGaussian`.

Optionally, you can control the element type of the output image by
passing in a type `T` as the first argument.

You can also dispatch to different implementations by passing in a
resource `r` as defined by the ComputationalResources package.  For
example,

    imfilter(ArrayFire(), img, kernel)

would request that the computation be performed on the GPU using the
ArrayFire libraries.

See also: imfilter!, centered, padarray, Pad, Fill, Inner, IIRGaussian.
"""
imfilter

# see below for imfilter! docstring
# imfilter! can be called directly, so we take steps 2&3 here too
function imfilter!(out::AbstractArray, img::AbstractArray, kernel::AbstractArray, args...)
    imfilter!(out, img, factorkernel(kernel), args...)
end

function imfilter!(r::AbstractResource, out::AbstractArray, img::AbstractArray, kernel::AbstractArray, args...)
    imfilter!(r, out, img, factorkernel(kernel), args...)
end

function imfilter!(out::AbstractArray, img::AbstractArray, kernel::ProcessedKernel)
    imfilter!(out, img, kernel, Pad{:replicate}())
end

function imfilter!(r::AbstractResource, out::AbstractArray, img::AbstractArray, kernel::ProcessedKernel)
    imfilter!(r, out, img, kernel, Pad{:replicate}())
end

# Step 5: if necessary, pick an algorithm
function imfilter!(out::AbstractArray, img::AbstractArray, kernel::ProcessedKernel, border::BorderSpec)
    imfilter!(out, img, kernel, border, filter_algorithm(out, img, kernel))
end

function imfilter!(out::AbstractArray, img::AbstractArray, kernel::ProcessedKernel, border::BorderSpec, alg::Alg)
    imfilter!(CPU1(alg), out, img, kernel, border)
end

"""
    imfilter!(imgfilt, img, kernel, [border=Pad{:replicate}()], [alg])
    imfilter!(r, imgfilt, img, kernel, border)

Filter an array `img` with kernel `kernel` by computing their
correlation, storing the result in `imgfilt`.

The indices of `imgfilt` determine the region over which the filtered
image is computed---you can use this fact to select just a specific region
of interest.

See also: imfilter.
"""
imfilter!

# Step 6: pad the input
function imfilter!{S,T,N}(r::AbstractResource,
                          out::AbstractArray{S,N},
                          img::AbstractArray{T,N},
                          kernel::ProcessedKernel,
                          border::BorderSpec)
    bord = border(kernel, img, Alg(r))
    A = padarray(S, img, bord)
    _imfilter_padded!(r, out, A, kernel)
end

function _imfilter_padded!{_,N}(r, out::AbstractArray{_,N}, A, kernel::Tuple{AbstractArray})
    imfilter!(r, out, A, _reshape(kernel[1], Val{N}))
end

function _imfilter_padded!(r, out, A, kernel)
    Ac = similar(A)
    _imfilter_padded!(r, out, A, Ac, kernel)
    return out
end

function _imfilter_padded!{_,N}(r, out::AbstractArray{_,N}, A, Ac, kernel::Tuple{AbstractArray})
    kern = kernel[1]
    imfilter!(r, out, A, _reshape(kern, Val{N}))
end

function _imfilter_padded!(r, out, A, Ac, kernel::Tuple)
    kern = kernel[1]
    imfilter!(r, Ac, A, kern, interior(Ac, kern))
    _imfilter_padded!(r, out, Ac, A, tail(kernel))
end

# NA "padding": normalizing by the number of available values (similar to nanmean)
function imfilter!{T,S,N}(r::AbstractResource,
                          out::AbstractArray{S,N},
                          img::AbstractArray{T,N},
                          kernel::ProcessedKernel,
                          border::Pad{:na,0})
    _imfilter_na!(r, out, img, kernel, border)
end

function _imfilter_na!{T,S,N}(r::AbstractResource,
                              out::AbstractArray{S,N},
                              img::AbstractArray{T,N},
                              kernel::ProcessedKernel,
                              border::Pad{:na,0})
    fc, fn = Fill(zero(T)), Fill(zero(eltype(T)))  # color, numeric
    nanflag = map(isnan, img)
    hasnans = any(nanflag)
    if hasnans
        copy!(out, img)
        out[nanflag] = zero(T)
        validpixels = copy!(similar(Array{T}, indices(img)), mappedarray(x->!x, nanflag))
        imfilter!(r, out, out, kernel, fc)
        imfilter!(r, validpixels, validpixels, kernel, fn)
        for I in eachindex(out)
            out[I] /= validpixels[I]
        end
        out[nanflag] = convert(T, NaN)
    else
        imfilter!(r, out, img, kernel, fc)
        normalize_separable!(r, out, kernel, fn)
    end
    out
end

# for types that can't have NaNs, we can skip the isnan check
function _imfilter_na!{S,T<:Union{Integer,FixedColorant},N}(r::AbstractResource,
                                                            out::AbstractArray{S,N},
                                                            img::AbstractArray{T,N},
                                                            kernel::ProcessedKernel,
                                                            border::Pad{:na,0})
    fc, fn = Fill(zero(T)), Fill(zero(eltype(T)))
    imfilter!(r, out, img, kernel, fc)
    normalize_separable!(r, out, kernel, fn)
end

## FIR filtering

"""
    imfilter!(::AbstractResource{FIR}, imgfilt, img, kernel, [R=CartesianRange(indices(imfilt))])

Filter an array `img` with kernel `kernel` by computing their
correlation, storing the result in `imgfilt`, using a finite-impulse
response (FIR) algorithm. Any necessary padding must have already been
supplied to `img`.

If the CartesianRange `R` is supplied, only the indices of `imgfilt`
in the domain of `R` will be calculated. This can be particularly
useful for "cascaded filters" where you pad over a larger area and
then calculate the result only over the necessary region at each
stage.

See also: imfilter.
"""
function imfilter!{S,T,K,N}(::CPU1{FIR},
                            out::AbstractArray{S,N},
                            A::AbstractArray{T,N},
                            kern::AbstractArray{K,N},
                            R::CartesianRange=CartesianRange(indices(out)))
    (isempty(A) || isempty(kern)) && return out
    indso, indsA, indsk = indices(out), indices(A), indices(kern)
    if all(x->x==0:0, indsk) && first(kern) == 1
        return copy!(out, R, A, R)
    end
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
    # For performance reasons, we now dispatch to a method that strips
    # off index-shift containers (in Julia 0.5 we shouldn't remove
    # boundschecks from OffsetArray).
    _imfilter_inbounds!(out, A, kern, R)
end

function _imfilter_inbounds!(out, A::OffsetArray, kern::OffsetArray, R)
    ΔI = CartesianIndex(kern.offsets) - CartesianIndex(A.offsets)
    _imfilter_inbounds!(out, (parent(A), ΔI), parent(kern), R)
end
function _imfilter_inbounds!(out, A::AbstractArray, kern::OffsetArray, R)
    ΔI = CartesianIndex(kern.offsets)
    _imfilter_inbounds!(out, (A, ΔI), parent(kern), R)
end
function _imfilter_inbounds!(out, A::OffsetArray, kern::AbstractArray, R)
    ΔI = CartesianIndex(kern.offsets)
    _imfilter_inbounds!(out, (parent(A), ΔI), kern, R)
end
function _imfilter_inbounds!(out, A::AbstractArray, kern::AbstractArray, R)
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
function _imfilter_inbounds!(out, Ashift::Tuple{AbstractArray,CartesianIndex}, kern, R)
    A, ΔI = Ashift
    indsk = indices(kern)
    p = first(A) * first(kern)
    TT = typeof(p+p)
    for I in R
        tmp = zero(TT)
        @inbounds for J in CartesianRange(indsk)
            tmp += A[I+ΔI+J]*kern[J]
        end
        @inbounds out[I] = tmp
    end
    out
end


### FFT filtering

"""
    imfilter!(::AbstractResource{FFT}, imgfilt, img, kernel, border)

Filter an array `img` with kernel `kernel` by computing their
correlation, storing the result in `imgfilt`, using a fast Fourier
transform (FFT) algorithm. `border` specifies the type of padding to
be supplied to `img`.

See also: imfilter.
"""
function imfilter!{S,T,N}(r::CPU1{FFT},
                          out::AbstractArray{S,N},
                          img::AbstractArray{T,N},
                          kernel::Tuple{AbstractArray,Vararg{AbstractArray}},
                          border::Type{Pad{:na}})
    throw(ArgumentError("na padding is not yet available for FFT"))
end

function imfilter!{S,T,N}(r::CPU1{FFT},
                          out::AbstractArray{S,N},
                          img::AbstractArray{T,N},
                          kernel::Tuple{AbstractArray,Vararg{AbstractArray}},
                          border::BorderSpec)
    bord = border(kernel, img, Alg(r))
    A = padarray(S, img, bord)
    kern = prod_kernel(Val{N}, kernel...)
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
        src = OffsetArray(view(FFTView(Af), indices(dest)...), indices(dest))
        copy!(dest, src)
    end
    out
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

# Note this is safe for inplace use, i.e., out === img

typealias BorderSpecRF{T} Union{Pad{:replicate,0}, Fill{T,0}}

function imfilter!{S,T,N}(r::AbstractResource,
                          out::AbstractArray{S,N},
                          img::AbstractArray{T,N},
                          kernel::Tuple{TriggsSdika, Vararg{TriggsSdika}},
                          border::BorderSpecRF)
    length(kernel) <= N || throw(DimensionMismatch("cannot have more kernels than dimensions"))
    inds = indices(img)
    _imfilter_inplace_tuple!(r, out, img, kernel, CartesianRange(()), inds, CartesianRange(tail(inds)), border)
end

"""
    imfilter!(r::AbstractResource, imgfilt, img, kernel::Tuple{TriggsSdika...}, border)
    imfilter!(r::AbstractResource, imgfilt, img, kernel::TriggsSdika, dim::Integer, border)

Filter an array `img` with a Triggs-Sdika infinite impulse response
(IIR) `kernel`, storing the result in `imgfilt`. Unlike the `FIR` and
`FFT` algorithms, this version is safe for inplace operations, i.e.,
`imgfilt` can be the same array as `img`.

Either specify one kernel per dimension (as a tuple), or a particular
dimension `dim` along which to filter. If you exhaust `kernel`s before
you run out of array dimensions, the remaining dimension(s) will not
be filtered.

With Triggs-Sdika filtering, the only border options are `Pad{:na}()`,
`Pad{:replicate}()`, or `Fill(value)`.

See also: imfilter, TriggsSdika, IIRGaussian.
"""
function imfilter!(r::AbstractResource, out, img, kernel::TriggsSdika, dim::Integer, border::BorderSpec)
    inds = indices(img)
    k, l = length(kernel.a), length(kernel.b)
    # This next part is not type-stable, which is why _imfilter_dim! has a @noinline
    Rbegin = CartesianRange(inds[1:dim-1])
    Rend   = CartesianRange(inds[dim+1:end])
    _imfilter_dim!(r, out, img, kernel, Rbegin, inds[dim], Rend, border)
end

# Lispy and type-stable inplace (currently just Triggs-Sdika) filtering over each dimension
function _imfilter_inplace_tuple!(r, out, img, kernel, Rbegin, inds, Rend, border)
    ind = first(inds)
    _imfilter_dim!(r, out, img, first(kernel), Rbegin, ind, Rend, border)
    _imfilter_inplace_tuple!(r,
                             out,
                             out,
                             tail(kernel),
                             CartesianRange(CartesianIndex((Rbegin.start.I..., first(ind))),
                                            CartesianIndex((Rbegin.stop.I...,  last(ind)))),
                             tail(inds),
                             _tail(Rend),
                             border)
end
# When the final kernel has been used, return the output
_imfilter_inplace_tuple!(r, out, img, ::Tuple{}, Rbegin, inds, Rend, border) = out

# This is the "workhorse" function that performs Triggs-Sdika IIR
# filtering along a particular dimension. The "pre" dimensions are
# encoded in Rbegin, the "post" dimensions in Rend, and the dimension
# we're filtering is sandwiched between these. This design is
# type-stable and cache-friendly for any dimension---we update values
# in memory-order rather than along the chosen dimension. Nor does it
# require that the arrays have efficient linear indexing. For more
# information, see http://julialang.org/blog/2016/02/iteration.
@noinline function _imfilter_dim!{T,k,l}(r::AbstractResource,
                                         out, img, kernel::TriggsSdika{T,k,l},
                                         Rbegin::CartesianRange, ind::AbstractUnitRange,
                                         Rend::CartesianRange, border::BorderSpec)
    if all(x->x==0, kernel.a) && all(x->x==0, kernel.b) && kernel.scale == 1
        if !(out === img)
            copy!(out, img)
        end
        return out
    end
    if length(ind) <= max(k, l)
        throw(DimensionMismatch("size of img along dimension $dim $(length(inds[dim])) is too small for filtering with IIR kernel of length $(max(k,l))"))
    end
    for Iend in Rend
        # Initialize the left border
        indleft = ind[1:k]
        for Ibegin in Rbegin
            leftborder!(out, img, kernel, Ibegin, indleft, Iend, border)
        end
        # Propagate forwards. We omit the final point in case border
        # is Pad{:replicate}, so that the original value is still
        # available. rightborder! will handle that point.
        for i = ind[k]+1:ind[end-1]
            @inbounds for Ibegin in Rbegin
                tmp = one(T)*img[Ibegin, i, Iend]
                for j = 1:k
                    tmp += kernel.a[j]*out[Ibegin, i-j, Iend]
                end
                out[Ibegin, i, Iend] = tmp
            end
        end
        # Initialize the right border
        indright = ind[end-l+1:end]
        for Ibegin in Rbegin
            rightborder!(out, img, kernel, Ibegin, indright, Iend, border)
        end
        # Propagate backwards
        for i = ind[end-l]:-1:ind[1]
            @inbounds for Ibegin in Rbegin
                tmp = one(T)*out[Ibegin, i, Iend]
                for j = 1:l
                    tmp += kernel.b[j]*out[Ibegin, i+j, Iend]
                end
                out[Ibegin, i, Iend] = tmp
            end
        end
        # Final scaling
        for i in ind
            @inbounds for Ibegin in Rbegin
                out[Ibegin, i, Iend] *= kernel.scale
            end
        end
    end
    out
end

# Implements the initialization in the first paragraph of Triggs & Sdika, section II
function leftborder!(out, img, kernel, Ibegin, indleft, Iend, border::Fill)
    _leftborder!(out, img, kernel, Ibegin, indleft, Iend, convert(eltype(img), border.value))
end
function leftborder!(out, img, kernel, Ibegin, indleft, Iend, border::Pad{:replicate})
    _leftborder!(out, img, kernel, Ibegin, indleft, Iend, img[Ibegin, indleft[1], Iend])
end
function _leftborder!{T,k,l}(out, img, kernel::TriggsSdika{T,k,l}, Ibegin, indleft, Iend, iminus)
    uminus = iminus/(1-kernel.asum)
    n = 0
    for i in indleft
        n += 1
        tmp = one(T)*img[Ibegin, i, Iend]
        for j = 1:n-1
            tmp += kernel.a[j]*out[Ibegin, i-j, Iend]
        end
        for j = n:k
            tmp += kernel.a[j]*uminus
        end
        out[Ibegin, i, Iend] = tmp
    end
    out
end

# Implements Triggs & Sdika, Eqs 14-15
function rightborder!(out, img, kernel, Ibegin, indright, Iend, border::Fill)
    _rightborder!(out, img, kernel, Ibegin, indright, Iend, convert(eltype(img), border.value))
end
function rightborder!(out, img, kernel, Ibegin, indright, Iend, border::Pad{:replicate})
    _rightborder!(out, img, kernel, Ibegin, indright, Iend, img[Ibegin, indright[end], Iend])
end
function _rightborder!{T,k,l}(out, img, kernel::TriggsSdika{T,k,l}, Ibegin, indright, Iend, iplus)
    # The final value from forward-filtering was not calculated, so do that here
    i = last(indright)
    tmp = one(T)*img[Ibegin, i, Iend]
    for j = 1:k
        tmp += kernel.a[j]*out[Ibegin, i-j, Iend]
    end
    out[Ibegin, i, Iend] = tmp
    # Initialize the v values at and beyond the right edge
    uplus = iplus/(1-kernel.asum)
    vplus = uplus/(1-kernel.bsum)
    vright = kernel.M * rightΔu(out, uplus, Ibegin, last(indright), Iend, kernel) + vplus
    out[Ibegin, last(indright), Iend] = vright[1]
    # Propagate inward
    n = 1
    for i in last(indright)-1:-1:first(indright)
        n += 1
        tmp = one(T)*out[Ibegin, i, Iend]
        for j = 1:n-1
            tmp += kernel.b[j]*out[Ibegin, i+j, Iend]
        end
        for j = n:l
            tmp += kernel.b[j]*vright[j-n+2]
        end
        out[Ibegin, i, Iend] = tmp
    end
    out
end

# Part of Triggs & Sdika, Eq. 14
function rightΔu{T,l}(img, uplus, Ibegin, i, Iend, kernel::TriggsSdika{T,3,l})
    @inbounds ret = SVector(img[Ibegin, i,   Iend]-uplus,
                            img[Ibegin, i-1, Iend]-uplus,
                            img[Ibegin, i-2, Iend]-uplus)
    ret
end

### Utilities

filter_type{S,T}(img::AbstractArray{S}, kernel::AbstractArray{T}) = typeof(zero(S)*zero(T) + zero(S)*zero(T))
filter_type{S,T}(img::AbstractArray{S}, kernel::Tuple{AbstractArray{T},Vararg{AbstractArray{T}}}) = typeof(zero(S)*zero(T) + zero(S)*zero(T))
filter_type{S,T}(img::AbstractArray{S}, kernel::Tuple{IIRFilter{T},Vararg{IIRFilter{T}}}) = typeof(zero(S)*zero(T) + zero(S)*zero(T))

factorkernel(kernel::AbstractArray) = (copy(kernelshift(indices(kernel), kernel)),)  # copy to ensure consistency

function factorkernel{T}(kernel::AbstractMatrix{T})
    inds = indices(kernel)
    m, n = map(length, inds)
    kern = Array{T}(m, n)
    copy!(kern, 1:m, 1:n, kernel, inds[1], inds[2])
    factorstridedkernel(inds, kern)
end

function factorstridedkernel(inds, kernel::StridedMatrix)
    SVD = svdfact(kernel)
    U, S, Vt = SVD[:U], SVD[:S], SVD[:Vt]
    separable = true
    EPS = sqrt(eps(eltype(S)))
    for i = 2:length(S)
        separable &= (abs(S[i]) < EPS)
    end
    if !separable
        ks = kernelshift(inds, kernel)
        return (copy(ks), dummykernel(indices(ks)))
    end
    s = S[1]
    u, v = U[:,1:1], Vt[1:1,:]
    ss = sqrt(s)
    (kernelshift((inds[1], dummyind(inds[1])), ss*u),
     kernelshift((dummyind(inds[2]), inds[2]), ss*v))
end

dummyind(::Base.OneTo) = Base.OneTo(1)
dummyind(::AbstractUnitRange) = 0:0

dummykernel{N}(inds::Indices{N}) = similar(dims->ones(ntuple(d->1,Val{N})), map(dummyind, inds))

prod_kernel(kern::AbstractArray) = kern
prod_kernel(kern::AbstractArray, kern1, kerns...) = prod_kernel(kern.*kern1, kerns...)
prod_kernel{N}(::Type{Val{N}}, args...) = prod_kernel(Val{N}, prod_kernel(args...))
prod_kernel{_,N}(::Type{Val{N}}, kernel::AbstractArray{_,N}) = kernel
function prod_kernel{N}(::Type{Val{N}}, kernel::AbstractArray)
    inds = indices(kernel)
    newinds = fill_to_length(inds, oftype(inds[1], 0:0), Val{N})
    reshape(kernel, newinds)
end
kernelshift{N}(inds::NTuple{N,Base.OneTo}, A::StridedArray) = _kernelshift(inds, A)
kernelshift{N}(inds::NTuple{N,Base.OneTo}, A) = _kernelshift(inds, A)
function _kernelshift(inds, A)
    warn("assuming that the origin is at the center of the kernel; to avoid this warning, call `centered(kernel)` or use an OffsetArray")  # this may be necessary long-term?
    centered(A)
end
kernelshift(inds::Any, A::StridedArray) = OffsetArray(A, inds...)
function kernelshift(inds::Any, A)
    @assert indices(A) == inds
    A
end

"""
    centered(kernel) -> shiftedkernel

Shift the origin-of-coordinates to the center of `kernel`. The
center-element of `kernel` will be accessed by `shiftedkernel[0, 0,
...]`.

This function makes it easy to supply kernels using regular Arrays,
and provides compatibility with other languages that do not support
arbitrary indices.

See also: imfilter.
"""
centered(A::AbstractArray) = OffsetArray(A, map(n->-((n+1)>>1), size(A)))

filter_algorithm(out, img, kernel) = FIR()
filter_algorithm(out, img, kernel::Tuple{IIRFilter,Vararg{IIRFilter}}) = IIR()

function normalize_separable!{N}(r::AbstractResource, A, kernels::NTuple{N}, border)
    inds = indices(A)
    function imfilter_inplace!(r, a, kern, border)
        imfilter!(r, a, a, (kern,), border)
    end
    filtdims = ntuple(d->imfilter_inplace!(r, similar(dims->ones(dims), inds[d]), kernels[d], border), Val{N})
    normalize_dims!(A, filtdims)
end

function normalize_dims!{T,N}(A::AbstractArray{T,N}, factors::NTuple{N})
    for I in CartesianRange(indices(A))
        tmp = A[I]/factors[1][I[1]]
        for d = 2:N
            tmp /= factors[d][I[d]]
        end
        A[I] = tmp
    end
    A
end

_tail(R::CartesianRange{CartesianIndex{0}}) = R
_tail(R::CartesianRange) = CartesianRange(CartesianIndex(tail(R.start.I)),
                                          CartesianIndex(tail(R.stop.I)))

to_ranges(R::CartesianRange) = map((b,e)->b:e, R.start.I, R.stop.I)

_reshape{_,N}(A::OffsetArray{_,N}, ::Type{Val{N}}) = A
_reshape{N}(A::OffsetArray, ::Type{Val{N}}) = OffsetArray(reshape(parent(A), Val{N}), fill_to_length(A.offsets, -1, Val{N}))
_reshape{N}(A::AbstractArray, ::Type{Val{N}}) = reshape(A, Val{N})
