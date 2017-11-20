module MapWindow

using DataStructures, TiledIteration
using ..ImageFiltering: BorderSpecAny, Pad, Fill, borderinstance, _interior, padindex, imfilter
using Base: Indices, tail

export mapwindow

"""
    mapwindow(f, img, window, [border="replicate"], [imginds=indices(img)]) -> imgf

Apply `f` to sliding windows of `img`, with window size or indices
specified by `window`. For example, `mapwindow(median!, img, window)`
returns an `Array` of values similar to `img` (median-filtered, of
course), whereas `mapwindow(extrema, img, window)` returns an `Array`
of `(min,max)` tuples over a window of size `window` centered on each
point of `img`.

The function `f` receives a buffer `buf` for the window of data
surrounding the current point. If `window` is specified as a
Dims-tuple (tuple-of-integers), then all the integers must be odd and
the window is centered around the current image point. For example, if
`window=(3,3)`, then `f` will receive an Array `buf` corresponding to
offsets `(-1:1, -1:1)` from the `imgf[i,j]` for which this is
currently being computed. Alternatively, `window` can be a tuple of
AbstractUnitRanges, in which case the specified ranges are used for
`buf`; this allows you to use asymmetric windows if needed.

`border` specifies how the edges of `img` should be handled; see
`imfilter` for details.

Finally `imginds` allows to omit unnecessary computations, if you want to do things
like `mapwindow` on a subimage, or a strided variant of mapwindow.
It works as follows:
```julia
mapwindow(f, img, window, border, (2:5, 1:2:7)) == mapwindow(f,img,window,border)[2:5, 1:2:7]
```
Except more efficiently because it omits computation of the unused values.

For functions that can only take `AbstractVector` inputs, you might have to
first specialize `default_shape`:

```julia
f = v->quantile(v, 0.75)
ImageFiltering.MapWindow.default_shape(::typeof(f)) = vec
```

and then `mapwindow(f, img, (m,n))` should filter at the 75th quantile.

See also: [`imfilter`](@ref).
"""
function mapwindow(f, img, window, border="replicate", imginds=indices(img); callmode=:copy!)
    _mapwindow_kernel(replace_function(f),
              img,
              resolve_window(window),
              resolve_border(border),
              resolve_imginds(imginds),
              default_shape(f);
              callmode=callmode)
end

function resolve_window(window::Dims)
    all(isodd(w) for w in window) || error("entries in window must be odd, got $window")
    halfsize = map(w->w>>1, window)
    map(h -> -h:h,halfsize)
end
function resolve_window(window::Integer)
    isodd(window) || error("window must be odd, got $window")
    h = window>>1
    -h:h
end
resolve_window(window::AbstractArray) = resolve_window((window...,))
resolve_window(window::AbstractUnitRange) = (window,)
resolve_window(window::Indices) = window

resolve_border(border::AbstractString) = borderinstance(border)
resolve_border(border::BorderSpecAny) = border

resolve_imginds(r::Range) = (r,)
resolve_imginds(imginds) = imginds

abstract type _IndexTransformer end

struct _AffineTransformer{N} <: _IndexTransformer
    offset::NTuple{N,Int}
    stride::NTuple{N,Int}
end
@inline function Base.getindex(t::_AffineTransformer, inds::CartesianIndex)
    CartesianIndex(t.offset .+ t.stride .* inds.I)
end

struct _OffsetTransformer{N} <: _IndexTransformer
    offset::NTuple{N,Int}
end
@inline function Base.getindex(t::_OffsetTransformer, inds::CartesianIndex)
    CartesianIndex(t.offset .+ inds.I)
end

struct _IdentityTransformer <: _IndexTransformer end
@inline Base.getindex(t::_IdentityTransformer, inds) = inds

function _IndexTransformer(ranges)
    stride = map(step, ranges)
    offset1 = map(first, ranges)
    offset = offset1 .- stride
    _AffineTransformer(offset, stride)
end

function _IndexTransformer(ranges::NTuple{N,Base.OneTo}) where {N}
    _IdentityTransformer()
end

function _IndexTransformer(ranges::NTuple{N,AbstractUnitRange}) where {N}
    offset1 = map(first, ranges)
    offset = offset1 .- 1
    _OffsetTransformer(offset)
end

compute_output_range(r::AbstractUnitRange) = r
compute_output_range(r::Range) = Base.OneTo(length(r))

function compute_output_indices(imginds)
    ranges = map(compute_output_range, imginds)
    # Base.similar does not like if some but not all ranges are Base.OneTo
    homogenize(ranges)
end
homogenize(ranges::NTuple{N, Range}) where {N}   = map(r-> first(r):step(r):last(r), ranges)
homogenize(ranges::NTuple{N, AbstractUnitRange}) where{N} = map(r-> first(r):last(r), ranges)
homogenize(ranges::NTuple{N, Base.OneTo}) where {N} = ranges

# Return indices of elements of `r` that are also elements of `full`.
function _intersectionindices(full::AbstractUnitRange, r::Range)
    r_sub = intersect(full, r)
    if isempty(r_sub)
        ret = 1:0
    else
        ret = _indexof(r,first(r_sub)):_indexof(r,last(r_sub))
    end
    @assert intersect(full, r) == r[ret]
    ret
end

function _indexof(r::Range, x)
    T = eltype(r)
    @assert x ∈ r
    i = one(T) + T((x - first(r)) / step(r))
    @assert r[i] == x
    i
end

function _indices_of_interiour_range(
        fullimgr::AbstractUnitRange, 
        imgr::Range,
        kerr::Range)
    kmin, kmax = extrema(kerr)
    idx1 = _intersectionindices(fullimgr, kmin + imgr)
    idx2 = _intersectionindices(fullimgr, kmax + imgr)
    idx = intersect(idx1, idx2)
    @assert imgr[idx] + kmin ⊆ fullimgr
    @assert imgr[idx] + kmax ⊆ fullimgr
    idx
end

function _indices_of_interiour_indices(fullimginds, imginds, kerinds)
    map(_indices_of_interiour_range, fullimginds, imginds, kerinds)
end

# replace median by ... outside of _mapwindow_kernel
function _mapwindow_kernel(f,
                    img::AbstractArray{T,N},
                    window::NTuple{N,AbstractUnitRange},
                    border::BorderSpecAny,
                    imginds::NTuple{N, Range},
                    shape=default_shape(f);
                    callmode::Symbol=:copy!) where {T,N}
    
    if callmode != :copy!
        # TODO: implement :view
        error("callmode $callmode not supported")
    end
    outinds = compute_output_indices(imginds)
    @assert map(length, imginds) == map(length, outinds)
    
    indind_full = map(r -> Base.OneTo(length(r)), imginds)
    indind_inner = _indices_of_interiour_indices(indices(img), imginds, window)
    Rindind_full = CartesianRange(indind_full)
    Rindind_inner = CartesianRange(indind_inner)
    
    outindtrafo = _IndexTransformer(outinds)
    imgindtrafo = _IndexTransformer(imginds)
    
    buf = Array{T}(map(length, window))
    bufrs = shape(buf)
    Rbuf = CartesianRange(size(buf))
    # To allocate the output, we have to evaluate f once on realistic values
    Iimg = imgindtrafo[first(Rindind_full)]
    offset = CartesianIndex(map(w->first(w)-1, window))
    copy_win!(buf, img, Iimg, border, offset)
    out = similar(img, typeof(f(bufrs)), outinds)
    for II ∈ Rindind_inner
        Iimg = imgindtrafo[II]
        Iout = outindtrafo[II]
        Rwin = CartesianRange(map(+, window, Iimg.I))
        copy!(buf, Rbuf, img, Rwin)
        @inbounds out[Iout] = f(bufrs)
    end
    # Now pick up the edge points we skipped over above
    Rindind_edge = EdgeIterator(Rindind_full, Rindind_inner)
    for II ∈ Rindind_edge
        Iimg = imgindtrafo[II]
        Iout = outindtrafo[II]
        copy_win!(buf, img, Iimg, border, offset)
        out[Iout] = f(bufrs)
    end
    out
end

# For copying along the edge of the image
function copy_win!(buf::AbstractArray{T,N}, img, I, border::Pad, offset) where {T,N}
    win_inds = map(+, indices(buf), (I+offset).I)
    win_img_inds = map(intersect, indices(img), win_inds)
    padinds = map((inner,outer)->padindex(border, inner, outer), win_img_inds, win_inds)
    docopy!(buf, img, padinds)
    buf
end
docopy!(buf, img, padinds::NTuple{1}) = buf[:] = view(img, padinds[1])
docopy!(buf, img, padinds::NTuple{2}) = buf[:,:] = view(img, padinds[1], padinds[2])
docopy!(buf, img, padinds::NTuple{3}) = buf[:,:,:] = view(img, padinds[1], padinds[2], padinds[3])
@inline function docopy!(buf, img, padinds::NTuple{N}) where N
    @show N
    colons = ntuple(d->Colon(), Val{N})
    buf[colons...] = view(img, padinds...)
end

function copy_win!(buf::AbstractArray{T,N}, img, I, border::Fill, offset) where {T,N}
    R = CartesianRange(indices(img))
    Ioff = I+offset
    for J in CartesianRange(indices(buf))
        K = Ioff+J
        buf[J] = K ∈ R ? img[K] : convert(eltype(img), border.value)
    end
    buf
end

### Optimizations for particular window-functions

mapwindow(::typeof(extrema), A::AbstractArray, window::Dims) = extrema_filter(A, window)
mapwindow(::typeof(extrema), A::AbstractVector, window::Integer) = extrema_filter(A, (window,))

# Max-min filter

# This is an implementation of the Lemire max-min filter
# http://arxiv.org/abs/cs.DS/0610046

# Monotonic wedge
struct Wedge{T}
    L::CircularDeque{T}
    U::CircularDeque{T}
end
Wedge{T}(n::Integer) where {T} = Wedge(CircularDeque{T}(n), CircularDeque{T}(n))

function Base.push!(W::Wedge, i::Integer)
    push!(W.L, i)
    push!(W.U, i)
    W
end

function addtoback!(W::Wedge, A, i, J)
    mn, mx = A[i, J]
    @inbounds while !isempty(W.L) && mn < A[back(W.L), J][1]
        pop!(W.L)
    end
    @inbounds while !isempty(W.U) && mx > A[back(W.U), J][2]
        pop!(W.U)
    end
    push!(W.L, i)
    push!(W.U, i)
    W
end

function Base.empty!(W::Wedge)
    empty!(W.L)
    empty!(W.U)
    W
end

@inline function getextrema(A, W::Wedge, J)
    (A[front(W.L), J][1], A[front(W.U), J][2])
end

"""
    extrema_filter(A, window) --> Array{(min,max)}

Calculate the running min/max over a window of width `window[d]` along
dimension `d`, centered on the current point. The returned array has
the same indices as the input `A`.
"""
function extrema_filter(A::AbstractArray{T,N}, window::NTuple{N,Integer}) where {T,N}
    _extrema_filter!([(a,a) for a in A], window...)
end
extrema_filter(A::AbstractArray, window::AbstractArray) = extrema_filter(A, (window...,))
extrema_filter(A::AbstractArray, window) = error("`window` must have the same number of entries as dimensions of `A`")

extrema_filter(A::AbstractArray{T,N}, window::Integer) where {T,N} = extrema_filter(A, ntuple(d->window, Val{N}))

function _extrema_filter!(A::Array, w1, w...)
    if w1 > 1
        a = first(A)
        cache = ntuple(i->a, w1>>1)
        _extrema_filter1!(A, w1, cache)
    end
    _extrema_filter!(permutedims(A, [2:ndims(A);1]), w...)
end
_extrema_filter!(A::Array) = A

# Extrema-filtering along "columns" (dimension 1). This implements Lemire
# Algorithm 1, with the following modifications:
#   - multidimensional array support by looping over trailing dimensions
#   - working with min/max pairs rather than plain values, to
#     facilitate multidimensional processing
#   - output for all points of the array, handling the edges as max-min
#     over halfwindow on either side
function _extrema_filter1!(A::AbstractArray{Tuple{T,T}}, window::Int, cache) where T
    # Initialise the internal wedges
    # U[1], L[1] are the location of the global (within the window) maximum and minimum
    # U[2], L[2] are the maximum and minimum over (U1, end] and (L1, end], respectively
    W = Wedge{Int}(window+1)
    tmp = Array{Tuple{T,T}}(window)
    c = z = first(cache)

    inds = indices(A)
    inds1 = inds[1]
    halfwindow = window>>1
    iw = min(last(inds1), first(inds1)+window-1)
    for J in CartesianRange(tail(inds))
        empty!(W)
        # Leading edge. We can't overwrite any values yet in A because
        # we'll need them again in later computations.
        for i = first(inds1):iw
            addtoback!(W, A, i, J)
            c, cache = cyclecache(cache, getextrema(A, W, J))
        end
        # Process the rest of the "column"
        for i = iw+1:last(inds1)
            A[i-window, J] = c
            if i == window+front(W.U)
                shift!(W.U)
            end
            if i == window+front(W.L)
                shift!(W.L)
            end
            addtoback!(W, A, i, J)
            c, cache = cyclecache(cache, getextrema(A, W, J))
        end
        for i = last(inds1)-window+1:last(inds1)-1
            if i >= first(inds1)
                A[i, J] = c
            end
            if i == front(W.U)
                shift!(W.U)
            end
            if i == front(W.L)
                shift!(W.L)
            end
            c, cache = cyclecache(cache, getextrema(A, W, J))
        end
        A[last(inds1), J] = c
    end
    A
end

# This is slightly faster than a circular buffer
@inline cyclecache(b, x) = b[1], (Base.tail(b)..., x)

replace_function(f) = f
replace_function(::typeof(median!)) = function(v)
    inds = indices(v,1)
    Base.middle(Base.select!(v, (first(inds)+last(inds))÷2, Base.Order.ForwardOrdering()))
end

default_shape(::Any) = identity
default_shape(::typeof(median!)) = vec

end
