module MapWindow

using DataStructures, TiledIteration
using ..ImageFiltering: BorderSpecAny, Pad, Fill, Inner,
    borderinstance, _interior, padindex, imfilter
using Base: Indices, tail
using Statistics
using OffsetArrays

export mapwindow, mapwindow!

"""
    mapwindow(f, img, window; [border="replicate"], [indices=axes(img)]) -> imgf

Apply `f` to sliding windows of `img`, with window size or axes
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

Finally `indices` allows to omit unnecessary computations, if you want to do things
like `mapwindow` on a subimage, or a strided variant of mapwindow.
It works as follows:
```julia
mapwindow(f, img, window, indices=(2:5, 1:2:7)) == mapwindow(f,img,window)[2:5, 1:2:7]
```
Except more efficiently because it omits computation of the unused values.

Because the data in the buffer `buf` that is received by `f` is copied from `img`, and the
buffer's memory is reused, `f` should not return references to `buf`.
This
```julia
f = buf->copy(buf) # as opposed to f = buf->buf
mapwindow(f, img, window, indices=(2:5, 1:2:7))
```
would work as expected.

For functions that can only take `AbstractVector` inputs, you might have to
first specialize `default_shape`:

```julia
f = v->quantile(v, 0.75)
ImageFiltering.MapWindow.default_shape(::typeof(f)) = vec
```

and then `mapwindow(f, img, (m,n))` should filter at the 75th quantile.

See also: [`imfilter`](@ref).
"""
function mapwindow(f::F, img, window; border="replicate",
                   indices=default_imginds(img, window, border), callmode=:copy!) where F
    if callmode != :copy!
        error("Only callmode=:copy! is currently supported")
    end
    _mapwindow(replace_function(f),
              img,
              resolve_window(window),
              resolve_border(border),
              resolve_imginds(indices))
end

# a unit range `s` having the same values as `r` and `axes(s) == (s,)`,
# which keeps its axes with broadcasting (e.g., `axes(s.+1) == axes(s)`),
# unlike Base.IdentityUnitRange
self_offset(r::AbstractUnitRange) = OffsetArrays.IdOffsetRange(1:length(r), first(r)-one(eltype(r)))

function default_imginds(img, window, border)
    axes(img)
end
function default_imginds(img, window, border::Inner)
    imginds = axes(img)
    win = resolve_window(window)
    indind = _indices_of_interiour_indices(imginds, imginds, win)::Tuple{Vararg{AbstractUnitRange}}
    map(self_offset, indind)
end

function _mapwindow(f::F, img, window, border, imginds) where F
    out = allocate_output(f, img, window, border, imginds)
    mapwindow_kernel!(f, out, img, window, border, imginds)
end

"""
    mapwindow!(f, out, img, window; border="replicate", indices=axes(img))

Variant of [`mapwindow`](@ref), with preallocated output.
If `out` and `img` have overlapping memory regions, behaviour is undefined.
"""
function mapwindow!(f, out, img, window; border="replicate",
                    indices=default_imginds(img, window, border))
    mapwindow_kernel!(replace_function(f),
              out,
              img,
              resolve_window(window),
              resolve_border(border),
              resolve_imginds(indices))
end

function median_fast!(v)
    # median! calls partialsort! which has keyword arguments. Keyword arguments are slow.
    # This replaces median! with a more efficient implementation free of keyword arguments.
    inds = axes(v,1)
    Statistics.middle(Base.partialsort!(v, (first(inds)+last(inds))÷2, Base.Order.ForwardOrdering()))
end

replace_function(f) = f
replace_function(::typeof(median!)) = median_fast!

function resolve_window(window::Dims)
    all(isodd(w) for w in window) || error("entries in window must be odd, got $window")
    halfsize = map(w->w>>1, window)
    map(h -> -h:h,halfsize)
end
function resolve_window(window::Integer)
    isodd(window) || error("window must be odd, got $window")
    h = window>>1
    (-h:h,)
end
resolve_window(window::AbstractArray) = resolve_window((window...,))
resolve_window(window::AbstractUnitRange) = (window,)
resolve_window(window::Indices) = window

# avoid method ambiguity between ::Dims and ::Indices
resolve_window(window::Tuple{}) = throw(ArgumentError("empty window"))

resolve_border(border::AbstractString) = borderinstance(border)
resolve_border(border::BorderSpecAny) = border

resolve_imginds(r::AbstractRange) = (r,)
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

function _IndexTransformer(from_ranges::NTuple{N,AbstractUnitRange}, to_ranges) where {N}
    stride = map(step, to_ranges)
    offset = first.(to_ranges) .- first.(from_ranges) .* stride
    _AffineTransformer(offset, stride)
end

function _IndexTransformer(from_ranges::NTuple{N,AbstractUnitRange}, to_ranges::NTuple{N,AbstractUnitRange}) where {N}
    offset = first.(to_ranges) .- first.(from_ranges)
    _OffsetTransformer(offset)
end

function _IndexTransformer(::NTuple{N,Base.OneTo}, ::NTuple{N,Base.OneTo}) where {N}
    _IdentityTransformer()
end

compute_output_range(r::AbstractUnitRange) = r
compute_output_range(r::AbstractRange) = Base.OneTo(length(r))

function compute_output_indices(imginds)
    ranges = map(i->compute_output_range(axes(i,1)), imginds)
    # Base.similar does not like if some but not all ranges are Base.OneTo
    homogenize(ranges)
end
homogenize(ranges::NTuple{N, AbstractRange}) where {N}   = map(r-> first(r):step(r):last(r), ranges)
homogenize(ranges::NTuple{N, AbstractUnitRange}) where{N} = map(r-> first(r):last(r), ranges)
homogenize(ranges::NTuple{N, Base.OneTo}) where {N} = ranges

# Return indices of elements of `r` that are also elements of `full`.
function _intersectionindices(full::AbstractUnitRange, r::AbstractRange)
    r_sub = intersect(full, r)
    if isempty(r_sub)
        ret = 1:0
    else
        ret = _indexof(r,first(r_sub)):_indexof(r,last(r_sub))
    end
    @assert r_sub == r[ret] || isempty(r_sub) && isempty(ret)
    ret
end

function _indexof(r::AbstractRange, x)
    T = eltype(axes(r,1))
    @assert x ∈ r
    i = T(firstindex(r) + (x - first(r)) / step(r))
    @assert r[i] == x
    i
end

function _indices_of_interiour_range(
        fullimgr::AbstractUnitRange,
        imgr::AbstractRange,
        kerr::AbstractRange)
    kmin, kmax = extrema(kerr)
    idx1 = _intersectionindices(fullimgr, kmin .+ imgr)
    idx2 = _intersectionindices(fullimgr, kmax .+ imgr)
    idx = intersect(idx1, idx2)
    @assert imgr[idx] .+ kmin ⊆ fullimgr
    @assert imgr[idx] .+ kmax ⊆ fullimgr
    idx
end

function _indices_of_interiour_indices(fullimginds, imginds, kerinds)
    map(_indices_of_interiour_range, fullimginds, imginds, kerinds)
end

function allocate_output(f::F, img, window, border, imginds) where F
    T = compute_output_eltype(f, img, window, border, imginds)
    outinds = compute_output_indices(imginds)
    similar(img, T, outinds)
end

function allocate_buffer(f::F, img, window) where F
    T = eltype(img)
    buf = Array{T}(undef,map(length, window))
    bufrs = default_shape(f)(buf)
    buf, bufrs
end

function compute_output_eltype(f::F, img, window, border, imginds) where F
    buf, bufrs = allocate_buffer(f, img, window)
    make_buffer_values_realistic!(buf, img, window, border, imginds)
    typeof(f(bufrs))
end

function make_buffer_values_realistic!(buf, img, window, border::Inner, imginds)
    x = oneunit(eltype(img))
    fill!(buf, x)
end

function make_buffer_values_realistic!(buf, img, window, border, imginds)
    Iimg = CartesianIndex(map(first, imginds))
    offset = CartesianIndex(map(w->first(w)-1, window))
    copy_win!(buf, img, Iimg, border, offset)
end

function mapwindow_kernel!(f,
                    out::AbstractArray{S,N},
                    img::AbstractArray{T,N},
                    window::NTuple{N,AbstractUnitRange},
                    border::BorderSpecAny,
                    imginds::NTuple{N, AbstractRange}) where {S,T,N}

    @assert map(length, imginds) == map(length, axes(out))

    indind_full = map(r -> axes(r,1), imginds)
    indind_inner = _indices_of_interiour_indices(axes(img), imginds, window)
    Rindind_full = CartesianIndices(indind_full)
    Rindind_inner = CartesianIndices(indind_inner)

    outindtrafo = _IndexTransformer(indind_full, axes(out))
    imgindtrafo = _IndexTransformer(indind_full, imginds)

    buf, bufrs = allocate_buffer(f, img, window)
    Rbuf = CartesianIndices(size(buf))
    for II ∈ Rindind_inner
        Iimg = imgindtrafo[II]
        Iout = outindtrafo[II]
        Rwin = CartesianIndices(map((w,o) -> w .+ o, window, Tuple(Iimg)))
        copyto!(buf, Rbuf, img, Rwin)
        out[Iout] = f(bufrs)
    end
    # Now pick up the edge points we skipped over above
    Rindind_edge = EdgeIterator(Rindind_full, Rindind_inner)
    offset = CartesianIndex(map(w->first(w)-1, window))
    for II ∈ Rindind_edge
        Iimg = imgindtrafo[II]
        Iout = outindtrafo[II]
        copy_win!(buf, img, Iimg, border, offset)
        out[Iout] = f(bufrs)
    end
    out
end


# For copying along the edge of the image
function copy_win!(buf::AbstractArray, img, I, border::Pad, offset)
    win_inds = map((x,y)->x .+ y, axes(buf), Tuple(I) .+ Tuple(offset))
    win_img_inds = map(intersect, axes(img), win_inds)
    padinds = map((inner,outer)->padindex(border, inner, outer), win_img_inds, win_inds)
    docopy!(buf, img, padinds)
    buf
end
docopy!(buf, img, padinds::NTuple{1}) = buf[:] = view(img, padinds[1])
docopy!(buf, img, padinds::NTuple{2}) = buf[:,:] = view(img, padinds[1], padinds[2])
docopy!(buf, img, padinds::NTuple{3}) = buf[:,:,:] = view(img, padinds[1], padinds[2], padinds[3])
@inline function docopy!(buf, img, padinds::NTuple{N}) where N
    colons = ntuple(d->Colon(), Val{N}())
    buf[colons...] = view(img, padinds...)
end

function copy_win!(buf::AbstractArray, img, I, border::Fill, offset)
    R = CartesianIndices(axes(img))
    Ioff = I+offset
    for J in CartesianIndices(axes(buf))
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
    @inbounds while !isempty(W.L) && mn < A[last(W.L), J][1]
        pop!(W.L)
    end
    @inbounds while !isempty(W.U) && mx > A[last(W.U), J][2]
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
    (A[first(W.L), J][1], A[first(W.U), J][2])
end

"""
    extrema_filter(A, window) --> Array{(min,max)}

Calculate the running min/max over a window of width `window[d]` along
dimension `d`, centered on the current point. The returned array has
the same axes as the input `A`.
"""
function extrema_filter(A::AbstractArray{T,N}, window::NTuple{N,Integer}) where {T,N}
    _extrema_filter!([(a,a) for a in A], window...)
end
extrema_filter(A::AbstractArray, window::AbstractArray) = extrema_filter(A, (window...,))
extrema_filter(A::AbstractArray, window) = error("`window` must have the same number of entries as dimensions of `A`")

extrema_filter(A::AbstractArray{T,N}, window::Integer) where {T,N} = extrema_filter(A, ntuple(d->window, Val{N}))

function _extrema_filter!(A::AbstractArray, w1, w...)
    if w1 > 1
        a = first(A)
        if w1 <= 20
            cache = ntuple(i->a, w1>>1)      # this line is not inferrable, and contributes to latency via...
            _extrema_filter1!(A, w1, cache)  # ...extensive specialization of _extrema_filter1! (on typeof(A) and value of w1)
        else
            n = w1>>1
            cache = CircularDeque{typeof(a)}(n)
            for i = 1:n
                push!(cache, a)
            end
            _extrema_filter1!(A, w1, cache)
        end
    end
    if ndims(A) > 1
        _extrema_filter!(permutedims(A, [2:ndims(A);1:1]), w...)
    else
        return A
    end
end
_extrema_filter!(A::AbstractArray) = A

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
    tmp = Array{Tuple{T,T}}(undef, window)
    c = z = first(cache)

    inds = axes(A)
    inds1 = inds[1]
    halfwindow = window>>1
    iw = min(last(inds1), first(inds1)+window-1)
    for J in CartesianIndices(tail(inds))
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
            if i == window+first(W.U)
                popfirst!(W.U)
            end
            if i == window+first(W.L)
                popfirst!(W.L)
            end
            addtoback!(W, A, i, J)
            c, cache = cyclecache(cache, getextrema(A, W, J))
        end
        for i = last(inds1)-window+1:last(inds1)-1
            if i >= first(inds1)
                A[i, J] = c
            end
            if i == first(W.U)
                popfirst!(W.U)
            end
            if i == first(W.L)
                popfirst!(W.L)
            end
            c, cache = cyclecache(cache, getextrema(A, W, J))
        end
        A[last(inds1), J] = c
    end
    A
end

# This is slightly faster than a circular buffer
@inline cyclecache(b::Tuple, x) = b[1], (Base.tail(b)..., x)
@inline function cyclecache(b::CircularDeque, x)
    ret1 = popfirst!(b)
    push!(b, x)
    return ret1, b
end

default_shape(::Any) = identity
default_shape(::typeof(median_fast!)) = vec

## Deprecations
function mapwindow(f, img, window, border)
    Base.depwarn("mapwindow(f, img, window, $border) is deprecated, use `mapwindow(f, img, window, border=$border)` instead.", :mapwindow)
    mapwindow(f,img,window,border=border)
end

function mapwindow(f, img, window, border, indices)
    Base.depwarn("mapwindow(f, img, window, $border, $indices) is deprecated, use `mapwindow(f, img, window, border=$border, indices=$indices)` instead.", :mapwindow)
    mapwindow(f,img,window,border=border,indices=indices)
end

function mapwindow!(f, out, img, window, border)
    Base.depwarn("mapwindow!(f, out, img, window, $border) is deprecated, use `mapwindow!(f, out, img, window, border=$border)` instead.", :mapwindow!)
    mapwindow!(f,out,img,window,border=border)
end

function mapwindow!(f, out, img, window, border, indices)
    Base.depwarn("mapwindow!(f, out, img, window, $border, $indices) is deprecated, use `mapwindow!(f, out, img, window, border=$border, indices=$indices)` instead.", :mapwindow!)
    mapwindow!(f,out,img,window,border=border,indices=indices)
end

end
