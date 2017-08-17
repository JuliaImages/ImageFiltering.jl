module MapWindow

using DataStructures, TiledIteration
using ..ImageFiltering: BorderSpecAny, Pad, Fill, borderinstance, _interior, padindex, imfilter
using Base: Indices, tail

export mapwindow

"""
    mapwindow(f, img, window, [border="replicate"]) -> imgf

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

For functions that can only take `AbstractVector` inputs, you might have to
first specialize `default_shape`:

```julia
f = v->quantile(v, 0.75)
ImageFiltering.MapWindow.default_shape(::typeof(f)) = vec
```

and then `mapwindow(f, img, (m,n))` should filter at the 75th quantile.

See also: [`imfilter`](@ref).
"""
function mapwindow(f, img::AbstractArray, window::Dims, args...; kwargs...)
    all(isodd(w) for w in window) || error("entries in window must be odd, got $window")
    halfsize = map(w->w>>1, window)
    mapwindow(f, img, map(h->-h:h, halfsize), args...; kwargs...)
end
function mapwindow(f, img::AbstractVector, window::Integer, args...; kwargs...)
    isodd(window) || error("window must be odd, got $window")
    h = window>>1
    mapwindow(f, img, (-h:h,), args...; kwargs...)
end

mapwindow(f, img::AbstractArray, window::Indices; kwargs...) =
    mapwindow(f, img, window, "replicate"; kwargs...)
mapwindow(f, img::AbstractVector, window::AbstractUnitRange; kwargs...) =
    mapwindow(f, img, (window,); kwargs...)

function mapwindow(f, img::AbstractArray, window::Indices, border::AbstractString;
                   kwargs...)
    mapwindow(f, img, window, borderinstance(border); kwargs...)
end
function mapwindow(f, img::AbstractVector, window::AbstractUnitRange, border::AbstractString;
                   kwargs...)
    mapwindow(f, img, (window,), border; kwargs...)
end

mapwindow(f, img, window::AbstractArray, args...; kwargs...) = mapwindow(f, img, (window...,), args...; kwargs...)

function mapwindow(f,
                   img::AbstractArray{T,N},
                   window::Indices{N},
                   border::BorderSpecAny;
                   callmode=:copy!) where {T,N}
    _mapwindow(replace_function(f), img, window, border, default_shape(f); callmode=callmode)
end
function _mapwindow(f,
                    img::AbstractArray{T,N},
                    window::Indices{N},
                    border::BorderSpecAny,
                    shape=default_shape(f);
                    callmode=:copy!) where {T,N}
    inds = indices(img)
    inner = _interior(inds, window)
    if callmode == :copy!
        buf = Array{T}(map(length, window))
        bufrs = shape(buf)
        Rbuf = CartesianRange(size(buf))
        offset = CartesianIndex(map(w->first(w)-1, window))
        # To allocate the output, we have to evaluate f once
        Rinner = CartesianRange(inner)
        if !isempty(Rinner)
            Rwin = CartesianRange(map(+, window, first(Rinner).I))
            copy!(buf, Rbuf, img, Rwin)
            out = similar(img, typeof(f(bufrs)))
            # Handle the interior
            for I in Rinner
                Rwin = CartesianRange(map(+, window, I.I))
                copy!(buf, Rbuf, img, Rwin)
                out[I] = f(bufrs)
            end
        else
            copy_win!(buf, img, first(CartesianRange(inds)), border, offset)
            out = similar(img, typeof(f(bufrs)))
        end
        # Now pick up the edge points we skipped over above
        for I in EdgeIterator(inds, inner)
            copy_win!(buf, img, I, border, offset)
            out[I] = f(bufrs)
        end
    else
        # TODO: implement :view
        error("callmode $callmode not supported")
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
