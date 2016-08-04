# module Border

using OffsetArrays, CatIndices
using Base: Indices, tail

abstract AbstractBorder

"""
`Pad{Style,N}` is a type that stores choices about padding. `Style` is a
Symbol specifying the boundary conditions of the image, one of
`:replicate` (repeat edge values to infinity), `:circular` (image edges
"wrap around"), `:symmetric` (the image reflects between pixels), or
`:reflect` (the image reflects on the pixel grid). The default value is
`:replicate`. You can add custom boundary conditions by adding
addition methods for `padindex`.
"""
type Pad{Style,N} <: AbstractBorder
    lo::Dims{N}    # number to extend by on the lower edge for each dimension
    hi::Dims{N}    # number to extend by on the upper edge for each dimension
end

"""
    Pad{Style}(m, n, ...)

Pad the input image symmetrically, `m` pixels at the lower and upper edge of dimension 1, `n` pixels for dimension 2, and so forth.
"""
(::Type{Pad{Style}}){Style}(both::Int...) = Pad{Style}(both, both)
"""
    Pad{Style}((m,n))

Pad the input image symmetrically, `m` pixels at the lower and upper edge of dimension 1, `n` pixels for dimension 2.
"""
(::Type{Pad{Style}}){Style,N}(both::Dims{N}) = Pad{Style,N}(both, both)

(::Type{Pad{Style}}){Style  }(lo::Tuple{}, hi::Tuple{}) = Pad{Style,0}(lo, hi)

"""
    Pad{Style}(lo::Dims, hi::Dims)

Pad the input image by `lo` pixels at the lower edge, and `hi` pixels at the upper edge.
"""
(::Type{Pad{Style}}){Style,N}(lo::Dims{N}, hi::Dims{N}) = Pad{Style,N}(lo, hi)
(::Type{Pad{Style}}){Style,N}(lo::Dims{N}, hi::Tuple{}) = Pad{Style,N}(lo, ntuple(d->0,Val{N}))
(::Type{Pad{Style}}){Style,N}(lo::Tuple{}, hi::Dims{N}) = Pad{Style,N}(ntuple(d->0,Val{N}), hi)
(::Type{Pad{Style}}){Style,N}(inds::Indices{N}) = Pad{Style,N}(map(lo,inds), map(hi,inds))
"""
    Pad{Style}(kernel)

Given a filter array `kernel`, determine the amount of padding from the `indices` of `kernel`.
"""
(::Type{Pad{Style}}){Style}(kernel::AbstractArray) = Pad{Style}(indices(kernel))
(::Type{Pad{Style}}){Style}(factkernel::Tuple) = Pad{Style}(extremize(indices(factkernel[1]), tail(factkernel)...))
(::Type{Pad})(args...) = Pad{:replicate}(args...)

function padindices(img::AbstractArray, border::Pad)
    throw(ArgumentError("$border lacks the proper padding sizes for an array with $(ndims(img)) dimensions"))
end
function padindices{_,Style,N}(img::AbstractArray{_,N}, border::Pad{Style,N})
    _padindices(border, border.lo, indices(img), border.hi)
end
function padindices{P<:Pad}(img::AbstractArray, ::Type{P})
    throw(ArgumentError("must supply padding sizes to $P"))
end

# The 3-argument map is not inferrable, so do it manually
@inline _padindices(border, lo, inds, hi) =
    (padindex(border, lo[1], inds[1], hi[1]),
     _padindices(border, tail(lo), tail(inds), tail(hi))...)
_padindices(border, ::Tuple{}, ::Tuple{}, ::Tuple{}) = ()

"""
    padarray([T], img, border) --> imgpadded

Generate a padded image from an array `img` and a specification
`border` of the boundary conditions and amount of padding to
add. `border` can be a `Pad` or `Fill` object.

Optionally provide the element type `T` of `imgpadded`.
"""
padarray(img, border::Pad)  = padarray(eltype(img), img, border)
function padarray{T}(::Type{T}, img, border::Pad)
    inds = padindices(img, border)
    dest = similar(img, T, map(Base.indices1, inds))
    Base._unsafe_getindex!(dest, img, inds...)
    dest
end
padarray{P}(img, ::Type{P}) = img[padindices(img, P)...]      # just to throw the nice error

# Make this a separate type because the dispatch almost surely needs to be different
immutable Inner{N} <: AbstractBorder
    lo::Dims{N}
    hi::Dims{N}
end

# padarray(img, ::Inner) = img # do we need to make a copy here? (for consistency) or throw error?

"""
    Fill(val)
    Fill(val, lo, hi)

Pad the edges of the image with a constant value, `val`.

Optionally supply the extent of the padding, see `Pad`.
"""
type Fill{T,N} <: AbstractBorder
    value::T
    lo::Dims{N}
    hi::Dims{N}

    Fill(value::T) = new(value)
    Fill(value::T, lo::Dims{N}, hi::Dims{N}) = new(value, lo, hi)
end

Fill{T}(value::T) = Fill{T,0}(value)
Fill{T,N}(value::T, lo::Dims{N}, hi::Dims{N}) = Fill{T,N}(value, lo, hi)
Fill{T,N}(value::T, inds::Base.Indices{N}) = Fill{T,N}(value, map(lo,inds), map(hi,inds))
Fill(value, kernel::AbstractArray) = Fill(value, indices(kernel))
# Fill(value, factkernel::Tuple) = Fill(value, flatten(map(indices, factkernel)))  # FIXME

(p::Fill)(kernel::AbstractArray) = Fill(p.value, kernel)
(p::Fill)(factkernel::Tuple) = Fill(p.value, factkernel)

function padarray(img::AbstractArray, border::Fill)
    throw(ArgumentError("$border lacks the proper padding sizes for an array with $(ndims(img)) dimensions"))
end
function padarray{T,_,N}(img::AbstractArray{T,N}, f::Fill{_,N})
    A = similar(Array{T}, map((l,r,h)->first(r)-l:last(r)+h, f.lo, indices(img), f.hi))
    fill!(A, f.value)
    A[indices(img)...] = img
    A
end

# There are other ways to define these, but using `mod` makes it safe
# for cases where the padding is bigger than length(inds)
padindex(border::Pad{:replicate}, lo::Integer, inds::AbstractUnitRange, hi::Integer) =
    vcat(fill(first(inds), lo), PinIndices(inds), fill(last(inds), hi))
padindex(border::Pad{:circular}, lo::Integer, inds::AbstractUnitRange, hi::Integer) =
    modrange(extend(lo, inds, hi), inds)
function padindex(border::Pad{:symmetric}, lo::Integer, inds::AbstractUnitRange, hi::Integer)
    I = [inds; reverse(inds)]
    I[modrange(extend(lo, inds, hi), 1:2*length(inds))]
end
function padindex(border::Pad{:reflect}, lo::Integer, inds::AbstractUnitRange, hi::Integer)
    I = [inds; last(inds)-1:-1:first(inds)+1]
    I[modrange(extend(lo, inds, hi), 1:2*length(inds)-2)]
end
"""
    padindex(border::Pad, lo::Integer, inds::AbstractUnitRange, hi::Integer)

Generate an index-vector to be used for padding. `inds` specifies the image indices along a particular axis; `lo` and `hi` are the amount to pad on the lower and upper, respectively, sides of this axis. `border` specifying the style of padding.

You can specialize this for custom `Pad{:name}` types to generate
arbitrary (cartesian) padding types.
"""
padindex

function inner(lo::Integer, inds::AbstractUnitRange, hi::Integer)
    first(inds)+lo:last(inds)-hi
end

lo(o::Integer) = max(-o, zero(o))
lo(r::AbstractUnitRange) = lo(first(r))

hi(o::Integer) = max(o, zero(o))
hi(r::AbstractUnitRange) = hi(last(r))

# extend(lo::Integer, inds::AbstractUnitRange, hi::Integer) = CatIndices.URange(first(inds)-lo, last(inds)+hi)
function extend(lo::Integer, inds::AbstractUnitRange, hi::Integer)
    newind = first(inds)-lo:last(inds)+hi
    OffsetArray(newind, newind)
end

# @inline flatten(t::Tuple) = _flatten(t...)
# @inline _flatten(t1::Tuple, t...) = (flatten(t1)..., flatten(t)...)
# @inline _flatten(t1,        t...) = (t1, flatten(t)...)
# _flatten() = ()

extremize(inds::Indices, kernel1::AbstractArray, kernels...) = extremize(_extremize(inds, indices(kernel1)), kernels...)
extremize(inds) = inds
_extremize(inds1, inds2) = map(__extremize, inds1, inds2)
__extremize(ind1, ind2) = min(first(ind1),first(ind2)):max(last(ind1),last(ind2))

modrange(x, r::AbstractUnitRange) = mod(x-first(r), length(r))+first(r)
modrange(A::AbstractArray, r::AbstractUnitRange) = map(x->modrange(x, r), A)

# end
