# module Border

using OffsetArrays, CatIndices
using Base: Indices, tail

abstract AbstractBorder

# Style is a Symbol: :replicate, :circular, :symmetric, or :reflect
type Pad{Style,N} <: AbstractBorder
    lo::Dims{N}    # number to extend by on the lower edge for each dimension
    hi::Dims{N}    # number to extend by on the upper edge for each dimension
end
(::Type{Pad{Style}}){Style  }(lo::Tuple{}, hi::Tuple{}) = Pad{Style,0}(lo, hi)
(::Type{Pad{Style}}){Style,N}(lo::Dims{N}, hi::Dims{N}) = Pad{Style,N}(lo, hi)
(::Type{Pad{Style}}){Style,N}(lo::Dims{N}, hi::Tuple{}) = Pad{Style,N}(lo, ntuple(d->0,Val{N}))
(::Type{Pad{Style}}){Style,N}(lo::Tuple{}, hi::Dims{N}) = Pad{Style,N}(ntuple(d->0,Val{N}), hi)
(::Type{Pad{Style}}){Style,N}(both::Dims{N}) = Pad{Style,N}(both, both)
(::Type{Pad{Style}}){Style}(both::Int...) = Pad{Style}(both, both)
(::Type{Pad{Style}}){Style,N}(inds::Indices{N}) = Pad{Style,N}(map(lo,inds), map(hi,inds))
(::Type{Pad{Style}}){Style}(kernel::AbstractArray) = Pad{Style}(indices(kernel))
(::Type{Pad{Style}}){Style}(factkernel::Tuple) = Pad{Style}(flatten(map(indices, factkernel)))
(::Type{Pad})(args...) = Pad{:replicate}(args...)

function padindices(img::AbstractArray, border::Pad)
    throw(ArgumentError("$border lacks the proper padding sizes for an array with $(ndims(img)) dimensions"))
end
function padindices{_,Style,N}(img::AbstractArray{_,N}, border::Pad{Style,N})
    __padindices(border, border.lo, indices(img), border.hi)
end
function padindices{P<:Pad}(img::AbstractArray, ::Type{P})
    throw(ArgumentError("must supply padding sizes to $P"))
end

# The 3-argument map is not inferrable, so do it manually
@inline __padindices(border, lo, inds, hi) =
    (_padindices(border, lo[1], inds[1], hi[1]),
     __padindices(border, tail(lo), tail(inds), tail(hi))...)
__padindices(border, ::Tuple{}, ::Tuple{}, ::Tuple{}) = ()

padarray(img, border::Pad)  = img[padindices(img, border)...]
padarray{P}(img, ::Type{P}) = img[padindices(img, P)...]      # just to throw the nice error

# Make this a separate type because the dispatch almost surely needs to be different
immutable Inner{N} <: AbstractBorder
    lo::Dims{N}
    hi::Dims{N}
end

# Fill is a little different, so handle it separately
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
Fill(value, factkernel::Tuple) = Fill(value, flatten(map(indices, factkernel)))

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
_padindices(border::Pad{:replicate}, lo::Integer, inds::AbstractUnitRange, hi::Integer) =
    vcat(fill(first(inds), lo), PinIndices(inds), fill(last(inds), hi))
_padindices(border::Pad{:circular}, lo::Integer, inds::AbstractUnitRange, hi::Integer) =
    modrange(extend(lo, inds, hi), inds)
function _padindices(border::Pad{:symmetric}, lo::Integer, inds::AbstractUnitRange, hi::Integer)
    I = [inds; reverse(inds)]
    I[modrange(extend(lo, inds, hi), 1:2*length(inds))]
end
function _padindices(border::Pad{:reflect}, lo::Integer, inds::AbstractUnitRange, hi::Integer)
    I = [inds; last(inds)-1:-1:first(inds)+1]
    I[modrange(extend(lo, inds, hi), 1:2*length(inds)-2)]
end

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

@inline flatten(t::Tuple) = _flatten(t...)
@inline _flatten(t1::Tuple, t...) = (flatten(t1)..., flatten(t)...)
@inline _flatten(t1,        t...) = (t1, flatten(t)...)
_flatten() = ()

modrange(x, r::AbstractUnitRange) = mod(x-first(r), length(r))+first(r)
modrange(A::AbstractArray, r::AbstractUnitRange) = map(x->modrange(x, r), A)

# end
