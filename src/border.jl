# module Border

using OffsetArrays, CatIndices

@compat abstract type AbstractBorder end

immutable NoPad{T} <: AbstractBorder
    border::T
end
NoPad() = NoPad(nothing)

"""
    NoPad()
    NoPad(border)

Indicates that no padding should be applied to the input array, or that you have already pre-padded the input image. Passing a `border` object allows you to preserve "memory" of a border choice; it can be retrieved by indexing with `[]`.

# Example

    np = NoPad(Pad(:replicate))
    imfilter!(out, img, kernel, np)

runs filtering directly, skipping any padding steps.  Every entry of
`out` must be computable using in-bounds operations on `img` and
`kernel`.
"""
NoPad

Base.getindex(np::NoPad) = np.border

"""
`Pad` is a type that stores choices about padding. Instances must set `style`, a
Symbol specifying the boundary conditions of the image, one of:

- `:replicate` (repeat edge values to infinity)
- `:circular` (image edges "wrap around")
- `:symmetric` (the image reflects relative to a position between pixels)
- `:reflect` (the image reflects relative to the edge itself)

The default value is `:replicate`.

It's worth emphasizing that padding is most straightforwardly specified as a string,

    imfilter(img, kernel, "replicate")

rather than

    imfilter(img, kernel, Pad(:replicate))
"""
immutable Pad{N} <: AbstractBorder
    style::Symbol
    lo::Dims{N}    # number to extend by on the lower edge for each dimension
    hi::Dims{N}    # number to extend by on the upper edge for each dimension
end

const valid_borders = ("replicate", "circular", "reflect", "symmetric")

function borderinstance(border::AbstractString)
    if border ∈ valid_borders
        return Pad(Symbol(border))
    elseif border == "inner"
        throw(ArgumentError("specifying Inner as a string is deprecated, use `imfilter(img, kern, Inner())` instead"))
    else
        throw(ArgumentError("$border not a recognized border"))
    end
end
borderinstance(b::AbstractBorder) = b

"""
    Pad(style::Symbol, m, n, ...)

Pad the input image symmetrically, `m` pixels at the lower and upper edge of dimension 1, `n` pixels for dimension 2, and so forth.
"""
Pad(style::Symbol, both::Int...) = Pad(style, both, both)
Pad(both::Int...) = Pad(:replicate, both, both)

"""
    Pad(style::Symbol, (m,n))

Pad the input image symmetrically, `m` pixels at the lower and upper edge of dimension 1, `n` pixels for dimension 2.
"""
Pad(style::Symbol, both::Dims) = Pad(style, both, both)
Pad(both::Dims) = Pad(:replicate, both, both)

"""
    Pad(style::Symbol, lo::Dims, hi::Dims)

Pad the input image by `lo` pixels at the lower edge, and `hi` pixels at the upper edge.
"""
Pad(lo::Dims, hi::Dims) = Pad(:replicate, lo, hi)
Pad(style::Symbol, lo::Tuple{}, hi::Tuple{}) = Pad{0}(style, lo, hi)
Pad{N}(style::Symbol, lo::Dims{N}, hi::Tuple{}) = Pad(style, lo, ntuple(d->0,Val{N}))
Pad{N}(style::Symbol, lo::Tuple{}, hi::Dims{N}) = Pad(style, ntuple(d->0,Val{N}), hi)
Pad(style::Symbol, lo::AbstractVector{Int}, hi::AbstractVector{Int}) = Pad(style, (lo...,), (hi...,))

Pad(style::Symbol, inds::Indices) = Pad(style, map(lo,inds), map(hi,inds))

"""
    Pad(style, kernel)
    Pad(style)(kernel)

Given a filter array `kernel`, determine the amount of padding from the `indices` of `kernel`.
"""
(p::Pad{0})(kernel) = Pad(p.style, calculate_padding(kernel))
(p::Pad{0})(kernel, img, ::Alg) = p(kernel)

# Padding for FFT: round up to next size expressible as 2^m*3^n
function (p::Pad{0})(kernel, img, ::FFT)
    inds = calculate_padding(kernel)
    newinds = map(padfft, inds, map(length, indices(img)))
    Pad(p.style, newinds)
end
function padfft(indk::AbstractUnitRange, l::Integer)
    lk = length(indk)
    range(first(indk), nextprod([2,3], l+lk)-l+1)
end

function padindices{_,N}(img::AbstractArray{_,N}, border::Pad)
    throw(ArgumentError("$border lacks the proper padding sizes for an array with $(ndims(img)) dimensions"))
end
function padindices{_,N}(img::AbstractArray{_,N}, border::Pad{N})
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
add. `border` can be a `Pad`, `Fill`, or `Inner` object.

Optionally provide the element type `T` of `imgpadded`.
"""
padarray(img::AbstractArray, border::Pad)  = padarray(eltype(img), img, border)
function padarray{T}(::Type{T}, img::AbstractArray, border::Pad)
    inds = padindices(img, border)
    # like img[inds...] except that we can control the element type
    newinds = map(Base.indices1, inds)
    dest = similar(img, T, newinds)
    copydata!(dest, img, inds)
end

padarray{P}(img, ::Type{P}) = img[padindices(img, P)...]      # just to throw the nice error

function copydata!(dest, img, inds)
    isempty(inds) && return dest
    idest = indices(dest)
    # Work around julia #9080
    i1, itail = idest[1], tail(idest)
    inds1, indstail = inds[1], tail(inds)
    @unsafe for I in CartesianRange(itail)
        J = CartesianIndex(map((i,x)->x[i], I.I, indstail))
        for i in i1
            j = inds1[i]
            dest[i,I] = img[j,J]
        end
    end
    dest
end

function copydata!(dest::OffsetArray, img, inds::Tuple{Vararg{OffsetArray}})
    copydata!(parent(dest), img, map(parent, inds))
    dest
end

Base.ndims{N}(::Pad{N}) = N

# Make these separate types because the dispatch almost surely needs to be different
"""
    Inner()
    Inner(lo, hi)

Indicate that edges are to be discarded in filtering, only the interior of the result it to be returned.

# Example:

    imfilter(img, kernel, Inner())
"""
immutable Inner{N} <: AbstractBorder
    lo::Dims{N}
    hi::Dims{N}
end

"""
    NA()
    NA(lo, hi)

Choose filtering using "NA" (Not Available) boundary conditions. This
is most appropriate for filters that have only positive weights, such
as blurring filters. Effectively, the output pixel value is normalized
in the following way:

              filtered img with Fill(0) boundary conditions
    output =  ---------------------------------------------
              filtered 1   with Fill(0) boundary conditions

As a consequence, filtering has the same behavior as
`nanmean`. Indeed, invalid pixels in `img` can be marked as `NaN` and
then they are effectively omitted from the filtered result.
"""
immutable NA{N} <: AbstractBorder
    lo::Dims{N}
    hi::Dims{N}
end

for T in (:Inner, :NA)
    @eval begin
        (::Type{$T})(both::Int...) = $T(both, both)
        (::Type{$T}){N}(both::Dims{N}) = $T(both, both)
        (::Type{$T})(lo::Tuple{}, hi::Tuple{}) = $T{0}(lo, hi)
        (::Type{$T}){N}(lo::Dims{N}, hi::Tuple{}) = $T{N}(lo, ntuple(d->0,Val{N}))
        (::Type{$T}){N}(lo::Tuple{}, hi::Dims{N}) = $T{N}(ntuple(d->0,Val{N}), hi)
        (::Type{$T}){N}(inds::Indices{N}) = $T{N}(map(lo,inds), map(hi,inds))
        (::Type{$T{N}}){N}(lo::AbstractVector, hi::AbstractVector) = $T{N}((lo...,), (hi...,))
        (::Type{$T})(lo::AbstractVector, hi::AbstractVector) = $T((lo...,), (hi...,)) # not inferrable

        (p::$T{0})(kernel, img, ::Alg) = p(kernel)
        (p::$T{0})(kernel) = $T(calculate_padding(kernel))
    end
end

padarray(img, border::Inner) = padarray(eltype(img), img, border)
padarray{T}(::Type{T}, img::AbstractArray{T}, border::Inner) = copy(img)
padarray{T}(::Type{T}, img::AbstractArray, border::Inner) = copy!(similar(Array{T}, indices(img)), img)

"""
    Fill(val)
    Fill(val, lo, hi)

Pad the edges of the image with a constant value, `val`.

Optionally supply the extent of the padding, see `Pad`.

# Example:

    imfilter(img, kernel, Fill(zero(eltype(img))))
"""
immutable Fill{T,N} <: AbstractBorder
    value::T
    lo::Dims{N}
    hi::Dims{N}

    (::Type{Fill{T,N}}){T,N}(value::T) = new{T,N}(value)
    (::Type{Fill{T,N}}){T,N}(value::T, lo::Dims{N}, hi::Dims{N}) = new{T,N}(value, lo, hi)
end

Fill{T}(value::T) = Fill{T,0}(value)
Fill{T,N}(value::T, lo::Dims{N}, hi::Dims{N}) = Fill{T,N}(value, lo, hi)
Fill{T,N}(value::T, both::Dims{N}) = Fill{T,N}(value, both, both)
Fill(value, lo::AbstractVector, hi::AbstractVector) = Fill(value, (lo...,), (hi...,))
Fill{T,N}(value::T, inds::Base.Indices{N}) = Fill{T,N}(value, map(lo,inds), map(hi,inds))
Fill(value, kernel) = Fill(value, calculate_padding(kernel))

(p::Fill)(kernel) = Fill(p.value, kernel)
(p::Fill)(kernel, img, ::Alg) = Fill(p.value, kernel)
function (p::Fill)(kernel, img, ::FFT)
    inds = calculate_padding(kernel)
    newinds = map(padfft, inds, map(length, indices(img)))
    Fill(p.value, newinds)
end

function padarray{T}(::Type{T}, img::AbstractArray, border::Fill)
    throw(ArgumentError("$border lacks the proper padding sizes for an array with $(ndims(img)) dimensions"))
end
function padarray{T,S,_,N}(::Type{T}, img::AbstractArray{S,N}, f::Fill{_,N})
    A = similar(arraytype(img, T), map((l,r,h)->first(r)-l:last(r)+h, f.lo, indices(img), f.hi))
    try
        fill!(A, f.value)
    catch
        error("Unable to fill! an array of element type $(eltype(A)) with the value $(f.value). Supply an appropriate value to `Fill`, such as `zero(eltype(A))`.")
    end
    A[indices(img)...] = img
    A
end
padarray(img::AbstractArray, f::Fill) = padarray(eltype(img), img, f)

# There are other ways to define these, but using `mod` makes it safe
# for cases where the padding is bigger than length(inds)
"""
    padindex(border::Pad, lo::Integer, inds::AbstractUnitRange, hi::Integer)

Generate an index-vector to be used for padding. `inds` specifies the image indices along a particular axis; `lo` and `hi` are the amount to pad on the lower and upper, respectively, sides of this axis. `border` specifying the style of padding.
"""
function padindex(border::Pad, lo::Integer, inds::AbstractUnitRange, hi::Integer)
    if border.style == :replicate
        return vcat(fill(first(inds), lo), PinIndices(inds), fill(last(inds), hi))
    elseif border.style == :circular
        return modrange(extend(lo, inds, hi), inds)
    elseif border.style == :symmetric
        I = [inds; reverse(inds)]
        r = modrange(extend(lo, inds, hi), 1:2*length(inds))
        return I[r]
    elseif border.style == :reflect
        I = [inds; last(inds)-1:-1:first(inds)+1]
        return I[modrange(extend(lo, inds, hi), 1:2*length(inds)-2)]
    else
        error("border style $(border.style) unrecognized")
    end
end
function padindex(border::Pad, inner::AbstractUnitRange, outer::AbstractUnitRange)
    lo = max(0, first(inner)-first(outer))
    hi = max(0, last(outer)-last(inner))
    padindex(border, lo, inner, hi)
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

calculate_padding(kernel) = indices(kernel)
@inline function calculate_padding(kernel::Tuple{Any, Vararg{Any}})
    inds = accumulate_padding(indices(kernel[1]), tail(kernel)...)
    if hasiir(kernel) && hasfir(kernel)
        inds = map(doublepadding, inds)
    end
    inds
end

hasiir(kernel) = _hasiir(false, kernel...)
_hasiir(ret) = ret
_hasiir(ret, kern, kernel...) = _hasiir(ret, kernel...)
_hasiir(ret, kern::AnyIIR, kernel...) = true

hasfir(kernel) = _hasfir(false, kernel...)
_hasfir(ret) = ret
_hasfir(ret, kern, kernel...) = true
_hasfir(ret, kern::AnyIIR, kernel...) = _hasfir(ret, kernel...)

function doublepadding(ind::AbstractUnitRange)
    f, l = first(ind), last(ind)
    f = f < 0 ? 2f : f
    l = l > 0 ? 2l : l
    f:l
end

accumulate_padding(inds::Indices, kernel1, kernels...) =
    accumulate_padding(expand(inds, indices(kernel1)), kernels...)
accumulate_padding(inds::Indices) = inds

modrange(x, r::AbstractUnitRange) = mod(x-first(r), length(r))+first(r)
modrange(A::AbstractArray, r::AbstractUnitRange) = map(x->modrange(x, r), A)

arraytype{T}(A::AbstractArray, ::Type{T}) = Array{T}  # fallback
arraytype(A::BitArray, ::Type{Bool}) = BitArray

interior(A, kernel) = _interior(indices(A), indices(kernel))
interior(A, factkernel::Tuple) = _interior(indices(A), accumulate_padding(indices(factkernel[1]), tail(factkernel)...))
function _interior{N}(indsA::NTuple{N}, indsk)
    indskN = fill_to_length(indsk, 0:0, Val{N})
    map(intersect, indsA, shrink(indsA, indsk))
end

next_shrink(inds::Indices, ::Tuple{}) = inds
function next_shrink(inds::Indices, kernel::Tuple)
    kern = first(kernel)
    iscopy(kern) && return next_shrink(inds, tail(kernel))
    shrink(inds, kern)
end

"""
    expand(inds::Indices, kernel)
    expand(inds::Indices, indskernel::Indices)

Expand an image region `inds` to account for necessary padding by `kernel`.
"""
expand(inds::Indices, kernel) = expand(inds, calculate_padding(kernel))
expand(inds::Indices, pad::Indices) = firsttype(map_copytail(expand, inds, pad))
expand(ind::AbstractUnitRange, pad::AbstractUnitRange) = oftype(ind, first(ind)+first(pad):last(ind)+last(pad))
expand(ind::Base.OneTo, pad::AbstractUnitRange) = expand(UnitRange(ind), pad)

"""
    shrink(inds::Indices, kernel)
    shrink(inds::Indices, indskernel)

Remove edges from an image region `inds` that correspond to padding needed for `kernel`.
"""
shrink(inds::Indices, kernel) = shrink(inds, calculate_padding(kernel))
shrink(inds::Indices, pad::Indices) = firsttype(map_copytail(shrink, inds, pad))
shrink(ind::AbstractUnitRange, pad::AbstractUnitRange) = oftype(ind, first(ind)-first(pad):last(ind)-last(pad))
shrink(ind::Base.OneTo, pad::AbstractUnitRange) = shrink(UnitRange(ind), pad)

allocate_output{T}(::Type{T}, img, kernel, border) = similar(img, T)
function allocate_output{T}(::Type{T}, img, kernel, ::Inner{0})
    inds = interior(img, kernel)
    similar(img, T, inds)
end
allocate_output(img, kernel, border) = allocate_output(filter_type(img, kernel), img, kernel, border)

"""
    map_copytail(f, a::Tuple, b::Tuple)

Apply `f` to paired elements of `a` and `b`, copying any tail elements
when the lengths of `a` and `b` are not equal.
"""
@inline map_copytail(f, a::Tuple, b::Tuple) = (f(a[1], b[1]), map_copytail(f, tail(a), tail(b))...)
map_copytail(f, a::Tuple{}, b::Tuple{}) = ()
map_copytail(f, a::Tuple, b::Tuple{}) = a
map_copytail(f, a::Tuple{}, b::Tuple) = b

function firsttype(t::Tuple)
    T = typeof(t[1])
    map(T, t)
end
firsttype(::Tuple{}) = ()

# end
