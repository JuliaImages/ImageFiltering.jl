# module Border

using OffsetArrays, CatIndices

abstract type AbstractBorder end

struct NoPad{T} <: AbstractBorder
    border::T
end
NoPad() = NoPad(nothing)

"""
    NoPad()
    NoPad(border)

Indicates that no padding should be applied to the input array, or that you have already pre-padded the input image. Passing a `border` object allows you to preserve "memory" of a border choice; it can be retrieved by indexing with `[]`.

# Example
The commands

    np = NoPad(Pad(:replicate))
    imfilter!(out, img, kernel, np)

run filtering directly, skipping any padding steps.  Every entry of
`out` must be computable using in-bounds operations on `img` and
`kernel`.
"""
NoPad

Base.getindex(np::NoPad) = np.border

"""
```julia
    struct Pad{N} <: AbstractBorder
        style::Symbol
        lo::Dims{N}    # number to extend by on the lower edge for each dimension
        hi::Dims{N}    # number to extend by on the upper edge for each dimension
    end
```

`Pad` is a type that designates the form of padding which should be used to
extrapolate pixels beyond the boundary of an image. Instances must set `style`,
a Symbol specifying the boundary conditions of the image.

# Output

The type `Pad` specifying how the boundary of an image should be padded.

# Extended help

When representing a spatial two-dimensional image filtering operation as a
discrete convolution between the image and a ``D \\times D `` filter, the
results are undefined for pixels closer than ``D`` pixels from the border of the
image. To define the operation near and at the border, one needs a scheme for
extrapolating pixels beyond the edge. The `Pad` type allows one to specify the
necessary extrapolation scheme.

The type facilitates the padding of one, two or multi-dimensional images.

You can specify a different amount of padding at the lower and upper borders of
each dimension of the image (top, left, bottom and right in two dimensions).

## Options

Some valid `style` options are described below. As an indicative example of each
option the results of the padding are illustrated on an image consisting of a
row of six pixels which are specified alphabetically:

```plain
        ┏━━━━━━┓ 
        ┃abcdef┃ 
        ┗━━━━━━┛ 
```

We show the effects of padding only on the left and right
border, but analogous consequences hold for the top and bottom border.

### `:replicate` (Default)

The border pixels extend beyond the image boundaries.

```plain
   ╭────┏━━━━━━┓────╮
   │aaaa┃abcdef┃ffff│
   ╰────┗━━━━━━┛────╯
```

See also: [`Fill`](@ref), [`padarray`](@ref), [`Inner`](@ref) and
[`NoPad`](@ref)

### `:circular`

The border pixels wrap around. For instance, indexing beyond the left border
returns values starting from the right border.

```plain

   ╭────┏━━━━━━┓────╮
   │cdef┃abcdef┃abcd│
   ╰────┗━━━━━━┛────╯

```

See also: [`Fill`](@ref), [`padarray`](@ref), [`Inner`](@ref) and
[`NoPad`](@ref)

### `:symmetric`

The border pixels reflect relative to a position between pixels. That is, the
border pixel is omitted when mirroring.

```plain

   ╭────┏━━━━━━┓────╮
   │edcb┃abcdef┃edcb│
   ╰────┗━━━━━━┛────╯

```

### `:reflect`

The border pixels reflect relative to the edge itself.

```plain

   ╭────┏━━━━━━┓────╮
   │dcba┃abcdef┃fedc│
   ╰────┗━━━━━━┛────╯

```

See also: [`Fill`](@ref),[`padarray`](@ref), [`Inner`](@ref) and
[`NoPad`](@ref).

"""
struct Pad{N} <: AbstractBorder
    style::Symbol
    lo::Dims{N}    # number to extend by on the lower edge for each dimension
    hi::Dims{N}    # number to extend by on the upper edge for each dimension
end

Pad{N}(style, lo::AbstractVector, hi::AbstractVector) where {N} =
    Pad{N}(style, (lo...,), (hi...,))

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
```julia
    Pad(style::Symbol, m, n, ...)
    Pad(style::Symbol, (m,n))
```

Construct an instance of [`Pad`](@ref) such that the image is prepended and appended symmetrically with `m` pixels at the lower and upper edge of dimension 1, `n` pixels for dimension 2, and so forth.

#### Usage illustration
Use `Pad(:replicate,2,4)` to designate that the top and bottom border should be
replicated by two pixels, and the left and right border by four pixels.

Use `Pad(:circular,(0,3))` to designate that the top and bottom border should
not be padded, and that the left and right border should wrap around by three
pixels.

---

"""
Pad(style::Symbol, both::Int...) = Pad(style, both, both)
Pad(both::Int...) = Pad(:replicate, both, both)
Pad(style::Symbol, both::Dims) = Pad(style, both, both)

"""
```julia
    Pad(both::Dims)
```
Construct an instance of [`Pad`](@ref) with default `:replicate` extrapolation, where the tuple `both` specifies the number of pixels which will be prepended and appended for each dimension.

#### Usage illustration
Use `Pad((5,5))` to designate that the top, bottom, left and right border should
be replicated by five pixels.

---

"""
Pad(both::Dims) = Pad(:replicate, both, both)

"""
```julia
    Pad(style::Symbol, lo::Dims, hi::Dims)
```
Construct an instance of [`Pad`](@ref) such that the image is prepended by `lo` pixels  and appended by `hi` pixels  in each dimension.

#### Usage illustration

Use `Pad(:replicate,(1,2),(3,4))` to designate that the top and bottom border
should be replicated by one and two pixels, and that the left and right border
should be replicated by three and four pixels.

---

"""
Pad(lo::Dims, hi::Dims) = Pad(:replicate, lo, hi)
Pad(style::Symbol, lo::Tuple{}, hi::Tuple{}) = Pad{0}(style, lo, hi)
Pad(style::Symbol, lo::Dims{N}, hi::Tuple{}) where {N} = Pad(style, lo, ntuple(d -> 0, Val(N)))
Pad(style::Symbol, lo::Tuple{}, hi::Dims{N}) where {N} = Pad(style, ntuple(d -> 0, Val(N)), hi)
Pad(style::Symbol, lo::AbstractVector{Int}, hi::AbstractVector{Int}) = Pad(style, (lo...,), (hi...,))

Pad(style::Symbol, ::Tuple{}) = Pad(style, (), ())    # ambiguity resolution
Pad(style::Symbol, inds::Indices) = Pad(style, map(lo, inds), map(hi, inds))

"""
```julia
    Pad(style, kernel)
    Pad(style)(kernel)
```
Construct an instance of [`Pad`](@ref) by designating the value `val` and a filter array `kernel` which will be used to determine the amount of padding from the `axes` of `kernel`.

#### Usage illustration

Use `Pad(:circular,Kernel.sobel())` to specify a `:circular` border style and
the minimal amount of padding necessary to ensure that convolution with
[`Kernel.sobel`](@ref) will be defined at the borders of an image.

---

"""
(p::Pad{0})(kernel) = Pad(p.style, calculate_padding(kernel))
(p::Pad{0})(kernel, img, ::Alg) = p(kernel)

# Padding for FFT: round up to next size expressible as 2^m*3^n
function (p::Pad{0})(kernel, img, ::FFT)
    inds = calculate_padding(kernel)
    newinds = map(padfft, inds, map(length, axes(img)))
    Pad(p.style, newinds)
end
function padfft(indk::AbstractUnitRange, l::Integer)
    lk = length(indk)
    range(first(indk), length=nextprod([2, 3], l + lk) - l + 1)
end

function error_bad_padding_size(inner, border)
    ArgumentError("$border lacks the proper padding sizes for an array with $(ndims(inner)) dimensions")
end

function padindices(img::AbstractArray{<:Any,N}, border::Pad) where {N}
    throw(error_bad_padding_size(img, border))
end
function padindices(img::AbstractArray{<:Any,N}, border::Pad{N}) where {N}
    _padindices(border, border.lo, axes(img), border.hi)
end
function padindices(img::AbstractArray, ::Type{P}) where {P<:Pad}
    throw(ArgumentError("must supply padding sizes to $P"))
end

# The 3-argument map is not inferrable, so do it manually
@inline _padindices(border, lo, inds, hi) =
    (padindex(border, lo[1], inds[1], hi[1]),
        _padindices(border, tail(lo), tail(inds), tail(hi))...)
_padindices(border, ::Tuple{}, ::Tuple{}, ::Tuple{}) = ()

"""
    padarray([T], img, border)
        
Generate a padded image from an array `img` and a specification
`border` of the boundary conditions and amount of padding to add.
    
Return a padded image. The function supports one, two or multi-dimensional images. You can specify the
element type `T` of the output image.

See [`Pad`](@ref) and [`Fill`](@ref) for details.

# Examples

## Padding

The main syntax for `Pad` is `(style, m, n, ...)` or `(style, (m, n))`, where `m`
pixels are added to dimension 1 (top and bottom), `n` pixels for dimension 2,
and so forth.

Add 30 to left and right, 40 to top and bottom:

```julia
padarray(A, Pad(:replicate, 30, 40))
padarray(A, Pad(:circular, 30, 40))
padarray(A, Pad(:symmetric, 30, 40))
padarray(A, Pad(:reflect, 30, 40))
```

Add 30 above, 40 to left, 50 to bottom, 60 to right:

```julia
padarray(A, Pad(0, (30, 40), (50, 60)))
padarray(A, Pad(0, (30, 40), (50, 60)))
```

# 3D

```julia
padarray(A, Pad(:replicate, 1, 1, 1)) 
padarray(A, Fill(0, (1, 1, 1))) 
```

## Filling

The main syntax for `Fill` is `(value, m, n)` or `(value, (m, n))` where the image is
prepended by `m` pixels and appended by `n` pixels in each dimension.

Add 20 `-1` values above, 30 to left, 40 to bottom, 50 to right:

```julia
padarray(A, Fill(-1, (20, 30), (40, 50))) 
```

"""
padarray(img::AbstractArray, border::AbstractBorder) = padarray(eltype(img), img, border)
function padarray(::Type{T}, img::AbstractArray, border) where {T}
    ba = BorderArray(img, border)
    out = similar(ba, T, axes(ba))
    copy!(out, ba)
end

padarray(img, ::Type{P}) where {P} = img[padindices(img, P)...]      # just to throw the nice error

function copydata!(dest, img, inds)
    isempty(inds) && return dest
    idest = axes(dest)
    # Work around julia #9080
    i1, itail = idest[1], tail(idest)
    inds1, indstail = inds[1], tail(inds)
    @inbounds for I in CartesianIndices(itail)
        J = CartesianIndex(map((i, x) -> x[i], Tuple(I), indstail))
        for i in i1
            j = inds1[i]
            dest[i, I] = img[j, J]
        end
    end
    dest
end

function copydata!(dest::OffsetArray, img, inds::Tuple{Vararg{OffsetArray}})
    copydata!(parent(dest), img, map(parent, inds))
    dest
end

Base.ndims(b::AbstractBorder) = ndims(typeof(b))
Base.ndims(::Type{Pad{N}}) where {N} = N

# Make these separate types because the dispatch almost surely needs to be different
"""
    Inner()
    Inner(lo, hi)

Indicate that edges are to be discarded in filtering, only the interior of the result is to be returned.

# Example:

    imfilter(img, kernel, Inner())
"""
struct Inner{N} <: AbstractBorder
    lo::Dims{N}
    hi::Dims{N}
end

Inner(both::Int...) = Inner(both, both)
Inner(::Tuple{}) = Inner((), ())   # ambiguity resolution
Inner(both::Dims{N}) where {N} = Inner(both, both)
Inner(lo::Tuple{}, hi::Tuple{}) = Inner{0}(lo, hi)
Inner(lo::Dims{N}, hi::Tuple{}) where {N} = Inner{N}(lo, ntuple(d -> 0, Val(N)))
Inner(lo::Tuple{}, hi::Dims{N}) where {N} = Inner{N}(ntuple(d -> 0, Val(N)), hi)
Inner(inds::Indices{N}) where {N} = Inner{N}(map(lo, inds), map(hi, inds))
Inner{N}(lo::AbstractVector, hi::AbstractVector) where {N} = Inner{N}((lo...,), (hi...,))
Inner(lo::AbstractVector, hi::AbstractVector) = Inner((lo...,), (hi...,)) # not inferrable

(p::Inner{0})(kernel, img, ::Alg) = p(kernel)
(p::Inner{0})(kernel) = Inner(calculate_padding(kernel))

Base.ndims(::Type{Inner{N}}) where {N} = N

"""
    NA(na=isnan)

Choose filtering using "NA" (Not Available) boundary conditions. This
is most appropriate for filters that have only positive weights, such
as blurring filters. Effectively, the output value is normalized in the
following way:

              filtered array with Fill(0) boundary conditions
    output =  -----------------------------------------------
              filtered 1     with Fill(0) boundary conditions

Array elements for which `na` returns `true` are also considered outside
array boundaries.
"""
struct NA{na} <: AbstractBorder
    NA(na=isnan) = new{na}()
end

"""
```julia
struct Fill{T,N} <: AbstractBorder
    value::T
    lo::Dims{N}
    hi::Dims{N}
end
```

`Fill` is a type that designates a particular value which will be used to
extrapolate pixels beyond the boundary of an image.

# Output

The type `Fill` specifying the value with which the boundary of the image should
be padded.

# Details

When representing a two-dimensional spatial image filtering operation as a
discrete convolution between an image and a ``D \\times D `` filter, the
results are undefined for pixels closer than ``D`` pixels from the border of the
image. To define the operation near and at the border, you need a scheme for
extrapolating pixels beyond the edge. The `Fill` type allows you to specify a
particular value which will be used in the extrapolation. For more elaborate
extrapolation schemes, see [`Pad`](@ref).

The type facilitates the padding of one, two, or multi-dimensional images.

You can specify a different amount of padding at the lower and upper borders of
each dimension of the image (top, left, bottom, and right in two dimensions).

# Example

To illustrate this, consider an image consisting of a
row of six pixels which are specified alphabetically:

```plain
    ┏━━━━━━┓ 
    ┃abcdef┃ 
    ┗━━━━━━┛ 
```

Padding with a constant value `m` only on the left and
right border looks like this:

```plain
╭────┏━━━━━━┓────╮
│mmmm┃abcdef┃mmmm│
╰────┗━━━━━━┛────╯
```

(Analogous consequences hold for the top and bottom border.)

See also: [`Pad`](@ref), [`padarray`](@ref), [`Inner`](@ref) and
[`NoPad`](@ref)

"""
struct Fill{T,N} <: AbstractBorder
    value::T
    lo::Dims{N}
    hi::Dims{N}

    Fill{T,N}(value::T) where {T,N} = new{T,N}(value)
    Fill{T,N}(value::T, lo::Dims{N}, hi::Dims{N}) where {T,N} = new{T,N}(value, lo, hi)
end
Base.ndims(::Type{Fill{T,N}}) where {T,N} = N

"""
```julia
Fill(value::T)
```

Construct an instance of [`Fill`](@ref) designating a `value` and zero padding (i.e. no padding).

---

"""
Fill(value::T) where {T} = Fill{T,0}(value)

"""
```julia
Fill(value::T, lo::Dims{N}, hi::Dims{N})
Fill(value, lo::AbstractVector, hi::AbstractVector)
```

Construct an instance of [`Fill`](@ref) designating a `value` such that the image is prepended by `lo` pixels  and appended by `hi` pixels  in each dimension.

#### Usage illustration

Use `Fill(5, (2, 2), (2, 2))` to specify a padding of two pixels for the top,
bottom, left, and right edge with the value 5.


Use `Fill(zero(eltype(img))(1, 2), (3, 4))` to specify a padding of one, two, three,
and four pixels for the top, left, bottom, and right edge respectively using a
value of zero with the same type as `img`.

Use `Fill(0, [1, 2], [3, 4]` to specify a padding of one, two, three, and four pixels
for the top, left, bottom, and right edge respectively with the value 0.

"""
Fill(value::T, lo::Dims{N}, hi::Dims{N}) where {T,N} = Fill{T,N}(value, lo, hi)

"""
```julia
    Fill(value::T, both::Dims{N})
```
Construct an instance of [`Fill`](@ref) designating a `value`  and a tuple
`both` which stipulates the number of row and columns which will be prepended
and appended to the image.

#### Usage illustration

Use `Fill(0, (5, 10))` to stipulate a padding of five pixels for the top and left
edge, and a padding of ten pixels for the bottom and right edge with a value of
zero.
"""
Fill(value::T, both::Dims{N}) where {T,N} = Fill{T,N}(value, both, both)

Fill(value, lo::AbstractVector, hi::AbstractVector) = Fill(value, (lo...,), (hi...,))
Fill(value::T, ::Tuple{}) where {T} = Fill{T,0}(value, (), ())   # ambiguity resolution
Fill(value::T, inds::Base.Indices{N}) where {T,N} = Fill{T,N}(value, map(lo, inds), map(hi, inds))

"""
```julia
    Fill(value, kernel)
```

Construct an instance of [`Fill`](@ref) by designating a `value` and a
`kernel` which will be used to infer an appropriate padding.

A minimal amount of padding is added which ensures that a convolution between
the image and the kernel is defined at the boundary.

#### Usage illustration
Use `Fill(0,Kernel.sobel())` to specify a value of zero and the minimal amount
of padding necessary to ensure that convolution with [`Kernel.sobel`](@ref) will
be defined at the borders of an image.
"""
Fill(value, kernel) = Fill(value, calculate_padding(kernel))

(p::Fill)(kernel) = Fill(p.value, kernel)
(p::Fill)(kernel, img, ::Alg) = Fill(p.value, kernel)
function (p::Fill)(kernel, img, ::FFT)
    inds = calculate_padding(kernel)
    newinds = map(padfft, inds, map(length, axes(img)))
    Fill(p.value, newinds)
end

# There are other ways to define these, but using `mod` makes it safe
# for cases where the padding is bigger than length(inds)
"""
    padindex(border::Pad, lo::Integer, inds::AbstractUnitRange, hi::Integer)

Generate an index-vector to be used for padding. `inds` specifies the image axes along a particular axis; `lo` and `hi` are the amount to pad on the lower and upper, respectively, sides of this axis. `border` specifying the style of padding.
"""
function padindex(border::Pad, lo::Int, inds::UnitRange{Int}, hi::Int)
    if border.style == :replicate
        indsnew = OffsetArray{Int}(undef, first(inds)-lo:last(inds)+hi)
        offview = OffsetArrays.no_offset_view(indsnew)
        offview[1:lo] .= first(inds)
        offview[lo .+ eachindex(inds)] .= inds
        offview[lo + length(inds) + 1:end] .= last(inds)
        return indsnew
    elseif border.style == :circular
        return modrange(extend(lo, inds, hi), inds)
    elseif border.style == :symmetric
        I = OffsetArray{Int}(undef, (0:2*length(inds)-1) .+ first(inds))
        offview = OffsetArrays.no_offset_view(I)
        offview[eachindex(inds)] .= inds
        offview[end:-1:length(inds) + 1] .= inds
        r = modrange(extend(lo, inds, hi), axes(I, 1))
        return I[r]
    elseif border.style == :reflect
        I = OffsetArray{Int}(undef, (0:2*length(inds)-3) .+ first(inds))
        offview = OffsetArrays.no_offset_view(I)
        offview[eachindex(inds)] .= inds
        offview[length(inds) + 1:end] .= last(inds)-1:-1:first(inds)+1
        return I[modrange(extend(lo, inds, hi), axes(I, 1))]
    else
        error("border style $(border.style) unrecognized")
    end
end
padindex(border::Pad, lo::Integer, inds::AbstractUnitRange, hi::Integer) = padindex(border, Int(lo)::Int, UnitRange{Int}(inds)::UnitRange{Int}, Int(hi)::Int)
function padindex(border::Pad, inner::AbstractUnitRange, outer::AbstractUnitRange)
    lo = max(0, first(inner) - first(outer))
    hi = max(0, last(outer) - last(inner))
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

calculate_padding(kernel) = axes(kernel)
@inline function calculate_padding(kernel::Tuple{Any,Vararg{Any}})
    inds = accumulate_padding(axes(kernel[1]), tail(kernel)...)
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
    accumulate_padding(expand(inds, axes(kernel1)), kernels...)
accumulate_padding(inds::Indices) = inds

modrange(x, r::AbstractUnitRange) = mod(x - first(r), length(r)) + first(r)
modrange(A::AbstractArray, r::AbstractUnitRange) = map(x -> modrange(x, r), A)

arraytype(A::AbstractArray, ::Type{T}) where {T} = Array{T}  # fallback
arraytype(A::BitArray, ::Type{Bool}) = BitArray

interior(A, kernel) = _interior(axes(A), axes(kernel))
interior(A, factkernel::Tuple) = _interior(axes(A), accumulate_padding(axes(factkernel[1]), tail(factkernel)...))
function _interior(indsA::NTuple{N}, indsk) where {N}
    indskN = fill_to_length(indsk, 0:0, Val(N))
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
expand(ind::AbstractUnitRange, pad::AbstractUnitRange) = typeof(ind)(first(ind)+first(pad):last(ind)+last(pad))
expand(ind::Base.OneTo, pad::AbstractUnitRange) = expand(UnitRange(ind), pad)

"""
    shrink(inds::Indices, kernel)
    shrink(inds::Indices, indskernel)

Remove edges from an image region `inds` that correspond to padding needed for `kernel`.
"""
shrink(inds::Indices, kernel) = shrink(inds, calculate_padding(kernel))
shrink(inds::Indices, pad::Indices) = firsttype(map_copytail(shrink, inds, pad))
shrink(ind::AbstractUnitRange, pad::AbstractUnitRange) = typeof(ind)(first(ind)-first(pad):last(ind)-last(pad))
shrink(ind::Base.OneTo, pad::AbstractUnitRange) = shrink(UnitRange(ind), pad)

allocate_output(::Type{T}, img, kernel, border) where {T} = similar(img, T)
function allocate_output(::Type{T}, img, kernel, ::Inner{0}) where {T}
    inds = interior(img, kernel)
    similar(img, T, inds)
end
function allocate_output(::Type{T}, img, kernel, inr::Inner) where {T}
    ndims(img) == ndims(inr) || throw(DimensionMismatch("dimensionality of img and the border must agree, got $(ndims(img)) and $(ndims(inr))"))
    inds = inner.(inr.lo, axes(img), inr.hi)
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
