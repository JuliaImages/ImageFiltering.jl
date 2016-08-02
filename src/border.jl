module Border

using OffsetArrays, CatIndices

# export AbstractBorder, Replicate, Circular, Symmetric, Reflect, Inner, Fill

export padindices, padarray

abstract AbstractBorder

# lo and hi are the amount to pad by along each dimension
immutable Replicate{N} <: AbstractBorder
    lo::Dims{N}  # number to extend by on the lower edge
    hi::Dims{N}  # number to extend by on the upper edge
end

immutable Circular{N} <: AbstractBorder
    lo::Dims{N}
    hi::Dims{N}
end

immutable Symmetric{N} <: AbstractBorder
    lo::Dims{N}
    hi::Dims{N}
end

immutable Reflect{N} <: AbstractBorder
    lo::Dims{N}
    hi::Dims{N}
end

immutable Inner{N} <: AbstractBorder
    lo::Dims{N}
    hi::Dims{N}
end

for B in (:Replicate, :Circular, :Symmetric, :Reflect, :Inner)
    b = Symbol(lowercase(string(B)))
    @eval begin
        function $B(kernel::AbstractArray)
            inds = indices(kernel)
            $B(map(lo, inds), map(hi, inds))
        end
        function $B(factkernel::Tuple)
            inds = flatten(map(indices, factkernel))
            $B(map(lo, inds), map(hi, inds))
        end
        padindices(img, border::$B) = map($b, border.lo, indices(img), border.hi)
    end
end

padarray(img, border::AbstractBorder) = img[padindices(img, border)...]

# Fill is a little different, so handle it separately
immutable Fill{T} <: AbstractBorder
    value::T
end

immutable FillSized{T,N} <: AbstractBorder
    value::T
    lo::Dims{N}
    hi::Dims{N}
end
function (f::Fill)(kernel::AbstractArray)
    inds = indices(kernel)
    FillSized(f.value, map(lo, inds), map(hi, inds))
end
function (f::Fill)(factkernel::Tuple)
    inds = (map(indices, kernel)...,)
    FillSized(f.value, map(lo, inds), map(hi, inds))
end

padarray(img, f::Fill) = throw(ArgumentError("need indices of kernel, call padarray(img, f(kernel))"))
function padarray{T}(img, f::FillSized{T})
    S = promote_type(eltype(img), T)
    A = similar(Array{S}, map((l,r,h)->first(r)-l:last(r)+h, f.lo, indices(img), f.hi))
    fill!(A, f.value)
    A[indices(img)...] = img
    A
end

# There are other ways to define these, but using `mod` makes it safe
# for cases where the padding is bigger than length(inds)
function replicate(lo::Integer, inds::AbstractUnitRange, hi::Integer)
    vcat(fill(first(inds), lo), PinIndices(inds), fill(last(inds), hi))
end
function circular(lo::Integer, inds::AbstractUnitRange, hi::Integer)
    modrange(extend(lo, inds, hi), inds)
end
function symmetric(lo::Integer, inds::AbstractUnitRange, hi::Integer)
    I = [inds; reverse(inds)]
    I[modrange(extend(lo, inds, hi), 1:2*length(inds))]
end
function reflect(lo::Integer, inds::AbstractUnitRange, hi::Integer)
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

end
