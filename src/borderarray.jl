using Base: @propagate_inbounds

function compatible_dimensions(arr::AbstractArray, border::Inner)
    true
end

function compatible_dimensions(arr,border)
    ndims(arr) == ndims(border)
end

function convert_border_eltype(inner, border::Fill)
    T = eltype(inner)
    val::T = try
        convert(T, border.value)
    catch err
        msg = "Cannot convert elements of border=$border to eltype(inner)=$T."
        throw(ArgumentError(msg))
    end
    Fill(val, border.lo, border.hi)
end

function convert_border_eltype(inner, border)
    border
end

"""
    BorderArray(inner::AbstractArray, border::AbstractBorder) <: AbstractArray

Construct a thin wrapper around the array `inner`, with given `border`. No data is copied in the constructor, instead border values are computed on the fly in `getindex` calls. Usful for stencil computations. See also [padarray](@ref).

# Examples
```julia
julia> using ImageFiltering

julia> arr = reshape(1:6, (2,3))
2×3 reshape(::UnitRange{Int64}, 2, 3) with eltype Int64:
 1  3  5
 2  4  6

julia> BorderArray(arr, Pad((1,1)))
BorderArray{Int64,2,Base.ReshapedArray{Int64,2,UnitRange{Int64},Tuple{}},Pad{2}} with indices 0:3×0:4:
 1  1  3  5  5
 1  1  3  5  5
 2  2  4  6  6
 2  2  4  6  6

julia> BorderArray(arr, Fill(10, (2,1)))
BorderArray{Int64,2,Base.ReshapedArray{Int64,2,UnitRange{Int64},Tuple{}},Fill{Int64,2}} with indices -1:4×0:4:
 10  10  10  10  10
 10  10  10  10  10
 10   1   3   5  10
 10   2   4   6  10
 10  10  10  10  10
 10  10  10  10  10
```
"""
struct BorderArray{T,N,A,B} <: AbstractArray{T,N}
    inner::A
    border::B
    function BorderArray(arr::AbstractArray{T,N}, border::AbstractBorder) where {T,N}
        if !compatible_dimensions(arr, border)
            msg = "$border lacks the proper padding sizes for an array with $(ndims(arr)) dimensions."
            throw(ArgumentError(msg))
        end
        border = convert_border_eltype(arr, border)
        A = typeof(arr)
        B = typeof(border)
        new{T,N,A,B}(arr, border)
    end
end

# these are adapted from OffsetArrays
function Base.similar(A::BorderArray, ::Type{T}, dims::Dims) where T
    B = similar(A.inner, T, dims)
end
const OffsetAxis = Union{Integer, UnitRange, Base.OneTo, IdentityUnitRange}
function Base.similar(A::BorderArray, ::Type{T}, inds::Tuple{OffsetAxis,Vararg{OffsetAxis}}) where T
    similar(A.inner, T, axes(A))
end

@inline function Base.axes(o::BorderArray)
    _outeraxes(o.inner, o.border)
end

@inline function Base.size(o::BorderArray)
    map(length, axes(o))
end

@inline function _outeraxes(arr, border::Inner)
    axes(arr)
end

@inline function _outeraxes(arr, border)
    map(axes(arr), border.lo, border.hi) do r, lo, hi
        (first(r)-lo):(last(r)+hi)
    end
end

@inline @propagate_inbounds function Base.getindex(o::BorderArray{T,N}, inds::Vararg{Int,N}) where {T,N}
    ci = CartesianIndex(inds)
    o[ci]
end

@inline function Base.getindex(o::BorderArray, ci::CartesianIndex)
    if checkbounds(Bool, o.inner, ci)
        @inbounds o.inner[ci]
    else
        @boundscheck checkbounds(o, ci)
        getindex_outer_inbounds(o.inner, o.border, ci)
    end
end

function getindex_outer_inbounds(arr, b::Fill, index)
    b.value
end

function _inner_index(arr, b::Pad, index::CartesianIndex)
    s = b.style
    inds = if s == :replicate
        map(index.I, axes(arr)) do i, r
            clamp(i, first(r), last(r))
        end
    elseif s == :circular
        map(modrange, index.I, axes(arr))
    elseif s == :reflect
        map(index.I, axes(arr)) do i, r
            if i > last(r)
                i = 2last(r) - i
            elseif i < first(r)
                i = 2first(r) - i
            end
            i
        end
    elseif s == :symmetric
        map(index.I, axes(arr)) do i, r
            if i > last(r)
                i = 2last(r) + 1 - i
            elseif i < first(r)
                i = 2first(r) -1 - i
            end
            i
        end
    else
        error("border style $(s) unrecognized")
    end
    CartesianIndex(inds)
end

function getindex_outer_inbounds(arr, b::Pad, index::CartesianIndex)
    i = _inner_index(arr, b, index)
    @inbounds arr[i]
end

function Base.checkbounds(::Type{Bool}, arr::BorderArray, index::CartesianIndex)
    map(axes(arr), index.I) do r, i
        checkindex(Bool, r, i)
    end |> all
end

function Base.copy!(dst::AbstractArray, src::BorderArray)
    axes(dst) == axes(src) || throw(DimensionMismatch("axes(dst) == axes(src) must hold."))
    _copy!(dst, src, src.border)
end
function Base.copy!(dst::AbstractArray{T,1} where T, src::BorderArray{T,1,A,B} where B where A where T)
    # fix ambiguity
    axes(dst) == axes(src) || throw(DimensionMismatch("axes(dst) == axes(src) must hold."))
    _copy!(dst, src, src.border)
end

function _copy!(dst, src, ::Inner)
    copyto!(dst, src.inner)
end

function _copy!(dst, src, ::Pad)
    inds = padindices(src.inner, src.border)
    copydata!(dst, src.inner, inds)
end

function _copy!(dst, src, ::Fill)
    try
        fill!(dst, src.border.value)
    catch
        error("Unable to fill! an array of element type $(eltype(dst)) with the value $(src.border.value). Supply an appropriate value to `Fill`, such as `zero(eltype(A))`.")
    end
    dst[axes(src.inner)...] = src.inner
    dst
end
