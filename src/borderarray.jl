using Base: @propagate_inbounds

struct BorderArray{T,N,A,B} <: AbstractArray{T,N}
    inner::A
    border::B
    function BorderArray(arr::AbstractArray{T,N}, border::AbstractBorder) where {T,N}
        A = typeof(arr)
        B = typeof(border)
        new{T,N,A,B}(arr, border)
    end
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
