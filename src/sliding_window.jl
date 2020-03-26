
"""
    sliding_window(f, arr::AbstractArray, window;
        border=:replicate, [provisioning,]
    )

!!! compat "Julia 1.2"
    sliding_window is only available on Julia 1.2 and higher.
"""
function sliding_window end


abstract type GetindexProvisioning end
struct CopyProvisioning <: GetindexProvisioning end
struct ViewProvisioning <: GetindexProvisioning end

struct SlidingWindowArray{T,N,A,S} <: AbstractArray{T,N}
    padded_array::A
    axes::NTuple{N, UnitRange{Int}}
    window::NTuple{N, UnitRange{Int}}
    state::S
end

Base.IndexStyle(::Type{<:SlidingWindowArray}) = IndexCartesian()

Base.axes(arr::SlidingWindowArray) = arr.axes
Base.size(arr::SlidingWindowArray) = map(length, arr.axes)

function typeof_state(::Type{SlidingWindowArray{T,N,A,S}}) where {T,N,A,S}
    S
end

GetindexProvisioning(arr::SlidingWindowArray) = GetindexProvisioning(typeof(arr))

function GetindexProvisioning(::Type{A}) where {A<:SlidingWindowArray}
    S = typeof_state(A)
    if S == Nothing
        ViewProvisioning()
    else
        @assert S<: Array
        CopyProvisioning()
    end
end

Base.@propagate_inbounds function Base.getindex(arr::SlidingWindowArray, I...)
    ci = CartesianIndices(arr)[I...]
    @inbounds arr[ci]
end

Base.@propagate_inbounds function Base.getindex(arr::SlidingWindowArray, I::CartesianIndex)
    _getindex(arr, I, GetindexProvisioning(arr))
end

function _compute_lo_hi(window)
    lo = map(window) do win
        max(-first(win), 0)
    end
    hi = map(window) do win
        max(last(win), 0)
    end
    lo, hi
end

function resolve_border2(pad::Pad{0}, window)
    lo, hi = _compute_lo_hi(window)
    Pad(pad.style, lo, hi)
end
function resolve_border2(o::Fill{T,0}, window) where {T}
    lo, hi = _compute_lo_hi(window)
    Fill(o.value, lo, hi)
end
function resolve_border2(style::Symbol, window)
    resolve_border2(Pad(style,(),()), window)
end
function resolve_border2(border, window)
    border
end

function resolve_provisioning(provisioning::Symbol)
    if provisioning == :copy
        CopyProvisioning()
    elseif provisioning == :view
        ViewProvisioning()
    else
        msg = "Unknown provisioning $provisioning"
        throw(ArgumentError(msg))
    end
end
resolve_provisioning(p::GetindexProvisioning) = p

default_provisioning(::typeof(padarray)) = ViewProvisioning()
default_provisioning(::Type{<:BorderArray}) = CopyProvisioning()

function sliding_window(f_pad, arr::AbstractArray, window;
        border=:replicate,
        provisioning=default_provisioning(f_pad)
    )

    provisioning=resolve_provisioning(provisioning)
    window = resolve_window(window)
    border = resolve_border2(border, window)
    @assert length(window) == ndims(arr)
    padded_array = f_pad(arr, border)
    SlidingWindowArray(padded_array, axes(arr), window, provisioning)
end

function sliding_window(arr::AbstractArray, window; kw...)
    sliding_window(padarray, arr, window; kw...)
end

function compute_sliding_window_eltype(padded_array::AbstractArray{T0,N}, axes, window, ::CopyProvisioning) where {T0,N}
    Array{T0,N}
end

function compute_sliding_window_eltype(padded_array::AbstractArray, axes, window, ::ViewProvisioning)
    I = CartesianIndex(map(first, axes))
    inds = shift_window(I, window)
    typeof(view(padded_array, inds...))
end

@inline function SlidingWindowArray(arr::AbstractArray, window;
        border=:replicate,
        provisioning=CopyProvisioning())

    SlidingWindowArray(arr, window, border, provisioning)
end

create_state(padded_array, window, ::ViewProvisioning) = nothing

function create_state(padded_array, window, ::CopyProvisioning)
    T0 = eltype(padded_array)
    dims = map(length, window)
    N = ndims(padded_array)
    T = Array{T0,N}
    state = T(undef, dims)
end

function SlidingWindowArray(padded_array::AbstractArray{T0,N}, axes, window, provisioning) where {T0,N}
    state = create_state(padded_array, window, provisioning)
    S = typeof(state)
    A = typeof(padded_array)
    T = compute_sliding_window_eltype(padded_array, axes, window, provisioning)
    SlidingWindowArray{T,N,A,S}(padded_array, axes, window, state)
end

function shift_window(ci::CartesianIndex, window)
    map(Tuple(ci), window) do offset, range
        offset .+ range
    end
end

Base.@propagate_inbounds function _getindex(arr::SlidingWindowArray, I::CartesianIndex, ::ViewProvisioning)
    inds = shift_window(I, arr.window)
    view(arr.padded_array, inds...)
end

Base.@propagate_inbounds function _getindex(o::SlidingWindowArray, ci::CartesianIndex, ::CopyProvisioning)
    winaxes::Tuple = shift_window(ci, o.window)
    win = CartesianIndices(winaxes)
    copyto!(o.state, CartesianIndices(o.state), o.padded_array, win)
    o.state
end
