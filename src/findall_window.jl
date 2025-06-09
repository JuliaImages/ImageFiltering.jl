"""
    findall_window(f, img::AbstractArray{T,N}, window::Dims{N}; allow_edges::(true...,))

Return the list of window-centers for which `f(V, basepoint)` returns `true` for windowed-views `V`
"centered" on `basepoint`. At the edge of `img`, `basepoint` may not actually be in the center of `V`.
"""
function findall_window(f::F, img::AbstractArray{T,N}, window::Dims{N}; allow_edges::NTuple{N,Bool}=ntuple(_->true, N)) where {F,T<:Union{Gray,Number},N}
    basepoints = Vector{CartesianIndex{N}}(undef, 0)
    Iedge = CartesianIndex(map(!, allow_edges))
    R0 = CartesianIndices(img)
    R = clippedinds(R0, Iedge)
    halfwindow = CartesianIndex(map(x -> x >> 1, window))
    for i in R
        Rview = _colon(i-halfwindow, i+halfwindow) ∩ R0
        if f(OffsetArray(@inbounds(view(img, Rview)), Rview.indices), i)
            push!(basepoints, i)
        end
    end
    return basepoints
end

"""
    all_window(f, V, basepoint, excludepoint=nothing)

Returns `true` if `f(img[centerpoint], img[otherpoint])` for all indices `otherpoint`.
Optionally exclude a single point (e.g., `basepoint` itself) by setting `excludepoint`.
"""
@inline function all_window(f::F, V::AbstractArray, basepoint::CartesianIndex, excludepoint::Union{Nothing,CartesianIndex}=nothing) where F
    @inbounds ref = V[basepoint]
    @inbounds for (i, v) in pairs(V)
        i == excludepoint || f(ref, v) || return false
    end
    return true
end

"""
    ismax_window(V, basepoint)

Returns `true` if `V[basepoint] > V[otherpoint]` for all indices `otherpoint != basepoint` in `V`.
"""
@inline ismax_window(V, i) = all_window(>, V, i, i)

"""
    ismin_window(V, basepoint)

Returns `true` if `V[basepoint] < V[otherpoint]` for all indices `otherpoint != basepoint` in `V`.
"""
@inline ismin_window(V, i) = all_window(<, V, i, i)

findmax_window(A, window; kwargs...) = findall_window(ismax_window, A, window; kwargs...)
findmin_window(A, window; kwargs...) = findall_window(ismin_window, A, window; kwargs...)
