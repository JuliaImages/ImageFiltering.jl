"""
BlobLoG stores information about the location of peaks as discovered by `blob_LoG`.
It has fields:

- location: the location of a peak in the filtered image (a CartesianIndex)
- σ: the value of σ which lead to the largest `-LoG`-filtered amplitude at this location
- amplitude: the value of the `-LoG(σ)`-filtered image at the peak

Note that the radius is equal to σ√2.

See also: [`blob_LoG`](@ref).
"""
struct BlobLoG{T,S,N}
    location::CartesianIndex{N}
    σ::S
    amplitude::T
end

"""
    blob_LoG(img, σscales; edges=(true, false, ...), σshape=(1, ...)) -> Vector{BlobLoG}

Find "blobs" in an N-D image using the negative Lapacian of Gaussians
with the specifed vector or tuple of σ values. The algorithm searches for places
where the filtered image (for a particular σ) is at a peak compared to all
spatially- and σ-adjacent voxels, where σ is `σscales[i] * σshape` for some i.
By default, `σshape` is an ntuple of 1s.

The optional `edges` argument controls whether peaks on the edges are
included. `edges` can be `true` or `false`, or a N+1-tuple in which
the first entry controls whether edge-σ values are eligible to serve
as peaks, and the remaining N entries control each of the N dimensions
of `img`.

# Citation:

Lindeberg T (1998), "Feature Detection with Automatic Scale Selection",
International Journal of Computer Vision, 30(2), 79–116.

See also: [`BlobLoG`](@ref).
"""
function blob_LoG(img::AbstractArray{T,N}, σscales::Union{AbstractVector,Tuple};
                  edges::Union{Bool,Tuple{Vararg{Bool}}}=(true, ntuple(d->false, Val(N))...), σshape=ntuple(d->1, Val(N))) where {T,N}
    if edges isa Bool
        edges = (edges, ntuple(d->edges,Val(N))...)
    end
    sigmas = sort(σscales)
    img_LoG = Array{Float64}(undef, length(sigmas), size(img)...)
    colons = ntuple(d->Colon(), Val(N))
    @inbounds for isigma in eachindex(sigmas)
        img_LoG[isigma,colons...] = (-sigmas[isigma]) * imfilter(img, Kernel.LoG(ntuple(i->sigmas[isigma]*σshape[i],Val(N))))
    end
    maxima = findlocalmaxima(img_LoG; dims=1:ndims(img_LoG), edges=edges)
    [BlobLoG(CartesianIndex(tail(x.I)), sigmas[x[1]], img_LoG[x]) for x in maxima]
end


"""
    findlocalmaxima(img; dims=coords_spatial(img), edges=true) -> Vector{CartesianIndex}

Returns the coordinates of elements whose value is larger than all of
their immediate neighbors.  `dims` is a list of dimensions to
consider.  `edges` is a boolean specifying whether to include the
first and last elements of each dimension, or a tuple-of-Bool
specifying edge behavior for each dimension separately.
"""
findlocalmaxima(img::AbstractArray; dims=coords_spatial(img), edges=true) =
        findlocalextrema(img, dims, edges, Base.Order.Forward)

"""
    findlocalminima(img; dims=coords_spatial(img), edges=true) -> Vector{CartesianIndex}

Like [`findlocalmaxima`](@ref), but returns the coordinates of the smallest elements.
"""
findlocalminima(img::AbstractArray; dims=coords_spatial(img), edges=true) =
    findlocalextrema(img, dims, edges, Base.Order.Reverse)


findlocalextrema(img::AbstractArray{T,N}, dims, edges::Bool, order) where {T,N} = findlocalextrema(img, dims, ntuple(d->edges,Val(N)), order)

function findlocalextrema(img::AbstractArray{T,N}, dims, edges::NTuple{N,Bool}, order::Base.Order.Ordering) where {T<:Union{Gray,Number},N}
    dims ⊆ 1:ndims(img) || throw(ArgumentError("invalid dims"))
    extrema = Array{CartesianIndex{N}}(undef, 0)
    Iedge = CartesianIndex(map(!, edges))
    R0 = CartesianIndices(img)
    R = clippedinds(R0, Iedge)
    I1 = _oneunit(first(R0))
    Rinterior = clippedinds(R0, I1)
    iregion = CartesianIndex(ntuple(d->d ∈ dims, Val(N)))
    Rregion = CartesianIndices(map((f,l)->f:l,(-iregion).I, iregion.I))
    z = zero(iregion)
    for i in R
        isextrema = true
        img_i = img[i]
        if i ∈ Rinterior
            # If i is in the interior, we don't have to worry about i+j being out-of-bounds
            for j in Rregion
                j == z && continue
                if !Base.Order.lt(order, img[i+j], img_i)
                    isextrema = false
                    break
                end
            end
        else
            for j in Rregion
                (j == z || i+j ∉ R0) && continue
                if !Base.Order.lt(order, img[i+j], img_i)
                    isextrema = false
                    break
                end
            end
        end
        isextrema && push!(extrema, i)
    end
    extrema
end

clippedinds(Router, Iclip) = _colon(first(Router)+Iclip, last(Router)-Iclip)
