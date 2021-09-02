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
BlobLoG(; location, σ, amplitude) = BlobLoG(location, σ, amplitude)

function Base.show(io::IO, bl::BlobLoG)
    print(io, "BlobLoG(location=", bl.location, ", σ=", bl.σ, ", amplitude=", bl.amplitude, ")")
end


"""
    blob_LoG(img, σscales; edges::Tuple=(true, false, ...), σshape::Tuple=(1, ...), rthresh=0.001) -> Vector{BlobLoG}

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

`rthresh` controls the minimum amplitude of peaks in the -LoG-filtered image,
as a fraction of `maximum(abs, img)` and the volume of the Gaussian.

# Examples

While most images are 2- or 3-dimensional, it will be easier to illustrate this with
a one-dimensional "image" containing two Gaussian blobs of different sizes:

```jldoctest; setup=:(using ImageFiltering), filter=r"amplitude=.*"]
julia> σs = 2.0.^(1:6);

julia> img = zeros(100); img[20:30] = [exp(-x^2/(2*4^2)) for x=-5:5]; img[50:80] = [exp(-x^2/(2*8^2)) for x=-15:15];

julia> blob_LoG(img, σs; edges=false)
2-element Vector{BlobLoG{Float64, Tuple{Float64}, 1}}:
 location=CartesianIndex(25,), σ=(4.0,), amplitude=0.10453155018303673
 location=CartesianIndex(65,), σ=(8.0,), amplitude=0.046175719034527364
```

The other two are centered in their corresponding "features," and the width `σ`
reflects the width of the feature itself.

`blob_LoG` tends to work best for shapes that are "Gaussian-like" but does
generalize somewhat.

# Citation:

Lindeberg T (1998), "Feature Detection with Automatic Scale Selection",
International Journal of Computer Vision, 30(2), 79–116.

See also: [`BlobLoG`](@ref).
"""
function blob_LoG(img::AbstractArray{T,N}, σscales;
                  edges::Union{Bool,Tuple{Bool,Vararg{Bool,N}}}=(true, ntuple(d->false, Val(N))...),
                  σshape::NTuple{N,Real}=ntuple(d->1, Val(N)),
                  rthresh::Real=1//1000) where {T<:Union{AbstractGray,Real},N}
    if edges isa Bool
        edges = (edges, ntuple(d->edges,Val(N))...)
    end
    sigmas = sort(σscales)
    img_LoG = multiLoG(img, sigmas, σshape)
    maxima = findlocalmaxima(img_LoG; edges=edges)
    # The "density" should not be much smaller than 1/volume of the Gaussian
    if !iszero(rthresh)
        athresh = rthresh./(sigmas.^N .* prod(σshape))
        imgmax = maximum(abs, img)
        [BlobLoG(CartesianIndex(tail(x.I)), sigmas[x[1]].*σshape, img_LoG[x]) for x in maxima if img_LoG[x] > athresh[x[1]]*imgmax]
    else
        [BlobLoG(CartesianIndex(tail(x.I)), sigmas[x[1]].*σshape, img_LoG[x]) for x in maxima]
    end
end

function multiLoG(img::AbstractArray{T,N}, sigmas, σshape) where {T,N}
    issorted(sigmas) || error("sigmas must be sorted")
    img_LoG = similar(img, float(eltype(T)), (Base.OneTo(length(sigmas)), axes(img)...))
    colons = ntuple(d->Colon(), Val(N))
    @inbounds for (isigma, σ) in enumerate(sigmas)
        LoG_slice = @view img_LoG[isigma, colons...]
        imfilter!(LoG_slice, img, Kernel.LoG(ntuple(i->σ*σshape[i], Val(N))), "reflect")
        LoG_slice .*= -σ
    end
    return img_LoG
end

default_window(img) = (cs = coords_spatial(img); ntuple(d -> d ∈ cs ? 3 : 1, ndims(img)))

"""
    findlocalmaxima(img; window=default_window(img), edges=true) -> Vector{CartesianIndex}

Returns the coordinates of elements whose value is larger than all of
their immediate neighbors.  `edges` is a boolean specifying whether to include the
first and last elements of each dimension, or a tuple-of-Bool
specifying edge behavior for each dimension separately.

The `default_window` is 3 for each spatial dimension of `img`, and 1 otherwise, implying
that maxima are detected over nearest-neighbors in each spatial "slice" by default.
"""
findlocalmaxima(img::AbstractArray; window=default_window(img), edges=true) =
    findlocalextrema(>, img, window, edges)

"""
    findlocalminima(img; window=default_window(img), edges=true) -> Vector{CartesianIndex}

Like [`findlocalmaxima`](@ref), but returns the coordinates of the smallest elements.
"""
findlocalminima(img::AbstractArray; window=default_window(img), edges=true) =
    findlocalextrema(<, img, window, edges)


findlocalextrema(f, img::AbstractArray{T,N}, window, edges::Bool) where {T,N} = findlocalextrema(f, img, window, ntuple(d->edges,Val(N)))

function findlocalextrema(f::F, img::AbstractArray{T,N}, window::Dims{N}, edges::NTuple{N,Bool}) where {F,T<:Union{Gray,Number},N}
    extrema = Vector{CartesianIndex{N}}(undef, 0)
    Iedge = CartesianIndex(map(!, edges))
    R0 = CartesianIndices(img)
    R = clippedinds(R0, Iedge)
    halfwindow = CartesianIndex(map(x -> x >> 1, window))
    Rinterior = clippedinds(R0, halfwindow)
    Rwindow = _colon(-halfwindow, halfwindow)
    z = zero(halfwindow)
    for i in R
        isextrema = true
        img_i = img[i]
        if i ∈ Rinterior
            # If i is in the interior, we don't have to worry about i+j being out-of-bounds
            for j in Rwindow
                j == z && continue
                if !f(img_i, img[i+j])
                    isextrema = false
                    break
                end
            end
        else
            for j in Rwindow
                (j == z || i+j ∉ R0) && continue
                if !f(img_i, img[i+j])
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
