## Laplacian

function _imfilter_inbounds!(r, out, A::AbstractArray, L::Laplacian, border::NoPad, inds)
    TT = eltype(out) # accumtype(eltype(out), eltype(A))
    n = 2*length(L.offsets)
    R = CartesianRange(inds)
    @unsafe for I in R
        tmp = convert(TT, - n * A[I])
        for J in L.offsets
            tmp += A[I+J]
            tmp += A[I-J]
        end
        out[I] = tmp
    end
    out
end

## imgradients
"""
    imgradients(img, [points], [method], [border]) -> g1, g2, ...

Performs edge detection filtering of the N-dimensional array `img`.
Gradients are computed at specified `points` (or indexes) in the
array or everywhere. Available methods: `"sobel"` and `"ando3"`.
Border options:`"replicate"`, `"circular"`, `"reflect"`, `"symmetric"`.
Returns a 2D array `G` with the gradients as rows. The number of rows
is the number of points at which the gradient was computed and the
number of columns is the dimensionality of the array.
"""
function imgradients{T,N}(img::AbstractArray{T,N}, points::AbstractVector;
                          method::AbstractString="ando3", border::AbstractString="replicate")
    extent = size(img)
    ndirs = length(extent)
    npoints = length(points)

    # pad input image only on appropriate directions
    imgpad = _gradientpad(img, border)

    # gradient matrix
    Tret = typeof(zero(T)*zero(Float64))
    G = zeros(Tret, npoints, ndirs)

    for dir in 1:ndirs
        # kernel = centered difference + perpendicular smoothing
        if extent[dir] > 1
            kern = _directional_kernel(dir, extent, method)

            # compute gradient at specified points
            A = zeros(kern)
            shape = size(kern)
            for (k, p) in enumerate(points)
                icenter = CartesianIndex(ind2sub(extent, p))
                i1 = CartesianIndex(tuple(ones(Int, ndirs)...))
                for ii in CartesianRange(shape)
                    A[ii] = imgpad[ii + icenter - i1]
                end

                G[k,dir] = sum(kern .* A)
            end
        end
    end

    G
end

function imgradients{T,N}(img::AbstractArray{T,N}; method::AbstractString="ando3", border::AbstractString="replicate")
    extent = size(img)
    ndirs = length(extent)

    # pad input image only on appropriate directions
    imgpad = _gradientpad(img, border)

    # gradient tuple
    Tret = typeof(zero(T)*zero(Float64))
    G = Array(Array{Tret,N}, ndirs)

    for dir in 1:ndirs
        # kernel = centered difference + perpendicular smoothing
        if extent[dir] > 1
            kern = _directional_kernel(dir, extent, method)
            G[dir] = imfilter(imgpad, kern, "inner")
        end
    end

    result = (G...,)
    if ndims(img) == 2 && spatialorder(img) == yx
      result = (result[2], result[1])
    end

    result
end
