## Laplacian

function _imfilter_inbounds!(r::AbstractResource, out, A::AbstractArray, L::Laplacian, border::NoPad, inds)
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

function imgradients{T,N}(img::AbstractArray{T,N}, kernelfun::Function=KernelFactors.ando3, border=Pad{:replicate}())
    extended = map(isextended, indices(img))
    # kern1 = kernelfun(extended, 1)
    # S = filter_type(img, kern1)
    _imgradients((), img, kernelfun, extended, border)
end
isextended(ind) = length(ind) > 1

# When all N gradients have been calculated, return the result
_imgradients{T,N}(G::NTuple{N}, img::AbstractArray{T,N}, kernelfun::Function, extent, border) = G

# Add the next dimension to G
function _imgradients{T,M,N}(G::NTuple{M}, img::AbstractArray{T,N}, kernelfun::Function, extended, border)
    d = M+1  # the dimension we're working on now
    kern = kernelfun(extended, d)
    _imgradients((G..., imfilter(img, kern, border)), img, kernelfun, extended, border)
end
