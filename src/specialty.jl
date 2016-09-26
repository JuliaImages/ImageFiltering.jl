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
    imgradients(img, kernelfun=KernelFactors.ando3, border="replicate") -> gimg1, gimg2, ...

Estimate the gradient of `img` at all points of the image, using a
kernel specified by `kernelfun`. The gradient is returned as a
tuple-of-arrays, one for each dimension of the input; `gimg1`
corresponds to the derivative with respect to the first dimension,
`gimg2` to the second, and so on. At the image edges, `border` is used
to specify the boundary conditions.

`kernelfun` may be one of the filters defined in the `KernelFactors`
module, or more generally any function which satisfies the following
interface:

    kernelfun(extended::NTuple{N,Bool}, d) -> kern_d

`kern_d` is the kernel for producing the derivative with respect to
the `d`th dimension of an `N`-dimensional array. `extended[i]` is true
if the image is of size > 1 along dimension `i`. `kern_d` may be
provided as a dense or factored kernel, with factored representations
recommended when the kernel is separable.
"""
function imgradients(img::AbstractArray, kernelfun::Function, border="replicate")
    extended = map(isextended, indices(img))
    _imgradients((), img, kernelfun, extended, border)
end

# For the 0.6 release of Images, we need to warn users about the
# switch in the order of the outputs. Use the provision of kernelfun
# as the test for the new interface.
function imgradients(img::AbstractArray)
    depwarn("the order of outputs has switched (`grad1, grad2 = imgradients(img)` rather than `gradx, grady = imgradients`). Silence this warning by providing a kernelfun, e.g., imgradients(img, KernelFactors.ando3).", :imgradients)
    imgradients(img, KernelFactors.ando3)
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

"""
    imgradients(img, points, [kernelfunc], [border]) -> G

Performs edge detection filtering in the N-dimensional array `img`.
Gradients are computed at specified `points` (or indexes) in the
array.

All kernel functions are specified as `KernelFactors.func`. For 2d
images, the choices for `func` include `sobel`, `prewitt`, `ando3`,
`ando4`, and `ando5`. For other dimensionalities, the `ando4` and
`ando5` kernels are not available.

Border options:`"replicate"`, `"circular"`, `"reflect"`, `"symmetric"`.

Returns a 2D array `G` with the gradients as rows. The number of rows
is the number of points at which the gradient was computed and the
number of columns is the dimensionality of the array.
"""
function imgradients{T,N}(img::AbstractArray{T,N}, points::AbstractVector,
                          method=KernelFactors.ando3, border::AbstractString="replicate")
    extended = map(isextended, indices(img))
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
