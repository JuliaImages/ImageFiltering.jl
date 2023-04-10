## Laplacian

function _imfilter_inbounds!(r::AbstractResource, out, A::AbstractArray, L::Laplacian, border::NoPad, inds)
    TT = eltype(out) # accumtype(eltype(out), eltype(A))
    n = 2 * length(L.offsets)
    R = CartesianIndices(inds)
    @inbounds for I in R
        tmp = convert(TT, -n * A[I])
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

Estimate the gradient of `img` in the direction of the first and second
dimension at all points of the image, using a kernel specified by `kernelfun`.

Returns a tuple-of-arrays, `(gimg1, gimg2, ...)`, one for each dimension of the
input: `gimg1` corresponds to the derivative with respect to the first
dimension, `gimg2` to the second, and so on.

## Example

```julia
using Images, ImageFiltering, TestImages
img = testimage("mandrill")
imgr = imgradients(img, KernelFactors.sobel, "reflect")
mosaicview(imgr...)
```
"""
function imgradients(img::AbstractArray, kernelfun::Function, border="replicate")
    extended = map(isextended, axes(img))
    _imgradients(extended, img, kernelfun, extended, border)
end

isextended(ind) = length(ind) > 1

# Add the next dimension to G
function _imgradients(donewhenempty::NTuple{M}, img::AbstractArray{T,N}, kernelfun::Function, extended, border) where {T,M,N}
    d = N - M + 1  # the dimension we're working on now
    kern = kernelfun(extended, d)
    return (imfilter(img, kern, border), _imgradients(Base.tail(donewhenempty), img, kernelfun, extended, border)...)
end
# When all N gradients have been calculated, return the result
_imgradients(::Tuple{}, img::AbstractArray, kernelfun::Function, extent, border) = ()
