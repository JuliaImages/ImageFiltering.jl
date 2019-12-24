## Laplacian

function _imfilter_inbounds!(r::AbstractResource, out, A::AbstractArray, L::Laplacian, border::NoPad, inds)
    TT = eltype(out) # accumtype(eltype(out), eltype(A))
    n = 2*length(L.offsets)
    R = CartesianIndices(inds)
    @inbounds for I in R
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
```julia
    imgradients(img, kernelfun=KernelFactors.ando3, border="replicate") -> gimg1, gimg2, ...
```
Estimate the gradient of `img` in the direction of the first and second dimension
at all points of the image, using a kernel specified by `kernelfun`.

# Output

The gradient is returned as a tuple-of-arrays, one for each dimension of the
input; `gimg1` corresponds to the derivative with respect to the first
dimension, `gimg2` to the second, and so on.

# Details

To appreciate the difference between various gradient estimation methods
it is helpful to distinguish between: (1) a continuous scalar-valued
*analogue* image ``f_\\textrm{A}(x_1,x_2)``, where ``x_1,x_2 \\in
\\mathbb{R}``, and (2) its discrete *digital* realization
``f_\\textrm{D}(x_1',x_2')``, where ``x_1',x_2' \\in \\mathbb{N}``, ``1
\\le x_1' \\le M`` and ``1 \\le x_2' \\le N``.

## Analogue image

The gradient of a continuous analogue image ``f_{\\textrm{A}}(x_1,x_2)`` at
location ``(x_1,x_2)`` is defined as the vector
```math
\\nabla \\mathbf{f}_{\\textrm{A}}(x_1,x_2) = \\frac{\\partial
f_{\\textrm{A}}(x_1,x_2)}{\\partial x_1} \\mathbf{e}_{1} +
\\frac{\\partial f_{\\textrm{A}}(x_1,x_2)}{\\partial x_2} \\mathbf{e}_{2},
```
where ``\\mathbf{e}_{d}`` ``(d = 1,2)`` is the unit
vector in the ``x_d``-direction. The gradient points in the direction of
maximum rate of change of ``f_{\\textrm{A}}`` at the coordinates
``(x_1,x_2)``. The gradient can be used to compute the derivative of a
function in an arbitrary direction. In particular, the derivative of
``f_{\\textrm{A}}`` in the direction of a unit vector ``\\mathbf{u}`` is
given by ``\\nabla_{\\mathbf{u}}f_\\textrm{A}(x_1,x_2) = \\nabla
\\mathbf{f}_{\\textrm{A}}(x_1,x_2) \\cdot \\mathbf{u}``, where
``\\cdot`` denotes the dot product.

## Digital image

In practice, we acquire a digital image ``f_\\textrm{D}(x_1',x_2')`` where
the light intensity is known only at a discrete set of locations. This
means that the required partial derivatives are undefined and need to be
approximated using discrete difference formulae [1].

A straightforward way to approximate the partial derivatives is to use
central-difference formulae
```math
 \\frac{\\partial f_{\\textrm{D}}(x_1',x_2')}{\\partial x_1'}  \\approx
        \\frac{f_{\\textrm{D}}(x_1'+1,x_2') - f_{\\textrm{D}}(x_1'-1,x_2') }{2}
```
and
```math
 \\frac{\\partial f_{\\textrm{D}}(x_1',x_2')}{\\partial x_2'}   \\approx
         \\frac{f_{\\textrm{D}}(x_1',x_2'+1) - f_{\\textrm{D}}(x_1',x_2'+1)}{2}.
```
However, the central-difference formulae are very sensitive to noise.
When working with noisy image data,
one can obtain a better approximation of the partial
derivatives by using a suitable weighted combination of the neighboring
image intensities. The weighted combination can be represented as a
*discrete convolution* operation between the image and a
*kernel* which characterizes the requisite weights. In particular,
if ``h_{x_d}`` (``d = 1,2)`` represents a ``2r+1 \\times 2r+1`` kernel, then
```math
 \\frac{\\partial f_{\\textrm{D}}(x_1',x_2')}{\\partial x_d'}  \\approx
\\sum_{i = -r}^r \\sum_{j = -r}^r
f_\\textrm{D}(x_1'-i,x_2'-j)
  h_{x_d}(i,j).
```
The kernel is frequently also called a *mask* or *convolution matrix*.

### Weighting schemes and approximation error

The choice of weights determines the magnitude of the approximation
error and whether the finite-difference scheme is *isotropic*. A
finite-difference scheme is isotropic if the approximation error does
not depend on the orientation of the coordinate system and
*anisotropic* if the approximation error has a directional bias [2].
With a continuous analogue image the magnitude of the gradient would be
invariant upon rotation of the coordinate system, but in practice one
cannot obtain perfect isotropy with a finite set of discrete points.
Hence a finite-difference scheme is typically considered isotropic if
the leading error term in the approximation does not have preferred
directions.


Most finite-difference schemes that are used in image processing are
based on ``3 \\times 3`` kernels, and as noted by [7], many can also be
parametrized by a single parameter ``\\alpha`` as follows:

```math
\\mathbf{H}_{x_{1}} =
\\frac{1}{4 + 2\\alpha}
\\begin{bmatrix}
-1 & -\\alpha & -1 \\\\
0 & 0 & 0 \\\\
 1 & \\alpha & 1
\\end{bmatrix}
\\quad
\\text{and}
\\quad
\\mathbf{H}_{x_{2}} =
\\frac{1}{2 + 4\\alpha}
\\begin{bmatrix}
-1 & 0 & 1 \\\\
-\\alpha & 0 & \\alpha \\\\
 -1 & 0 & 1
\\end{bmatrix},
```
where
```math
\\alpha =
\\begin{cases}
0,  & \\text{Simple Finite Difference}; \\\\
1, &  \\text{Prewitt}; \\\\
2, &  \\text{Sobel}; \\\\
2.4351, &  \\text{Ando}; \\\\
\\frac{10}{3}, &  \\text{Scharr}; \\\\
4, &  \\text{Bickley}.
\\end{cases}
```

## Separable kernel

A kernel is called *separable* if it can be expressed as the
convolution of two one-dimensional filters. With a matrix representation
of the kernel, separability means that the kernel matrix can be written
as an outer product of two vectors. Separable kernels offer
computational advantages since instead of performing a two-dimensional
convolution one can perform a sequence of one-dimensional convolutions.

# Options
You can specify your choice of the finite-difference scheme via the `kernelfun`
parameter. You can also indicate how to deal with the pixels on the border
of the image with the `border` parameter.

## Choices for `kernelfun`
In general `kernelfun` can be any function which satisfies the following
interface:
```julia
    kernelfun(extended::NTuple{N,Bool}, d) -> kern_d,
```
where `kern_d` is the kernel for producing the derivative with respect to
the ``d``th dimension of an ``N``-dimensional array. The parameter `extended[i]` is true
if the image is of size > 1 along dimension ``i``. The parameter `kern_d` may be
provided as a dense or factored kernel, with factored representations
recommended when the kernel is separable.

Some valid `kernelfun` options are described below.

### `KernelFactors.prewitt`

With the *prewit* option [3] the computation of the gradient is based on
the kernels
```math
\\begin{aligned}
\\mathbf{H}_{x_1} & = \\frac{1}{6}
    \\begin{bmatrix}
    -1 & -1 & -1 \\\\
    0 & 0 & 0 \\\\
    1 & 1 & 1
    \\end{bmatrix}
&
\\mathbf{H}_{x_2} & =  \\frac{1}{6}
    \\begin{bmatrix}
    -1 & 0 & 1 \\\\
    -1 & 0 & 1 \\\\
    -1 & 0 & 1
    \\end{bmatrix} \\\\
& = \\frac{1}{6}
    \\begin{bmatrix}
    1 \\\\
    1  \\\\
    1
    \\end{bmatrix}
    \\begin{bmatrix}
    -1 & 0 & 1
    \\end{bmatrix}
&
& = \\frac{1}{6}
    \\begin{bmatrix}
    -1 \\\\
    0  \\\\
    1
    \\end{bmatrix}
    \\begin{bmatrix}
    1 & 1 & 1
    \\end{bmatrix}.
\\end{aligned}
```
See also: [`KernelFactors.prewitt`](@ref) and [`Kernel.prewitt`](@ref)

### `KernelFactors.sobel`

The *sobel* option [4] designates the kernels
```math
\\begin{aligned}
\\mathbf{H}_{x_1} & = \\frac{1}{8}
    \\begin{bmatrix}
    -1 & -2 & -1 \\\\
     0 & 0 & 0 \\\\
     1 & 2 & 1
    \\end{bmatrix}
&
\\mathbf{H}_{x_2} & = \\frac{1}{8}
    \\begin{bmatrix}
    -1 & 0 & 1 \\\\
    -2 & 0 & 2 \\\\
    -1 & 0 & 1
    \\end{bmatrix} \\\\
& = \\frac{1}{8}
    \\begin{bmatrix}
    -1 \\\\
    0  \\\\
    1
    \\end{bmatrix}
    \\begin{bmatrix}
    1 & 2 & 1
    \\end{bmatrix}
&
& = \\frac{1}{8}
    \\begin{bmatrix}
    1 \\\\
    2  \\\\
    1
    \\end{bmatrix}
    \\begin{bmatrix}
    -1 & 0 & 1
    \\end{bmatrix}.
\\end{aligned}
```
See also:  [`KernelFactors.sobel`](@ref) and [`Kernel.sobel`](@ref)

### `KernelFactors.ando3`
The *ando3* option [5] specifies the kernels
```math
\\begin{aligned}
\\mathbf{H}_{x_1} &  =
    \\begin{bmatrix}
    -0.112737 & -0.274526 & -0.112737 \\\\
     0 & 0 & 0 \\\\
     0.112737 & 0.274526 & 0.112737
    \\end{bmatrix}
&
\\mathbf{H}_{x_2}  & =
    \\begin{bmatrix}
    -0.112737 & 0 & 0.112737 \\\\
    -0.274526 & 0 & 0.274526 \\\\
    -0.112737 & 0 & 0.112737
    \\end{bmatrix} \\\\
&  = \\begin{bmatrix}
    -1 \\\\
    0  \\\\
    1
    \\end{bmatrix}
    \\begin{bmatrix}
    0.112737 & 0.274526 & 0.112737
    \\end{bmatrix}
&
&  = \\begin{bmatrix}
    0.112737 \\\\
    0.274526  \\\\
    0.112737
    \\end{bmatrix}
    \\begin{bmatrix}
    -1 & 0 & 1
    \\end{bmatrix}.
\\end{aligned}
```
See also:  [`KernelFactors.ando3`](@ref), and
[`Kernel.ando3`](@ref);  [`KernelFactors.ando4`](@ref), and
[`Kernel.ando4`](@ref); [`KernelFactors.ando5`](@ref), and
[`Kernel.ando5`](@ref)

### `KernelFactors.scharr`

The *scharr* option [6] designates the kernels
```math
\\begin{aligned}
\\mathbf{H}_{x_{1}} & =
\\frac{1}{32}
\\begin{bmatrix}
-3 & -10 & -3 \\\\
0 & 0 & 0 \\\\
 3 & 10 & 3
\\end{bmatrix}
&
\\mathbf{H}_{x_{2}} & =
\\frac{1}{32}
\\begin{bmatrix}
-3 & 0 & 3 \\\\
-10 & 0 & 10\\\\
-3 & 0 & 3
\\end{bmatrix} \\\\
& = \\frac{1}{32}
\\begin{bmatrix}
    -1 \\\\
    0  \\\\
    1
\\end{bmatrix}
\\begin{bmatrix}
    3 & 10 & 3
\\end{bmatrix}
&
& = \\frac{1}{32}
\\begin{bmatrix}
    3 \\\\
    10  \\\\
    3
\\end{bmatrix}
\\begin{bmatrix}
    -1 & 0 & 1
\\end{bmatrix}.
\\end{aligned}
```
See also:  [`KernelFactors.scharr`](@ref) and [`Kernel.scharr`](@ref)

### `KernelFactors.bickley`

The *bickley* option [7,8] designates the kernels
```math
\\begin{aligned}
\\mathbf{H}_{x_1} & = \\frac{1}{12}
    \\begin{bmatrix}
        -1 & -4 & -1 \\\\
         0 & 0 & 0 \\\\
         1 & 4 & 1
    \\end{bmatrix}
&
\\mathbf{H}_{x_2} & = \\frac{1}{12}
    \\begin{bmatrix}
        -1 & 0 & 1 \\\\
        -4 & 0 & 4 \\\\
        -1 & 0 & 1
    \\end{bmatrix} \\\\
& = \\frac{1}{12}
    \\begin{bmatrix}
        -1 \\\\
        0  \\\\
        1
    \\end{bmatrix}
    \\begin{bmatrix}
        1 & 4 & 1
    \\end{bmatrix}
&
&  = \\frac{1}{12}
   \\begin{bmatrix}
        1 \\\\
        4  \\\\
        1
   \\end{bmatrix}
   \\begin{bmatrix}
        -1 & 0 & 1
   \\end{bmatrix}.
\\end{aligned}
```
See also:  [`KernelFactors.bickley`](@ref) and [`Kernel.bickley`](@ref)

## Choices for `border`
At the image edge, `border` is used to specify the padding which will be used
to extrapolate the image beyond its original bounds. As an indicative example
of each option the results of the padding are illustrated on an image consisting of
a row of six pixels which are specified alphabetically: ``\\boxed{a \\, b \\, c \\, d \\, e \\, f}``.
We show the effects of padding only on the left and right border, but analogous
consequences hold for the top and bottom border.

### `"replicate"`

The border pixels extend beyond the image boundaries.
```math
\\boxed{
\\begin{array}{l|c|r}
  a\\, a\\, a\\, a  &  a \\, b \\, c \\, d \\, e \\, f & f \\, f \\, f \\, f
\\end{array}
}
```
See also: [`Pad`](@ref), [`padarray`](@ref), [`Inner`](@ref) and
[`NoPad`](@ref)

### `"circular"`

The border pixels wrap around. For instance, indexing beyond the left border
returns values starting from the right border.
```math
\\boxed{
\\begin{array}{l|c|r}
  c\\, d\\, e\\, f  &  a \\, b \\, c \\, d \\, e \\, f & a \\, b \\, c \\, d
\\end{array}
}
```
See also: [`Pad`](@ref), [`padarray`](@ref), [`Inner`](@ref) and
[`NoPad`](@ref)

### `"symmetric"`
The border pixels reflect relative to a position between pixels. That is, the
border pixel is omitted when mirroring.
```math
\\boxed{
\\begin{array}{l|c|r}
  e\\, d\\, c\\, b  &  a \\, b \\, c \\, d \\, e \\, f & e \\, d \\, c \\, b
\\end{array}
}
```
See also: [`Pad`](@ref), [`padarray`](@ref), [`Inner`](@ref) and
[`NoPad`](@ref)

### `"reflect"`
The border pixels reflect relative to the edge itself.
```math
\\boxed{
\\begin{array}{l|c|r}
  d\\, c\\, b\\, a  &  a \\, b \\, c \\, d \\, e \\, f & f \\, e \\, d \\, c
\\end{array}
}
```
See also: [`Pad`](@ref), [`padarray`](@ref), [`Inner`](@ref) and
[`NoPad`](@ref)

# Example

This example compares the quality of the gradient estimation methods in terms of
the accuracy with which the orientation of the gradient is estimated.

```julia
using Images

values = linspace(-1,1,128);
w = 1.6*pi;

# Define a function of a sinusoidal grating, f(x,y) = sin( (w*x)^2 + (w*y)^2 ),
# together with its exact partial derivatives.
I = [sin( (w*x)^2 + (w*y)^2 ) for y in values, x in values];
Ix = [2*w*x*cos( (w*x)^2 + (w*y)^2 ) for y in values, x in values];
Iy = [2*w*y*cos( (w*x)^2 + (w*y)^2 ) for y in values, x in values];

# Determine the exact orientation of the gradients.
direction_true = atan.(Iy./Ix);

for kernelfunc in (KernelFactors.prewitt, KernelFactors.sobel,
                   KernelFactors.ando3, KernelFactors.scharr,
                   KernelFactors.bickley)

    # Estimate the gradients and their orientations.
    Gy, Gx = imgradients(I,kernelfunc, "replicate");
    direction_estimated = atan.(Gy./Gx);

    # Determine the mean absolute deviation between the estimated and true
    # orientation. Ignore the values at the border since we expect them to be
    # erroneous.
    error = mean(abs.(direction_true[2:end-1,2:end-1] -
                     direction_estimated[2:end-1,2:end-1]));

    error = round(error,5);
    println("Using \$kernelfunc results in a mean absolute deviation of \$error")
end

# output

Using ImageFiltering.KernelFactors.prewitt results in a mean absolute deviation of 0.01069
Using ImageFiltering.KernelFactors.sobel results in a mean absolute deviation of 0.00522
Using ImageFiltering.KernelFactors.ando3 results in a mean absolute deviation of 0.00365
Using ImageFiltering.KernelFactors.scharr results in a mean absolute deviation of 0.00126
Using ImageFiltering.KernelFactors.bickley results in a mean absolute deviation of 0.00038
```
# References
  1. B. Jahne, *Digital Image Processing* (5th ed.). Springer Publishing Company, Incorporated, 2005. [10.1007/3-540-27563-0](http://dx.doi.org/10.1007/3-540-27563-0)
  2. M. Patra  and  M. Karttunen, "Stencils with isotropic discretization error for differential operators," *Numer. Methods Partial Differential Eq.*, vol. 22, pp. 936–953, 2006. [doi:10.1002/num.20129](http://dx.doi.org/doi:10.1002/num.20129)
  3. J. M. Prewitt, "Object enhancement and extraction," *Picture processing and Psychopictorics*, vol. 10, no. 1, pp. 15–19, 1970.
  4. P.-E. Danielsson and O. Seger, "Generalized and separable sobel operators," in  *Machine Vision for Three-Dimensional Scenes*,  H. Freeman, Ed.  Academic Press, 1990,  pp. 347–379. [doi:10.1016/b978-0-12-266722-0.50016-6](http://dx.doi.org/doi:10.1016/b978-0-12-266722-0.50016-6)
  5. S. Ando, "Consistent gradient operators," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 22, no.3, pp. 252–265, 2000. [doi:10.1109/34.841757](http://dx.doi.org/doi:10.1109/34.841757)
  6. H. Scharr and  J. Weickert, "An anisotropic diffusion algorithm with optimized rotation invariance," *Mustererkennung 2000*, pp. 460–467, 2000. [doi:10.1007/978-3-642-59802-9_58](http://dx.doi.org/doi:10.1007/978-3-642-59802-9_58)
  7. A. Belyaev, "Implicit image differentiation and filtering with applications to image sharpening," *SIAM Journal on Imaging Sciences*, vol. 6, no. 1, pp. 660–679, 2013. [doi:10.1137/12087092x](http://dx.doi.org/doi:10.1137/12087092x)
  8. W. G. Bickley, "Finite difference formulae for the square lattice," *The Quarterly Journal of Mechanics and Applied Mathematics*, vol. 1, no. 1, pp. 35–42, 1948.  [doi:10.1093/qjmam/1.1.35](http://dx.doi.org/doi:10.1093/qjmam/1.1.35)

***

"""
function imgradients(img::AbstractArray, kernelfun::Function, border="replicate")
    extended = map(isextended, axes(img))
    _imgradients(extended, img, kernelfun, extended, border)
end

isextended(ind) = length(ind) > 1

# Add the next dimension to G
function _imgradients(donewhenempty::NTuple{M}, img::AbstractArray{T,N}, kernelfun::Function, extended, border) where {T,M,N}
    d = N-M+1  # the dimension we're working on now
    kern = kernelfun(extended, d)
    return (imfilter(img, kern, border), _imgradients(Base.tail(donewhenempty), img, kernelfun, extended, border)...)
end
# When all N gradients have been calculated, return the result
_imgradients(::Tuple{}, img::AbstractArray, kernelfun::Function, extent, border) = ()
