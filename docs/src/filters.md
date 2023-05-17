# Filtering images

The `imfilter()` function filters a one, two or multidimensional array `img` with a `kernel` by computing their correlation.

## Introduction

The term *filtering* emerges in the context of a Fourier transformation of an
image, which maps an image from its canonical spatial domain to its concomitant
frequency domain. Manipulating an image in the frequency domain amounts to
retaining or discarding particular frequency components —- a process analogous to
sifting or filtering [1]. Because the Fourier transform establishes a link
between the spatial and frequency representation of an image, one can interpret
various image manipulations in the spatial domain as filtering operations which
accept or reject specific frequencies.

The phrase *spatial filtering* is often used to emphasize that an operation is,
at least conceptually, devised in the context of the spatial domain of an image.
We further distinguish between linear and non-linear spatial filtering. A
filter is called *linear* if the operation performed on the pixels is linear,
and is labeled non-linear otherwise.

## Function options

The syntax for `imfilter()` is as follows:

```julia
imfilter([T], img, kernel, [border="replicate"], [alg])
imfilter([r], img, kernel, [border="replicate"], [alg])
imfilter(r, T, img, kernel, [border="replicate"], [alg])
```

The following subsections describe valid options for the function arguments in
more detail.

### Choices for `r`

Optionally, you can dispatch to different implementations by passing in a resource `r`
as defined by the [ComputationalResources](https://github.com/timholy/ComputationalResources.jl) package.

For example:

```julia
imfilter(ArrayFireLibs(), img, kernel)
```

would request that the computation be performed on the GPU using the
ArrayFire libraries.

### Choices for `T`

Optionally, you can control the element type of the output image by
passing in a type `T` as the first argument.

### Choices for `img`

You must provide a one, two, or multidimensional array as the input image.

### Choices for `kernel`

The `kernel[0, 0,..]` parameter corresponds to the origin (zero displacement) of
the kernel; you can use `centered` to place the origin at the array center, or
use the OffsetArrays package to set `kernel`'s indices manually. For example, to
filter with a random *centered* 3x3 kernel, you could use either of the
following:

```julia
kernel = centered(rand(3,3))
kernel = OffsetArray(rand(3,3), -1:1, -1:1)
```

The `kernel` parameter can be specified as an array or as a "factored kernel", a
tuple `(filt1, filt2, ...)` of filters to apply along each axis of the image. In
cases where you know your kernel is separable, this format can speed processing.
Each of these should have the same dimensionality as the image itself, and be
shaped in a manner that indicates the filtering axis, e.g., a 3x1 filter for
filtering the first dimension and a 1x3 filter for filtering the second
dimension. In two dimensions, any kernel passed as a single matrix is checked
for separability; if you want to eliminate that check, pass the kernel as a
single-element tuple, `(kernel,)`.

### Choices for `border`

At the image edge, `border` is used to specify the padding which will be used
to extrapolate the image beyond its original bounds. 

As an indicative example of each option, the results of the padding are
illustrated on an image consisting of a row of six pixels which are specified
alphabetically: 

```plain
        ┏━━━━━━┓ 
        ┃abcdef┃ 
        ┗━━━━━━┛ 
```

We show the effects of
padding only on the left and right border, but analogous consequences hold for
the top and bottom border.

#### `"replicate"` (default)

The border pixels extend beyond the image boundaries.

```plain
   ╭────┏━━━━━━┓────╮
   │aaaa┃abcdef┃ffff│
   ╰────┗━━━━━━┛────╯
```

#### `"circular"`

The border pixels wrap around. For instance, indexing beyond the left border
returns values starting from the right border.

```plain

   ╭────┏━━━━━━┓────╮
   │cdef┃abcdef┃abcd│
   ╰────┗━━━━━━┛────╯

```

#### `"reflect"`

The border pixels reflect relative to a position between pixels. That is, the
border pixel is omitted when mirroring.

```plain

   ╭────┏━━━━━━┓────╮
   │dcba┃abcdef┃fedc│
   ╰────┗━━━━━━┛────╯

```

#### `"symmetric"`

The border pixels reflect relative to the edge itself.

```plain

   ╭────┏━━━━━━┓────╮
   │edcb┃abcdef┃edcb│
   ╰────┗━━━━━━┛────╯

```

#### `Fill(m)`

The border pixels are filled with a specified value ``m``.

```plain

   ╭────┏━━━━━━┓────╮
   │mmmm┃abcdef┃mmmm│
   ╰────┗━━━━━━┛────╯

```

#### `Inner()`

Indicate that edges are to be discarded in filtering, only the interior of the
result is to be returned.

#### `NA()`

Choose filtering using "NA" (Not Available) boundary conditions. This
is most appropriate for filters that have only positive weights summing to 1,
such as blurring filters---rather than "make up" values beyond the edges,
the result is normalized by the number of in-bounds pixels (similar to [`nanmean`](https://brenhinkeller.github.io/NaNStatistics.jl/dev/#NaNStatistics.nanmean-Tuple{Any})).

See also: [`Pad`](@ref), [`padarray`](@ref), [`Inner`](@ref), [`NA`](@ref)  and
[`NoPad`](@ref)

### Choices for `alg`

The `alg` parameter allows you to choose the particular algorithm: `Algorithm.FIR()`
(finite impulse response, aka traditional digital filtering) or `Algorithm.FFT()`
(Fourier-based filtering). If no choice is specified, one will be chosen based
on the size of the image and kernel in a way that strives to deliver good
performance. Alternatively you can use a custom filter type, like
[`KernelFactors.IIRGaussian`](@ref).

## Convolution versus correlation

The default operation of `imfilter` is correlation.  By reflecting `w` we
compute the convolution of `f` and `w`. `Fill(0,w)` indicates that we wish to
pad the border of `f` with zeros. The amount of padding is automatically
determined by considering the length of w.

```julia
# Create a two-dimensional discrete unit impulse function.
f = fill(0,(9,9));
f[5,5] = 1;

# Specify a filter coefficient mask and set the center of the mask as the origin.
w = centered([1 2 3; 4 5 6 ; 7 8 9]);

correlation = imfilter(f,w,Fill(0,w))
convolution = imfilter(f,reflect(w),Fill(0,w))
```

## Miscellaneous border padding options

Given the following example values:

```julia
f = reshape(1.0:81.0, 9, 9)
w = centered(reshape(1.0:9.0, 3, 3))
```

you can designate the type of padding by supplying an appropriate string:

```julia
imfilter(f, w, "replicate")
imfilter(f, w, "circular")
imfilter(f, w, "symmetric")
imfilter(f, w, "reflect")
```

Alternatively, you can explicitly use the `Pad` type to designate the padding style:

```julia
imfilter(f, w, Pad(:replicate))
imfilter(f, w, Pad(:circular))
imfilter(f, w, Pad(:symmetric))
imfilter(f, w, Pad(:reflect))
```

If you want to pad with a specific value, use the `Fill` type.

```julia
imfilter(f, w, Fill(0, w))
imfilter(f, w, Fill(1, w))
imfilter(f, w, Fill(-1, w))
```

Specify `Inner()` if you want to retrieve the interior sub-array of f for which
the filtering operation is defined without padding:

```julia
imfilter(f, w, Inner())
```

## The `imfilter!` function

The `imfilter!()` function filters an array `img` with kernel `kernel` by computing their
correlation, storing the result in `imgfilt`.

```julia
imfilter!(imgfilt, img, kernel, [border="replicate"], [alg])
imfilter!(r, imgfilt, img, kernel, border::Pad)
imfilter!(r, imgfilt, img, kernel, border::NoPad, [inds=axes(imgfilt)])
```

The indices of `imgfilt` determine the region over which the filtered
image is computed -- you can use this fact to select just a specific
region of interest, although be aware that the input `img` might still
get padded.

Alteratively, explicitly provide the indices `inds` of
`imgfilt` that you want to calculate, and use `NoPad` boundary
conditions. In such cases, you are responsible for supplying
appropriate padding: `img` must be indexable for all of the locations
needed for calculating the output. This syntax is best-supported for
FIR filtering; in particular, the IIR filtering can lead to
results that are inconsistent with respect to filtering the entire
array.

See also: [`imfilter`](@ref), [`centered`](@ref OffsetArrays.centered), [`padarray`](@ref), [`Pad`](@ref), [`Fill`](@ref), [`Inner`](@ref), [`KernelFactors.IIRGaussian`](@ref).

## References

1. R. C. Gonzalez and R. E. Woods. *Digital Image Processing (3rd Edition)*.  Upper Saddle River, NJ, USA: Prentice-Hall,  2006.
