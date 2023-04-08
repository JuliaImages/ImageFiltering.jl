# Kernels

`Kernel` is a module implementing filtering (correlation) kernels of full
dimensionality.

## Supported kernels

The following kernels are supported:

  - `sobel`
  - `prewitt`
  - `ando3`, `ando4`, and `ando5`
  - `scharr`
  - `bickley`
  - `gaussian`
  - `DoG` (Difference-of-Gaussian)
  - `LoG` (Laplacian-of-Gaussian)
  - `Laplacian`
  - `gabor`
  - `moffat`

`KernelFactors` is a module implementing separable filtering kernels,
each stored in terms of their factors. The following kernels are
supported:

  - `box`
  - `sobel`
  - `prewitt`
  - `ando3`, `ando4`, and `ando5` (the latter in 2d only)
  - `scharr`
  - `bickley`
  - `gaussian`
  - `IIRGaussian` (approximate gaussian filtering, fast even for large σ)

## Correlation, not convolution

The ImageFiltering package uses the following formula to calculate the filtered
image `F` from an input image `A` and kernel `K`:

```math
F[I] = \sum_J A[I+J] K[J]
```

Consequently, the resulting image is the correlation, not convolution,
of the input and the kernel. If you want the convolution, first call
[`reflect`](@ref) on the kernel.

## Kernel indices

ImageFiltering exploits Julia's ability to define arrays whose indices span an arbitrary range:

```julia
julia> Kernel.gaussian(1)
OffsetArray{Float64,2,Array{Float64,2}} with indices -2:2×-2:2:
 0.00296902  0.0133062  0.0219382  0.0133062  0.00296902
 0.0133062   0.0596343  0.0983203  0.0596343  0.0133062
 0.0219382   0.0983203  0.162103   0.0983203  0.0219382
 0.0133062   0.0596343  0.0983203  0.0596343  0.0133062
 0.00296902  0.0133062  0.0219382  0.0133062  0.00296902
```

The indices of this array span the range `-2:2` along each axis, and
the center of the gaussian is at position `[0,0]`. As a consequence,
this filter "blurs" but does not "shift" the image; were the center
instead at, say, `[3,3]`, the filtered image would be shifted by 3
pixels downward and to the right compared to the original.

The `centered` function is a handy utility for converting an ordinary
array to one that has coordinates `[0,0,...]` at its center position:

```julia
julia> centered([1 0 1; 0 1 0; 1 0 1])
OffsetArray{Int64,2,Array{Int64,2}} with indices -1:1×-1:1:
 1  0  1
 0  1  0
 1  0  1
```

See [OffsetArrays](https://github.com/alsam/OffsetArrays.jl) for more information.

## Factored kernels

A key feature of Gaussian kernels---along with many other
commonly-used kernels---is that they are *separable*, meaning that
`K[j_1,j_2,...]` can be written as ``K_1[j_1] K_2[j_2] \cdots``.
As a consequence, the correlation:

```math
F[i_1,i_2] = \sum_{j_1,j_2} A[i_1+j_1,i_2+j_2] K[j_1,j_2]
```

can be written:

```math
F[i_1,i_2] = \sum_{j_2} \left(\sum_{j_1} A[i_1+j_1,i_2+j_2] K_1[j_1]\right) K_2[j_2]
```

If the kernel is of size `m×n`, then the upper version line requires `mn`
operations for each point of `filtered`, whereas the lower version
requires `m+n` operations. Especially when `m` and `n` are larger,
this can result in a substantial savings.

To enable efficient computation for separable kernels, `imfilter`
accepts a tuple of kernels, filtering the image by each
sequentially. You can either supply `m×1` and `1×n` filters directly,
or (somewhat more efficiently) call [`kernelfactors`](@ref) on a
tuple-of-vectors:

```julia
julia> kern1 = centered([1/3, 1/3, 1/3])
OffsetArray{Float64,1,Array{Float64,1}} with indices -1:1:
 0.333333
 0.333333
 0.333333

julia> kernf = kernelfactors((kern1, kern1))
(ImageFiltering.KernelFactors.ReshapedOneD{Float64,2,0,OffsetArray{Float64,1,Array{Float64,1}}}([0.333333,0.333333,0.333333]),ImageFiltering.KernelFactors.ReshapedOneD{Float64,2,1,OffsetArray{Float64,1,Array{Float64,1}}}([0.333333,0.333333,0.333333]))

julia> kernp = broadcast(*, kernf...)
OffsetArray{Float64,2,Array{Float64,2}} with indices -1:1×-1:1:
 0.111111  0.111111  0.111111
 0.111111  0.111111  0.111111
 0.111111  0.111111  0.111111

julia> imfilter(img, kernf) ≈ imfilter(img, kernp)
true
```

If the kernel is a two dimensional array, `imfilter` will attempt to
factor it; if successful, it will use the separable algorithm. You can
prevent this automatic factorization by passing the kernel as a tuple,
e.g., as `(kernp,)`.

## Popular kernels in Kernel and KernelFactors modules

The two modules [`Kernel`](@ref) and [`KernelFactors`](@ref) implement popular
correlation kernels in "dense" and "factored" forms, respectively. Type
`?Kernel` or `?KernelFactors` at the REPL to see which kernels are supported.

A common task in image processing and computer vision is computing
image *gradients* (derivatives), for which there is the dedicated
function [`imgradients`](@ref).
