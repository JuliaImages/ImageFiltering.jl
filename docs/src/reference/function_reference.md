# Filtering functions

## Functions

```@docs
imfilter
imfilter!
imgradients
mapwindow
mapwindow!
```

## Kernel

```@docs
Kernel
Kernel.ando3
Kernel.ando4
Kernel.ando5
Kernel.bickley
Kernel.DoG
Kernel.gabor
Kernel.gaussian
Kernel.Laplacian
Kernel.LoG
Kernel.moffat
Kernel.prewitt
Kernel.scharr
Kernel.sobel
```

## KernelFactors

```@docs
KernelFactors
KernelFactors.ando3
KernelFactors.ando4
KernelFactors.ando5
KernelFactors.bickley
KernelFactors.gaussian
KernelFactors.IIRGaussian
KernelFactors.prewitt
KernelFactors.scharr
KernelFactors.sobel
KernelFactors.TriggsSdika
```

## Kernel utilities

```@docs
OffsetArrays.center
OffsetArrays.centered
kernelfactors
reflect
```

## Boundaries and padding

```@docs
BorderArray
Fill
Inner
NA
NoPad
Pad
padarray
```

## Find local extrema

```@docs
findlocalmaxima
findlocalminima
```

## Algorithms

```@docs
Algorithm.FIR
Algorithm.FFT
Algorithm.IIR
Algorithm.Mixed
```

## Solvers for predefined models

```@autodocs
Modules = [ImageFiltering.Models]
```

## Internal machinery

```@docs
KernelFactors.ReshapedOneD
```
