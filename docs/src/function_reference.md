# Filtering functions

```@docs
imfilter
imfilter!
imgradients
mapwindow
mapwindow!
```

# Kernel

```@docs
Kernel
Kernel.sobel
Kernel.prewitt
Kernel.ando3
Kernel.ando4
Kernel.ando5
Kernel.bickley
Kernel.scharr
Kernel.gaussian
Kernel.DoG
Kernel.LoG
Kernel.Laplacian
Kernel.Gabor
Kernel.moffat
```

# KernelFactors

```@docs
KernelFactors
KernelFactors.sobel
KernelFactors.prewitt
KernelFactors.bickley
KernelFactors.scharr
KernelFactors.ando3
KernelFactors.ando4
KernelFactors.ando5
KernelFactors.gaussian
KernelFactors.IIRGaussian
KernelFactors.TriggsSdika
```

# Kernel utilities

```@docs
OffsetArrays.center
OffsetArrays.centered
kernelfactors
reflect
```

# Boundaries and padding

```@docs
padarray
BorderArray
Pad
Fill
Inner
NA
NoPad
```

# Algorithms

```@docs
Algorithm.FIR
Algorithm.FFT
Algorithm.IIR
Algorithm.Mixed
```

# Solvers for predefined models

```@autodocs
Modules = [ImageFiltering.Models]
```

# Internal machinery

```@docs
KernelFactors.ReshapedOneD
```
