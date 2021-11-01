# ---
# title: Log Gabor filter
# id: demo_log_gabor_filter
# cover: assets/log_gabor.png
# author: Johnny Chen
# date: 2021-11-01
# ---

# This example shows how one can apply frequency space kernesl [`LogGabor`](@ref
# Kernel.LogGabor) and [`LogGaborComplex`](@ref Kernel.LogGaborComplex) using fourier
# transformation and convolution theorem to extract image features. A similar example
# is the sptaial space kernel [Gabor filter](@ref demo_gabor_filter).

using ImageCore, ImageShow, ImageFiltering # or you could just `using Images`
using FFTW
using TestImages

# ## Definition
#
# Mathematically, log gabor filter is defined in spatial space as the composition of
# its frequency component `r` and angular component `a`:
#
# ```math
# r(\omega, \theta) = \exp(-\frac{(\log(\omega/\omega_0))^2}{2\sigma_\omega^2}) \\
# a(\omega, \theta) = \exp(-\frac{(\theta - \theta_0)^2}{2\sigma_\theta^2})
# ```
#
# `LogGaborComplex` provides a complex-valued matrix with value `Complex(r, a)`, while
# `LogGabor` provides real-valued matrix with value `r * a`.

kern_c = Kernel.LogGaborComplex((10, 10), 1/6, 0)
kern_r = Kernel.LogGabor((10, 10), 1/6, 0)
kern_r == @. real(kern_c) * imag(kern_c)

# !!! note "Lazy array"
#     The `LogGabor` and `LogGaborComplex` types are lazy arrays, which means when you build
#     the Log Gabor kernel, you actually don't need to allocate any memories. The computation
#     does not happen until you request the value.
#
#     ```julia
#     using BenchmarkTools
#     kern = @btime Kernel.LogGabor((64, 64), 1/6, 0); # 1.711 ns (0 allocations: 0 bytes)
#     @btime collect($kern); # 146.260 μs (2 allocations: 64.05 KiB)
#     ```


# To explain the parameters of Log Gabor filter, let's introduce small helper functions to
# display complex-valued kernels.
show_phase(kern) = @. Gray(log(abs(imag(kern)) + 1))
show_mag(kern) = @. Gray(log(abs(real(kern)) + 1))
show_abs(kern) = @. Gray(log(abs(kern) + 1))
nothing #hide

# From left to right are visualization of the kernel in frequency space: frequency `r`,
# algular `a`, `sqrt(r^2 + a^2)`, `r * a`, and its spatial space kernel.
kern = Kernel.LogGaborComplex((32, 32), 100, 0)
mosaic(
    show_mag(kern),
    show_phase(kern),
    show_abs(kern),
    Gray.(Kernel.LogGabor(kern)),
    show_abs(centered(ifftshift(ifft(kern)))),
    nrow=1
)

# ## Examples
#
# Because the filter is defined on frequency space, we can use [the convolution
# theorem](https://en.wikipedia.org/wiki/Convolution_theorem):
#
# ```math
# \mathcal{F}(x \circledast k) = \mathcal{F}(x) \odot \mathcal{F}(k)
# ```
# where ``\circledast`` is convolution, ``\odot`` is pointwise-multiplication, and
# ``\mathcal{F}`` is the fourier transformation.
#
# Also, since Log Gabor kernel is defined around center point (0, 0), we have to apply
# `ifftshift` first before we do pointwise-multiplication.

img = TestImages.shepp_logan(127)
kern = Kernel.LogGaborComplex(size(img), 50, π/4)
## we don't need to call `fft(kern)` here because it's already on frequency space
out = ifft(centered(fft(channelview(img))) .* ifftshift(kern))
mosaic(img, show_abs(kern), show_mag(out); nrow=1)

# A filter bank is just a list of filter kernels, applying the filter bank generates
# multiple outputs:

X_freq = centered(fft(channelview(img)))
filters = vcat(
    [Kernel.LogGaborComplex(size(img), 50, θ) for θ in -π/2:π/4:π/2],
    [Kernel.LogGabor(size(img), 50, θ) for θ in -π/2:π/4:π/2]
)
out = map(filters) do kern
    ifft(X_freq .* ifftshift(kern))
end
mosaic(
    map(show_abs, filters)...,
    map(show_abs, out)...;
    nrow=4, rowmajor=true
)

## save covers #src
using FileIO #src
mkpath("assets")  #src
filters = [Kernel.LogGaborComplex((32, 32), 5, θ) for θ in range(-π/2, stop=π/2, length=9)] #src
save("assets/log_gabor.png", mosaic(map(show_abs, filters); nrow=3, npad=2, fillvalue=Gray(1)); fps=2) #src
