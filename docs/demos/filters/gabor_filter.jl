# ---
# title: Gabor filter
# id: demo_gabor_filter
# cover: assets/gabor.png
# author: Johnny Chen
# date: 2021-11-01
# ---

# This example shows how one can apply spatial space kernesl [`Gabor`](@ref Kernel.Gabor)
# using fourier transformation and convolution theorem to extract image features. A similar
# example is [Log Gabor filter](@ref demo_log_gabor_filter).

using ImageCore, ImageShow, ImageFiltering # or you could just `using Images`
using FFTW
using TestImages

# ## Definition
#
# Mathematically, Gabor kernel is defined in spatial space:
#
# ```math
# g(x, y) = \exp(-\frac{x'^2 + \gamma^2y'^2}{2\sigma^2})\exp(i(2\pi\frac{x'}{\lambda} + \psi))
# ```
# where ``i`` is imaginary unit `Complex(0, 1)`, and
# ```math
# x' = x\cos\theta + x\sin\theta \\
# y' = -x\sin\theta + y\cos\theta
# ```
#

# First of all, Gabor kernel is a complex-valued matrix:

kern = Kernel.Gabor((10, 10), 2, 0.1)

# !!! tip "Lazy array"
#     The `Gabor` type is a lazy array, which means when you build the Gabor kernel, you
#     actually don't need to allocate any memories.
#
#     ```julia
#     using BenchmarkTools
#     kern = @btime Kernel.Gabor((64, 64), 5, 0); # 36.481 ns (0 allocations: 0 bytes)
#     @btime collect($kern); # 75.278 μs (2 allocations: 64.05 KiB)
#     ```

# To explain the parameters of Gabor filter, let's introduce some small helpers function to
# display complex-valued kernels.
show_phase(kern) = @. Gray(log(abs(imag(kern)) + 1))
show_mag(kern) = @. Gray(log(abs(real(kern)) + 1))
show_abs(kern) = @. Gray(log(abs(kern) + 1))
nothing #hide

# ## Keywords
#
# ### `wavelength` (λ)
# λ specifies the wavelength of the sinusoidal factor.

bandwidth, orientation, phase_offset, aspect_ratio = 1, 0, 0, 0.5
f(wavelength) = show_abs(Kernel.Gabor((100, 100); wavelength, bandwidth, orientation, aspect_ratio, phase_offset))
mosaic(f.((5, 10, 15)), nrow=1)

# ### `orientation` (θ)
# θ specifies the orientation of the normal to the parallel stripes of a Gabor function.

wavelength, bandwidth, phase_offset, aspect_ratio = 10, 1, 0, 0.5
f(orientation) = show_abs(Kernel.Gabor((100, 100); wavelength, bandwidth, orientation, aspect_ratio, phase_offset))
mosaic(f.((0, π/4, π/2)), nrow=1)

# ### `phase_offset` (ψ)

wavelength, bandwidth, orientation, aspect_ratio = 10, 1, 0, 0.5
f(phase_offset) = show_phase(Kernel.Gabor((100, 100); wavelength, bandwidth, orientation, aspect_ratio, phase_offset))
mosaic(f.((-π/2, 0, π/2, π)), nrow=1)

# ### `aspect_ratio` (γ)
# γ specifies the ellipticity of the support of the Gabor function.

wavelength, bandwidth, orientation, phase_offset = 10, 1, 0, 0
f(aspect_ratio) = show_abs(Kernel.Gabor((100, 100); wavelength, bandwidth, orientation, aspect_ratio, phase_offset))
mosaic(f.((0.5, 1, 2)), nrow=1)

# ### `bandwidth` (b)
# The half-response spatial frequency bandwidth (b) of a Gabor filter is related to the
# ratio σ / λ, where σ and λ are the standard deviation of the Gaussian factor of the Gabor
# function and the preferred wavelength, respectively, as follows:
#
# ```math
# b = \log_2\frac{\frac{\sigma}{\lambda}\pi + \sqrt{\frac{\ln 2}{2}}}{\frac{\sigma}{\lambda}\pi - \sqrt{\frac{\ln 2}{2}}}
# ```

wavelength, orientation, phase_offset, aspect_ratio = 10, 0, 0, 0.5
f(bandwidth) = show_abs(Kernel.Gabor((100, 100); wavelength, bandwidth, orientation, aspect_ratio, phase_offset))
mosaic(f.((0.5, 1, 2)), nrow=1)

# ## Examples
#
# There are two options to apply a spatial space kernel: 1) via `imfilter`, and 2) via the
# convolution theorem.

# ### `imfilter`
#
# [`imfilter`](@ref) does not require the kernel size to be the same as the image size.
# Usually kernel size is at least 5 times larger than the wavelength.

img = TestImages.shepp_logan(127)
kern = Kernel.Gabor((19, 19), 3, 0)
out = imfilter(img, real.(kern))
mosaic(img, show_abs(kern), show_mag(out); nrow=1)

# ### convolution theorem

# The [convolution theorem](https://en.wikipedia.org/wiki/Convolution_theorem) tells us
# that `fft(conv(X, K))` is equivalent to `fft(X) .* fft(K)`. Because Gabor kernel is
# defined with regard to its center point (0, 0), we need to do `ifftshift` first so that
# the frequency centers of both `fft(X)` and `fft(kern)` align well.

kern = Kernel.Gabor(size(img), 3, 0)
out = ifft(fft(channelview(img)) .* ifftshift(fft(kern)))
mosaic(img, show_abs(kern), show_mag(out); nrow=1)

# As you may have notice, using convolution theorem generates different results. This is
# simply because the kernel size are different. If we create a smaller kernel, we then need
# to apply [`freqkernel`](@ref) first so that we can do element-wise multiplication.

## freqkernel = zero padding + fftshift + fft
kern = Kernel.Gabor((19, 19), 3, 0)
kern_freq = freqkernel(real.(kern), size(img))
out = ifft(fft(channelview(img)) .* kern_freq)
mosaic(img, show_abs(kern), show_mag(out); nrow=1)

# !!! note "Performance on different kernel size"
#     When the kernel size is small, `imfilter` works more efficient than fft-based
#     convolution. This benchmark isn't backed by CI so the result might be outdated, but
#     you get the idea.
#
#     ```julia
#     using BenchmarkTools
#     img = TestImages.shepp_logan(127);
#
#     kern = Kernel.Gabor((19, 19), 3, 0);
#     fft_conv(img, kern) = ifft(fft(channelview(img)) .* freqkernel(real.(kern), size(img)))
#     @btime imfilter($img, real.($kern)); # 236.813 μs (118 allocations: 418.91 KiB)
#     @btime fft_conv($img, $kern) # 1.777 ms (127 allocations: 1.61 MiB)
#
#     kern = Kernel.Gabor(size(img), 3, 0)
#     fft_conv(img, kern) =  ifft(fft(channelview(img)) .* ifftshift(fft(kern)))
#     @btime imfilter($img, real.($kern)); # 5.318 ms (163 allocations: 5.28 MiB)
#     @btime fft_conv($img, $kern); # 2.218 ms (120 allocations: 1.73 MiB)
#     ```
#

## Filter bank

# A filter bank is just a list of filter kernels, applying the filter bank generates
# multiple outputs:

filters = [Kernel.Gabor(size(img), 3, θ) for θ in -π/2:π/4:π/2];
X_freq = fft(channelview(img))
out = map(filters) do kern
    ifft(X_freq .* ifftshift(fft(kern)))
end
mosaic(
    map(show_abs, filters)...,
    map(show_abs, out)...;
    nrow=2, rowmajor=true
)

## save covers #src
using FileIO #src
mkpath("assets")  #src
filters = [Kernel.Gabor((32, 32), 5, θ) for θ in range(-π/2, stop=π/2, length=9)] #src
save("assets/gabor.png", mosaic(map(show_abs, filters); nrow=3, npad=2, fillvalue=Gray(1)); fps=2) #src

# # References
#
# - [1] [Wikipedia: Gabor filter](https://en.wikipedia.org/wiki/Gabor_filter)
# - [2] [Wikipedia: Gabor transformation](https://en.wikipedia.org/wiki/Gabor_transform)
# - [3] [Wikipedia: Gabor atom](https://en.wikipedia.org/wiki/Gabor_atom)
# - [4] [Gabor filter for image processing and computer vision](http://matlabserver.cs.rug.nl/edgedetectionweb/web/edgedetection_params.html)
