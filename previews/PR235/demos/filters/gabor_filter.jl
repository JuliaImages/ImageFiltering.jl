using ImageCore, ImageShow, ImageFiltering # or you could just `using Images`
using FFTW
using TestImages

kern = Kernel.Gabor((10, 10), 2, 0.1)

# You can also try display the real part: `@. Gray(log(abs(real(kern)) + 1))`
show_phase(kern) = @. Gray(log(abs(imag(kern)) + 1))
show_mag(kern) = @. Gray(log(abs(real(kern)) + 1))
show_abs(kern) = @. Gray(log(abs(kern) + 1))
nothing #hide

bandwidth, orientation, phase_offset, aspect_ratio = 1, 0, 0, 0.5
f(wavelength) = show_abs(Kernel.Gabor((100, 100); wavelength, bandwidth, orientation, aspect_ratio, phase_offset))
mosaic(f.((5, 10, 15)), nrow=1)

wavelength, bandwidth, phase_offset, aspect_ratio = 10, 1, 0, 0.5
f(orientation) = show_abs(Kernel.Gabor((100, 100); wavelength, bandwidth, orientation, aspect_ratio, phase_offset))
mosaic(f.((0, π/4, π/2)), nrow=1)

wavelength, bandwidth, orientation, aspect_ratio = 10, 1, 0, 0.5
f(phase_offset) = show_phase(Kernel.Gabor((100, 100); wavelength, bandwidth, orientation, aspect_ratio, phase_offset))
mosaic(f.((-π/2, 0, π/2, π)), nrow=1)

wavelength, bandwidth, orientation, phase_offset = 10, 1, 0, 0
f(aspect_ratio) = show_abs(Kernel.Gabor((100, 100); wavelength, bandwidth, orientation, aspect_ratio, phase_offset))
mosaic(f.((0.5, 1, 2)), nrow=1)

wavelength, orientation, phase_offset, aspect_ratio = 10, 0, 0, 0.5
f(bandwidth) = show_abs(Kernel.Gabor((100, 100); wavelength, bandwidth, orientation, aspect_ratio, phase_offset))
mosaic(f.((0.5, 1, 2)), nrow=1)

img = TestImages.shepp_logan(127)
kern = Kernel.Gabor(size(img); orientation=0, wavelength=3, bandwidth=2, phase_offset=0)
out = ifft(fft(channelview(img)) .* ifftshift(fft(kern)))
mosaic(img, show_abs(kern), show_mag(out); nrow=1)

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

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

