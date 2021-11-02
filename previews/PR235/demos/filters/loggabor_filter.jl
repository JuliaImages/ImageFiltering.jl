using ImageCore, ImageShow, ImageFiltering # or you could just `using Images`
using FFTW
using TestImages

kern_c = Kernel.LogGaborComplex((10, 10), 1/6, 0)
kern_r = Kernel.LogGabor((10, 10), 1/6, 0)
kern_r == @. real(kern_c) * imag(kern_c)

show_phase(kern) = @. Gray(log(abs(imag(kern)) + 1))
show_mag(kern) = @. Gray(log(abs(real(kern)) + 1))
show_abs(kern) = @. Gray(log(abs(kern) + 1))
nothing #hide

kern = Kernel.LogGaborComplex((32, 32), 100, 0)
mosaic(
    show_mag(kern),
    show_phase(kern),
    show_abs(kern),
    Gray.(Kernel.LogGabor(kern)),
    show_abs(centered(ifftshift(ifft(kern)))),
    nrow=1
)

img = TestImages.shepp_logan(127)
kern = Kernel.LogGaborComplex(size(img), 50, π/4)
# we don't need to call `fft(kern)` here because it's already on frequency space
out = ifft(centered(fft(channelview(img))) .* ifftshift(kern))
mosaic(img, show_abs(kern), show_mag(out); nrow=1)

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

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

