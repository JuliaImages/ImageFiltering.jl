using ImageFiltering, ImageCore, ImageShow # or you could just `using Images`
using TestImages
using Statistics
using Random #hide
Random.seed!(0) #hide

img = Float64[isodd(i) - isodd(j) for i = 1:5, j = 1:5]
img[3, 3] = 1000

patch_size = (3, 3)
imgm = mapwindow(median, img, patch_size)

# For simplicity, border condition is not included here.
imgm = zeros(axes(img))
R = CartesianIndices(img)
I_first, I_last = first(R), last(R)
Δ = CartesianIndex(patch_size .÷ 2)
for I in R
    patch = max(I_first, I-Δ):min(I_last, I+Δ)
    imgm[I] = median(img[patch])
end
imgm

img[2,2] = NaN
imgm = mapwindow(median, img, (3,3))

_median(patch) = median(filter(x->!isnan(x), patch))
imgm = mapwindow(_median, img, patch_size)

img = testimage("cameraman")

n = 0.2 # noise level
noisy_img = map(img) do p
    sp = rand()
    if sp < n/2
        eltype(img)(gamutmin(eltype(img))...)
    elseif sp < n
        eltype(img)(gamutmax(eltype(img))...)
    else
        p
    end
end

denoised_img = mapwindow(median!, noisy_img, (5, 5))

mosaicview(img, noisy_img, denoised_img; nrow=1)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
