# ---
# title: Median filters
# cover: assets/median.gif
# author: Johnny Chen
# date: 2020-09-23
# ---

# With `median` filter as an example, this demo shows how you could construct a custom kernal and
# pass it to [`mapwindow`](@ref) for common stencil operations.

# `ImageFiltering` does not provide you an out-of-box median filter (e.g., `medfilt2` in MATLAB).
# This is because with [`mapwindow`](@ref) function, it is quite trivial a process to build and tailor one
# such for your own usage.

using ImageFiltering, ImageCore, ImageShow # or you could just `using Images`
using TestImages
using Statistics

img = float.(reshape(1:25, 5, 5))
patch_size = (3, 3)
imgm = mapwindow(median, img, patch_size)

# `mapwindow` provides a high-level interface to loop over the image and call a function (`median`
# in this demo) on each patch/window. That said, it does the following loop in a more efficient way.

## For simplicity, border condition is not included here.
imgm = zeros(axes(img))
R = CartesianIndices(img)
I_first, I_last = first(R), last(R)
Δ = CartesianIndex(patch_size .÷ 2)
for I in R
    patch = max(I_first, I-Δ):min(I_last, I+Δ)
    imgm[I] = median(img[patch])
end
imgm

# As you can see, except for the borders, a hand-written loop works basically the same as `mapwindow`.

# When the input array has `NaN` or some unwanted pixel values, a pre-filtering process is needed to
# exclude them first. This can be done quite easily by compositing a given kernel operation and a
# `filter` operation. Let's still take `median` filter as an example.

img[2,2] = NaN
imgm = mapwindow(median, img, (3,3))

# `NaN` has polluted the output, which is usually not wanted. This can be fixed quite easily by
# compositing a new filter kernel. This way, we only compute the median value on the non-NaN subset
# of each patch/window.

_median(patch) = median(filter(x->!isnan(x), patch))
imgm = mapwindow(_median, img, patch_size)


# ## Impulse (Salt and Pepper) noise removal

# Median filters are quite robust to outliers (unusual values), this property can be used to remove
# salt&pepper noise. An image with `n`-level salt&papper noise is defined as: each pixel `p` has
# `n/2` probabilty to be filled by `0`(the minimal gamut value) and `n/2` probabilty to be filled by
# `1`(the maximal gamut value).

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


## save covers #src
using ImageMagick #src
mkpath("assets") #src
ImageMagick.save("assets/median.gif", cat(noisy_img, denoised_img; dims=3); fps=1) #src
