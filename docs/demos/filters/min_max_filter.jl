# ---
# title: Max min filters
# cover: assets/max_min.gif
# author: Deeptendu Santra
# date: 2020-10-08
# ---

# In this tutorial we see how can we can effectively use max and min filter to distinguish
# between smooth and texture edges in grayscale images. The example in this demo comes from [1].

# We will use the [`mapwindow`](@ref) function in `ImageFiltering.jl` which provides a general
# functionality to apply any function to the window around each pixel. 
# [Custom median filters](@ref median_filter_example) is another usage example of `mapwindow`.

using ImageCore, ImageShow, ImageFiltering
using MappedArrays
using TestImages

img = Gray.(testimage("house")); # Original Image

# We need four statistics for our demo, they are: the local minimum, maximum, min-max, max-min of
# the input image. We can use `mapwindow` to get them.

window_size = (15, 15)
img_min = mapwindow(minimum, img, window_size)
img_max = mapwindow(maximum, img, window_size)
img_max_min = mapwindow(maximum, img_min, window_size)
img_min_max = mapwindow(minimum, img_max, window_size)
mosaicview(img_min, img_max, img_max_min, img_min_max; nrow=1)

# When `f` is one of `maximum`, `minimum` and `maximum`, `mapwindow(f, img, window_size)` will use
# the a streaming version Lemire max-min filter[2] to do the filtering work. This is more efficient
# than a plain maximum implementation.

# ```julia
# using BenchmarkTools
# f(x) = minimum(x) # do note that `f !== minimum`
# @btime mapwindow(f, $img, window_size) # 47.508 ms (202831 allocations: 12.91 MiB)
# @btime mapwindow(minimum, $img, window_size) # 13.216 ms (58 allocations: 1.75 MiB)
# ```

# Also, in this example, since we need both the local minimum and maximum image. We could use
# `extrema` to reduce the computation, and use `MappedArrays.mappedarray` to reduce allocation.
# This is more efficient than repeatedly doing `minimum` and `maximum`.

img_extrema = mapwindow(extrema, img, window_size) # only compute once
img_min = mappedarray(first, img_extrema) # 0 allocation
img_max = mappedarray(last, img_extrema) # 0 allocation
#md nothing #hide

# Now that we are done with the basic filtered images, we proceed to the next part
# which is edge detection using these filters.

# For edge detection, we need to define thresholds for our image. The threshold is an important tool
# to binarize a grayscale image. The threshold value for a given pixel practically
# decides if the pixel visible or not in the output image. However, appyling a global threshold might not consider the
# variation of colors/brightness within the image. Thus we consider an adaptive type theresholding method
# here using max/min results.

# The max(min) and min(max) filter effectively follows the smooth edges of the image. Therefore,
# their average also follows the smooth parts of the image. If we use this image as a threshold for the original
# image, the smooth parts of the original image will get filtered out, leaving only the texture and/or noise behind.
# So we can use the average of `img_max_min` and `img_min_max` as the texture threshold.

# The average of min and max filters also gives us the smooth edges, but it also includes the noise in the image.
# So using average of `img_min` and `img_max` as a threshold will yeild only texture.

img_texture_noise_threshold = (img_max_min + img_max_min) ./ 2
img_texture_threshold = (img_max + img_min) ./ 2
mosaicview(img_texture_noise_threshold, img_texture_threshold; nrow=1)
#
## The Dynamic Gist is obtained by subtracting the img_texture_threshold from the original image.
## The filtered image gives us the texture of the image.
img_dynamic_gist = img - img_texture_threshold
## The Texture Gist is obtained by subtracting img_texture_noise_threshold from the original image.
## The filtered image gives us the texture along with the noise of the image.
img_texture_gist = img - img_texture_noise_threshold
mosaicview(img_dynamic_gist, img_texture_gist; nrow=1)
#
## We extract out the smooth/ramp parts of the image.
## The darker section of the image consist of the ramp edges. The brighter pixels are mostly noise.
ramp = img_dynamic_gist - img_texture_gist
## Filtered-out edges
edge = img_max - img_min
## Smoothed-out version of edge
edge_smoothed = img_min_max - img_max_min
mosaicview(img, ramp, edge, edge_smoothed; nrow=2)

# # References

# [1] Verbeek, P. W., Vrooman, H. A., & Van Vliet, L. J. (1988). [Low-level image processing by max-min filters](https://core.ac.uk/download/pdf/194053536.pdf). Signal Processing, 15(3), 249-258.

# [2] Lemire, D. (2006). [Streaming maximum-minimum filter using no more than three comparisons per element](https://lemire.me/en/publication/arxiv0610046/). arXiv preprint cs/0610046.

## save covers #src
using ImageMagick #src
mkpath("assets")  #src
ImageMagick.save("assets/max_min.gif", cat(img, edge, edge_smoothed, map(clamp01nan, ramp);dims=3);fps=1) #src
