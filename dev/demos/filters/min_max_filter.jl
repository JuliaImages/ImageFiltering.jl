using ImageCore, ImageShow, ImageFiltering
using TestImages

img = Gray.(testimage("house");)      # Original Image

minimum([Gray(0.7),Gray(0.5),Gray(0.0)]) # Should return Gray(0.0) i.e black.

filter_size = (15, 15)
# Using the `mapwindow` function, we create an image of the local minimum.
# `mapwindow` maps the given function over a moving window of given size.
img_min = mapwindow(minimum, img, filter_size)
# Similarly for maximum
img_max = mapwindow(maximum, img, filter_size)
# The max(min) filter
img_max_min = mapwindow(maximum, img_min, filter_size)
# The min(max) filter
img_min_max = mapwindow(minimum, img_max, filter_size)
mosaicview(img_min, img_max, img_max_min, img_min_max; nrow=1)

img_texture_noise_threshold = (img_max_min + img_max_min) ./ 2
img_texture_threshold = (img_max + img_min) ./ 2
mosaicview(img_texture_noise_threshold, img_texture_threshold; nrow=1)

# The Dynamic Gist is obtained by subtracting the img_texture_threshold from the original image.
# The filtered image gives us the texture of the image.
img_dynamic_gist = img - img_texture_threshold
# The Texture Gist is obtained by subtracting img_texture_noise_threshold from the original image.
# The filtered image gives us the texture along with the noise of the image.
img_texture_gist = img - img_texture_noise_threshold
mosaicview(img_dynamic_gist, img_texture_gist; nrow=1)

# We extract out the smooth/ramp parts of the image.
# The darker section of the image consist of the ramp edges. The brighter pixels are mostly noise.
ramp = img_dynamic_gist - img_texture_gist
# Filtered-out edges
edge = img_max - img_min
# Smoothed-out version of edge
edge_smoothed = img_min_max - img_max_min
mosaicview(img, ramp, edge, edge_smoothed; nrow=2)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
