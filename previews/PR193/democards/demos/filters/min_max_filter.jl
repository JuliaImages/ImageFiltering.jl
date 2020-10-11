using ImageCore, ImageShow, ImageFiltering
using MappedArrays
using TestImages

img = Gray.(testimage("house")); # Original Image

window_size = (15, 15)
img_min = mapwindow(minimum, img, window_size)
img_max = mapwindow(maximum, img, window_size)
img_max_min = mapwindow(maximum, img_min, window_size)
img_min_max = mapwindow(minimum, img_max, window_size)
mosaicview(img_min, img_max, img_max_min, img_min_max; nrow=1)

img_extrema = mapwindow(extrema, img, window_size) # only compute once
img_min = mappedarray(first, img_extrema) # 0 allocation
img_max = mappedarray(last, img_extrema) # 0 allocation

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

