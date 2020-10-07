# ---
# title: Max min filters
# cover: assets/max_min.gif
# author: Deeptendu Santra
# date: 2020-10-08
# ---

# In this tutorial we see how can we can effectively use max and min filter to distinguish 
# between smooth and texture edges  in grayscale images.

# We will be using the [`mapwindow`](@ref) in `ImageFiltering.jl` which provides a genreal 
# functionality to apply any function to the window around each pixel. 

using ImageCore,ImageShow, ImageFiltering
using TestImages

img = Gray.(testimage("lighthouse"))      #Original Image       

# We can use a minimum function to check the minimum of GrayScale Values in the given matrix,or array
# . For Example : 
minimum([Gray(0.7),Gray(0.5),Gray(0.0)]) ##Should return Gray(0.0) i.e black.
# 
## Using the mapwindow, we create an image min.
## The mapwindow maps the function over a moving window of given size
img_min = mapwindow(minimum, img, (3, 3))
## Similarly for maximum
img_max = mapwindow(minimum, img, (3, 3))
## The max(min) filter
img_max_min = mapwindow(maximum, img_min, (3, 3))
## The min(max) filter
img_min_max = mapwindow(minimum, img_max, (3, 3))
mosaicview(img_min, img_max, img_max_min, img_min_max; nrow = 1)

# Now that we are done with the basic filtered images, we proceed to the next part
# which is edge detection using these filters.

# For edge detection we need define thresholds for our image.Threshold is an important tool 
# for binarizing a grayscale image.The threshold value for a given pixel practically
# decides if the pixel visible or not in the output image.Appyling a global threshold might underconsider the 
# variation of colors/brightness within the image.So we consider an adaptive type theresholding.

# The max(min) and min(max) filter effectively follows the smooth edges of the image.So consequently,
# their average also follows the smooth parts of the image.If we use this image as a threshold for the original
# image, the smooth parts of the original image gets filtered out, leaving only the texture and/or noise behind.
# So we can use  the average of `img_max_min` and `img_min_max` as the texture threshold.

# The average of min and max filters also gives us the smooth edges, but it also includes the noise in the image. 
# So using average of `img_min` and `img_max` as a threshold  will yeild only texture.

img_text_noise_threshold = (img_max_min + img_max_min)./ 2
img_text_threshold = (img_max + img_min)./2
mosaicview(img_text_noise_threshold, img_text_threshold; nrow = 1)
#
## The Dynamic Gist is obtained by subtracting the img_text_threshold from the original image.
## The filtered image gives us the texture of the image.
img_dynamic_gist = img - img_text_threshold
## The Texture Gist is obtained by subtracting img_text_noise_threshold from the original image.
## The filtered image gives us the texture along with the noise of the image.
img_texture_gist = img - img_text_noise_threshold
mosaicview(img_dynamic_gist, img_texture_gist; nrow = 1)
#
ramp = img_dynamic_gist - img_texture_gist
## Filtered Out edges
edge = img - img_min_max
## Smoothed our version of edge
edge_smoothed = img_max_min - img_min_max
mosaicview(img, ramp, edge, edge_smoothed; nrow = 2)

## save covers #src
using ImageMagick #src
mkpath("assets")  #src
ImageMagick.save("assets/max_min.gif",cat(img,edge,edge_smoothed,map(clamp01nan,ramp);dims=3);fps =1) #src