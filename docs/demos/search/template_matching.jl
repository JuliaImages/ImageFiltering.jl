# ---
# title: Template Matching
# cover: assets/template_matching.png
# author: Michael Pusterhofer
# date: 2021-06-22
# ---

# This demo shows how to find objects in an image using template matching. 

# The main idea is to check the similarity between a search target(template) and a subsection of the image.
# The subsection is usually the same size as the template and every subsection must be assigned a value.
# This can be done using the [`mapwindow`](@ref) function

# At first we import the following packages. 
using ImageCore: Gray
using ImageDistances: sqeuclidean
using ImageFiltering: mapwindow, Fill
using ImageContrastAdjustment: adjust_histogram, LinearStretching
using ImageMorphology: label_components, component_centroids
using Plots: scatter!, plot

# Images enables the generation of Images, ImageFiltering provides the mapwindow function and ImageFeatures 
# provides functions to label segments of an image.

# To test the algorithm we first generate an image. For our case we will repeat a square image section which 
# will also work as our template.

template = zeros(11,11)
template[2:10,2:10] .= 1
template[5:7,5:7] .= 0.5
Gray.(template)
#
img = ones(100,100)
img[(1:11).+20,(1:11).+50] .= template
img[(1:11).+50,(1:11).+10] .= template
img[(1:11).+70,(1:11).+80] .= template
img[:,:] .*= rand(100,100) 
Gray.(img)

# Now that we have an image and a template, the next step is to define how we measure the similarity between a
# section of the image and the template. This can be done in multiple way, but a sum of square distances should work quite well.
# The ImageDistance package provides an already optimized version called sqeuclidean, which can be used to define a function for mapwindow.
# Lets call it SDIFF.

function SDIFF(template)
  (subsection)->sqeuclidean(subsection, template)
end

# To actually generate our similarity map we use mapwindow in the following way

res = mapwindow(SDIFF(template), img, size(template), border=Fill(1)) .|> Gray
	
# If the subsection is located at the border of the image the image has to be extended and in our case we will
# fill all values outside the image with 1. As all of the square differences will be summed up per subsection 
# the resulting sum can be bigger than 1. This will be a problem if we just convert it to an image to check  the values.
# To rescale the values to be between 0 and 1 we can use imadjustintensity.

rescaled_map = adjust_histogram(res, LinearStretching()) 

#  To find the best locations we have to look for small values on the similarity map. This can be done by comparing
# if the pixel is below a certain value. Let's chose a value of `0.1`.

threshold = rescaled_map .< 0.1
Gray.(threshold)

# Now we see small blobs at the locations which match our template and we can label the connected regions by `label_components`.
# This will enumerate are connected regions and component_centroids can be used to get the centroid of each region. 
# `component_centroids` also return the centroid for the backgroud region, which is at the first position and we will ommit it.

centroids = component_centroids(label_components(threshold))[2:end]

# To now see it it worked ciorrectly we can overlay the centroids with the original image using the Plots package.
# As the images are stored 
plot(Gray.(img))
scatter!(reverse.(centroids),label="centroids",legend=:outertopleft)

savefig("assets/template_matching.png") #src
