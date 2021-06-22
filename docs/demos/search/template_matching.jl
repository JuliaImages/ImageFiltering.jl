using Images
using Plots 
using ImageFiltering
using ImageFeatures

function SDIFF(template)
  (patch)->sqeuclidean(patch .- template)
end
	
# generate a template
template = zeros(5,5)
template[1] = 1
template[2,1] = template[1,2]= 2

#generate an image
img = repeat(template,outer=(4,4))

#check correlation
res = mapwindow(SDIFF(template),img,size(template),border=Fill(1))

#select minimum values
th = res .< 0.1

#cluster into compenents and calculate centroid
centroids = component_centroids(label_components(th))[2:end]

#plot
plot(Gray.(img))
scatter!(reverse.(centroids))
