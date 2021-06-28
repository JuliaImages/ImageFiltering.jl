using ImageCore: Gray
using ImageMorphology: label_components, component_centroids
using ImageFiltering: mapwindow, Fill, imfilter, KernelFactors
using ImageDistances: sqeuclidean
using ImageContrastAdjustment: adjust_histogram, LinearStretching
using TestImages
using Plots: scatter!, plot

img = testimage("moonsurface")

template = img[12:22,20:30]
template = imfilter(template,KernelFactors.gaussian((0.5,0.5)))

function SDIFF(template)
  (subsection)->sqeuclidean(subsection, template)
end

res = mapwindow(SDIFF(template), img, size(template), border=Fill(1)) .|> Gray
rescaled_map = adjust_histogram(res, LinearStretching())

threshold = rescaled_map .< 0.05
Gray.(threshold)

centroids = component_centroids(label_components(threshold))[2:end]

plot(Gray.(img), size=(512,512))
scatter!(reverse.(centroids), label="centroids", ms=10, alpha=0.5, c=:red, msw=3)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

