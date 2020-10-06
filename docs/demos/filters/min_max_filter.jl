# Topic : Max Min Filters
# reference paper : https://core.ac.uk/download/pdf/194053536.pdf


# In this tutorial we see how can we can effectively use max and min filter to distinguish 
# between ramp("smooth") and ripple("texture") edges.We will using the mapwindow function of
# ImageFiltering.jl package.

# Note : If you are using the terminal use imshow(<Image var. name>) to view the image.


using Images, ImageFiltering

# We download an image from the internet and convert it into Grayscale image.
# To work with your own image, use :
# img_path = "<FULL PATH TO THE IMAGE"

img_path = download("https://i.pinimg.com/originals/01/bc/8d/01bc8d82a3e2b4fa869f478479b97a3f.png")
Img = Gray.(load(img_path));       

# We will use a minimum function to check the minimum of GrayScale Values in the given matrix,or array
# For Example
minimum([Gray(0.7),Gray(0.5),Gray(0.0)]) #Should reutrn Gray(0.0) i.e black.

# Using the mapwindow, create an image min.
Min = mapwindow(minimum, Img, (5, 5))

# Similarly for maximum
Max = mapwindow(minimum, Img, (5, 5))

# The max(min) filter
Upp = mapwindow(maximum, Min, (9, 9))

# The min(max) filter
Low = mapwindow(minimum, Max, (9, 9))

# No that we are done with the basic filtered images, we proceed to the next part
# which is edge detection using these filters.

# The trick is we define some thresholds.The thresholding can be dynamic for detecting
# smooth parts or it can be texture based thresholding for detecting ripple edges.

Dyt = (Max + Min)./ 2   # Dynamic Threshold(DYT)

Tet = (Upp + Low)./ 2   # Texture Threshold(TET)

# The Dynamic Gist(more focus on the the smooth edges)
Dyg = Img - Dyt

# The Texture Gist( isolates out the textures and/or noise)
Teg = Img - Tet

# Filtered Out edges
Edge = Img - Low

# Smoothed our version of edge
Edge_smoothed = Upp - Low