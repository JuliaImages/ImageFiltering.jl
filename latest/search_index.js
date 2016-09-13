var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "ImageFiltering.jl",
    "title": "ImageFiltering.jl",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#ImageFiltering.jl-1",
    "page": "ImageFiltering.jl",
    "title": "ImageFiltering.jl",
    "category": "section",
    "text": "ImageFiltering supports linear and nonlinear filtering operations on arrays, with an emphasis on the kinds of operations used in image processing. The core function is imfilter, and common kernels (filters) are organized in the Kernel and KernelFactors modules."
},

{
    "location": "index.html#Demonstration-1",
    "page": "ImageFiltering.jl",
    "title": "Demonstration",
    "category": "section",
    "text": "Let's start with a simple example of linear filtering:julia> using ImageFiltering, TestImages\n\njulia> img = testimage(\"mandrill\");\n\njulia> imgg = imfilter(img, Kernel.gaussian(3));\n\njulia> imgl = imfilter(img, Kernel.Laplacian());When displayed, these three images look like this:(Image: filterintro)The most commonly used function for filtering is imfilter."
},

{
    "location": "index.html#Linear-filtering:-noteworthy-features-1",
    "page": "ImageFiltering.jl",
    "title": "Linear filtering: noteworthy features",
    "category": "section",
    "text": "DocTestSetup = quote\n    using Colors, ImageFiltering, TestImages\n    img = testimage(\"mandrill\")\nend"
},

{
    "location": "index.html#Correlation,-not-convolution-1",
    "page": "ImageFiltering.jl",
    "title": "Correlation, not convolution",
    "category": "section",
    "text": "ImageFiltering uses the following formula to calculate the filtered image F from an input image A and kernel K:FI = sum_J AI+J KJConsequently, the resulting image is the correlation, not convolution, of the input and the kernel. If you want the convolution, first call reflect on the kernel."
},

{
    "location": "index.html#Kernel-indices-1",
    "page": "ImageFiltering.jl",
    "title": "Kernel indices",
    "category": "section",
    "text": "ImageFiltering exploits a feature introduced into Julia 0.5, the ability to define arrays whose indices span an arbitrary range:julia> Kernel.gaussian(1)\nOffsetArrays.OffsetArray{Float64,2,Array{Float64,2}} with indices -2:2×-2:2:\n 0.00296902  0.0133062  0.0219382  0.0133062  0.00296902\n 0.0133062   0.0596343  0.0983203  0.0596343  0.0133062\n 0.0219382   0.0983203  0.162103   0.0983203  0.0219382\n 0.0133062   0.0596343  0.0983203  0.0596343  0.0133062\n 0.00296902  0.0133062  0.0219382  0.0133062  0.00296902The indices of this array span the range -2:2 along each axis, and the center of the gaussian is at position [0,0].  As a consequence, this filter \"blurs\" but does not \"shift\" the image; were the center instead at, say, [3,3], the filtered image would be shifted by 3 pixels downward and to the right compared to the original.The centered function is a handy utility for converting an ordinary array to one that has coordinates [0,0,...] at its center position:julia> centered([1 0 1; 0 1 0; 1 0 1])\nOffsetArrays.OffsetArray{Int64,2,Array{Int64,2}} with indices -1:1×-1:1:\n 1  0  1\n 0  1  0\n 1  0  1See OffsetArrays for more information."
},

{
    "location": "index.html#Factored-kernels-1",
    "page": "ImageFiltering.jl",
    "title": "Factored kernels",
    "category": "section",
    "text": "A key feature of Gaussian kernels–-along with many other commonly-used kernels–-is that they are separable, meaning that K[j_1,j_2,...] can be written as K_1j_1 K_2j_2 cdots. As a consequence, the correlationFi_1i_2 = sum_j_1j_2 Ai_1+j_1i_2+j_2 Kj_1j_2can be writtenFi_1i_2 = sum_j_2 left(sum_j_1 Ai_1+j_1i_2+j_2 K_1j_1right) K_2j_2If the kernel is of size m×n, then the upper version line requires mn operations for each point of filtered, whereas the lower version requires m+n operations. Especially when m and n are larger, this can result in a substantial savings.To enable efficient computation for separable kernels, imfilter accepts a tuple of kernels, filtering the image by each sequentially. You can either supply m×1 and 1×n filters directly, or (somewhat more efficiently) call kernelfactors on a tuple-of-vectors:julia> kern1 = centered([1/3, 1/3, 1/3])\nOffsetArrays.OffsetArray{Float64,1,Array{Float64,1}} with indices -1:1:\n 0.333333\n 0.333333\n 0.333333\n\njulia> kernf = kernelfactors((kern1, kern1))\n(ImageFiltering.KernelFactors.ReshapedOneD{Float64,2,0,OffsetArrays.OffsetArray{Float64,1,Array{Float64,1}}}([0.333333,0.333333,0.333333]),ImageFiltering.KernelFactors.ReshapedOneD{Float64,2,1,OffsetArrays.OffsetArray{Float64,1,Array{Float64,1}}}([0.333333,0.333333,0.333333]))\n\njulia> kernp = broadcast(*, kernf...)\nOffsetArrays.OffsetArray{Float64,2,Array{Float64,2}} with indices -1:1×-1:1:\n 0.111111  0.111111  0.111111\n 0.111111  0.111111  0.111111\n 0.111111  0.111111  0.111111\n\njulia> imfilter(img, kernf) ≈ imfilter(img, kernp)\ntrueIf the kernel is a two dimensional array, imfilter will attempt to factor it; if successful, it will use the separable algorithm. You can prevent this automatic factorization by passing the kernel as a tuple, e.g., as (kernp,)."
},

{
    "location": "index.html#Popular-kernels-in-Kernel-and-KernelFactors-modules-1",
    "page": "ImageFiltering.jl",
    "title": "Popular kernels in Kernel and KernelFactors modules",
    "category": "section",
    "text": "The two modules Kernel and KernelFactors implement popular kernels in \"dense\" and \"factored\" forms, respectively. Type ?Kernel or ?KernelFactors at the REPL to see which kernels are supported.A common task in image processing and computer vision is computing image gradients (derivatives), for which there is the dedicated function imgradients."
},

{
    "location": "index.html#Automatic-choice-of-FIR-or-FFT-1",
    "page": "ImageFiltering.jl",
    "title": "Automatic choice of FIR or FFT",
    "category": "section",
    "text": "For linear filtering with a finite-impulse response filtering, one can either choose a direct algorithm or one based on the fast Fourier transform (FFT).  By default, this choice is made based on kernel size. You can manually specify the algorithm using Algorithm.FFT() or Algorithm.FIR()."
},

{
    "location": "index.html#Multithreading-1",
    "page": "ImageFiltering.jl",
    "title": "Multithreading",
    "category": "section",
    "text": "If you launch Julia with JULIA_NUM_THREADS=n (where n > 1), then FIR filtering will by default use multiple threads.  You can control the algorithm by specifying a resource as defined by ComputationalResources. For example, imfilter(CPU1(Algorithm.FIR()), img, ...) would force the computation to be single-threaded."
},

{
    "location": "index.html#Rank-filters-1",
    "page": "ImageFiltering.jl",
    "title": "Rank filters",
    "category": "section",
    "text": "This package also exports extrema_filter, which returns a (min,max) array representing the \"running\" (local) min/max around each point."
},

{
    "location": "function_reference.html#",
    "page": "Function reference",
    "title": "Function reference",
    "category": "page",
    "text": ""
},

{
    "location": "function_reference.html#ImageFiltering.imfilter",
    "page": "Function reference",
    "title": "ImageFiltering.imfilter",
    "category": "Function",
    "text": "imfilter([T], img, kernel, [border=\"replicate\"], [alg]) --> imgfilt\nimfilter([r], img, kernel, [border=\"replicate\"], [alg]) --> imgfilt\nimfilter(r, T, img, kernel, [border=\"replicate\"], [alg]) --> imgfilt\n\nFilter an array img with kernel kernel by computing their correlation.\n\nkernel[0,0,..] corresponds to the origin (zero displacement) of the kernel; you can use centered to place the origin at the array center, or use the OffsetArrays package to set kernel's indices manually. For example, to filter with a random centered 3x3 kernel, you could use either of the following:\n\nkernel = centered(rand(3,3))\nkernel = OffsetArray(rand(3,3), -1:1, -1:1)\n\nkernel can be specified as an array or as a \"factored kernel,\" a tuple (filt1, filt2, ...) of filters to apply along each axis of the image. In cases where you know your kernel is separable, this format can speed processing.  Each of these should have the same dimensionality as the image itself, and be shaped in a manner that indicates the filtering axis, e.g., a 3x1 filter for filtering the first dimension and a 1x3 filter for filtering the second dimension. In two dimensions, any kernel passed as a single matrix is checked for separability; if you want to eliminate that check, pass the kernel as a single-element tuple, (kernel,).\n\nOptionally specify the border, as one of Fill(value), \"replicate\", \"circular\", \"symmetric\", \"reflect\", NA(), or Inner(). The default is \"replicate\". These choices specify the boundary conditions, and therefore affect the result at the edges of the image. See padarray for more information.\n\nalg allows you to choose the particular algorithm: FIR() (finite impulse response, aka traditional digital filtering) or FFT() (Fourier-based filtering). If no choice is specified, one will be chosen based on the size of the image and kernel in a way that strives to deliver good performance. Alternatively you can use a custom filter type, like IIRGaussian.\n\nOptionally, you can control the element type of the output image by passing in a type T as the first argument.\n\nYou can also dispatch to different implementations by passing in a resource r as defined by the ComputationalResources package.  For example,\n\nimfilter(ArrayFire(), img, kernel)\n\nwould request that the computation be performed on the GPU using the ArrayFire libraries.\n\nSee also: imfilter!, centered, padarray, Pad, Fill, Inner, IIRGaussian.\n\n\n\n"
},

{
    "location": "function_reference.html#ImageFiltering.imfilter!",
    "page": "Function reference",
    "title": "ImageFiltering.imfilter!",
    "category": "Function",
    "text": "imfilter!(imgfilt, img, kernel, [border=\"replicate\"], [alg])\nimfilter!(r, imgfilt, img, kernel, border, [inds])\nimfilter!(r, imgfilt, img, kernel, border::NoPad, [inds=indices(imgfilt)])\n\nFilter an array img with kernel kernel by computing their correlation, storing the result in imgfilt.\n\nThe indices of imgfilt determine the region over which the filtered image is computed–-you can use this fact to select just a specific region of interest, although be aware that the input img might still get padded.  Alteratively, explicitly provide the indices inds of imgfilt that you want to calculate, and use NoPad boundary conditions. In such cases, you are responsible for supplying appropriate padding: img must be indexable for all of the locations needed for calculating the output. This syntax is best-supported for FIR filtering; in particular, that that IIR filtering can lead to results that are inconsistent with respect to filtering the entire array.\n\nSee also: imfilter.\n\n\n\n"
},

{
    "location": "function_reference.html#ImageFiltering.imgradients",
    "page": "Function reference",
    "title": "ImageFiltering.imgradients",
    "category": "Function",
    "text": "imgradients(img, [points], [method], [border])\n\nPerforms edge detection filtering in the N-dimensional array img. Gradients are computed at specified points (or indexes) in the array or everywhere.\n\nAvailable methods for 2D images: \"sobel\", \"prewitt\", \"ando3\", \"ando4\",                                  \"ando5\", \"ando4_sep\", \"ando5_sep\".\n\nAvailable methods for ND images: \"sobel\", \"prewitt\", \"ando3\", \"ando4\".\n\nBorder options:\"replicate\", \"circular\", \"reflect\", \"symmetric\".\n\nIf points is specified, returns a 2D array G with the gradients as rows. The number of rows is the number of points at which the gradient was computed and the number of columns is the dimensionality of the array.\n\nIf points is ommitted, returns a tuple of arrays, each of the same size of the input image: (gradx, grady, ...)\n\n\n\n"
},

{
    "location": "function_reference.html#ImageFiltering.Rank.extrema_filter",
    "page": "Function reference",
    "title": "ImageFiltering.Rank.extrema_filter",
    "category": "Function",
    "text": "extrema_filter(A, window) --> Array{(min,max)}\n\nCalculate the running min/max over a window of width window[d] along dimension d, centered on the current point. The returned array has the same indices as the input A.\n\n\n\n"
},

{
    "location": "function_reference.html#Filtering-functions-1",
    "page": "Function reference",
    "title": "Filtering functions",
    "category": "section",
    "text": "imfilter\nimfilter!\nimgradients\nextrema_filter"
},

{
    "location": "function_reference.html#ImageFiltering.Kernel.sobel",
    "page": "Function reference",
    "title": "ImageFiltering.Kernel.sobel",
    "category": "Function",
    "text": "diff1, diff2 = sobel()\n\nReturn kernels for two-dimensional gradient compution using the Sobel operator. diff1 computes the gradient along the first (y) dimension, and diff2 computes the gradient along the second (x) dimension.\n\nSee also: KernelFactors.sobel, Kernel.prewitt, Kernel.ando.\n\n\n\n"
},

{
    "location": "function_reference.html#ImageFiltering.Kernel.prewitt",
    "page": "Function reference",
    "title": "ImageFiltering.Kernel.prewitt",
    "category": "Function",
    "text": "diff1, diff2 = prewitt()\n\nReturn kernels for two-dimensional gradient compution using the Prewitt operator.  diff1 computes the gradient along the first (y) dimension, and diff2 computes the gradient along the second (x) dimension.\n\nSee also: KernelFactors.prewitt, Kernel.sobel, Kernel.ando.\n\n\n\n"
},

{
    "location": "function_reference.html#ImageFiltering.Kernel.ando3",
    "page": "Function reference",
    "title": "ImageFiltering.Kernel.ando3",
    "category": "Function",
    "text": "diff1, diff2 = ando3()\n\nReturn 3x3 kernels for two-dimensional gradient compution using the optimal \"Ando\" filters.  diff1 computes the gradient along the y-axis (first dimension), and diff2 computes the gradient along the x-axis (second dimension).\n\nCitation\n\nAndo Shigeru, IEEE Trans. Pat. Anal. Mach. Int., vol. 22 no 3, March 2000\n\nSee also: KernelFactors.ando3, Kernel.ando4, Kernel.ando5.\n\n\n\n"
},

{
    "location": "function_reference.html#ImageFiltering.Kernel.ando4",
    "page": "Function reference",
    "title": "ImageFiltering.Kernel.ando4",
    "category": "Function",
    "text": "diff1, diff2 = ando4()\n\nReturn 4x4 kernels for two-dimensional gradient compution using the optimal \"Ando\" filters.  diff1 computes the gradient along the y-axis (first dimension), and diff2 computes the gradient along the x-axis (second dimension).\n\nCitation\n\nAndo Shigeru, IEEE Trans. Pat. Anal. Mach. Int., vol. 22 no 3, March 2000\n\nSee also: KernelFactors.ando4, Kernel.ando3, Kernel.ando5.\n\n\n\n"
},

{
    "location": "function_reference.html#ImageFiltering.Kernel.ando5",
    "page": "Function reference",
    "title": "ImageFiltering.Kernel.ando5",
    "category": "Function",
    "text": "diff1, diff2 = ando5()\n\nReturn 5x5 kernels for two-dimensional gradient compution using the optimal \"Ando\" filters.  diff1 computes the gradient along the y-axis (first dimension), and diff2 computes the gradient along the x-axis (second dimension).\n\nCitation\n\nAndo Shigeru, IEEE Trans. Pat. Anal. Mach. Int., vol. 22 no 3, March 2000\n\nSee also: KernelFactors.ando5, Kernel.ando3, Kernel.ando4.\n\n\n\n"
},

{
    "location": "function_reference.html#ImageFiltering.Kernel.gaussian",
    "page": "Function reference",
    "title": "ImageFiltering.Kernel.gaussian",
    "category": "Function",
    "text": "gaussian((σ1, σ2, ...), [(l1, l2, ...]) -> g\ngaussian(σ)                  -> g\n\nConstruct a multidimensional gaussian filter, with standard deviation σd along dimension d. Optionally provide the kernel length l, which must be a tuple of the same length.\n\nIf σ is supplied as a single number, a symmetric 2d kernel is constructed.\n\nSee also: KernelFactors.gaussian.\n\n\n\n"
},

{
    "location": "function_reference.html#ImageFiltering.Kernel.DoG",
    "page": "Function reference",
    "title": "ImageFiltering.Kernel.DoG",
    "category": "Function",
    "text": "DoG((σp1, σp2, ...), (σm1, σm2, ...), [l1, l2, ...]) -> k\nDoG((σ1, σ2, ...))                                   -> k\nDoG(σ::Real)                                         -> k\n\nConstruct a multidimensional difference-of-gaussian kernel k, equal to gaussian(σp, l)-gaussian(σm, l).  When only a single σ is supplied, the default is to choose σp = σ, σm = √2 σ. Optionally provide the kernel length l; the default is to extend by two max(σp,σm) in each direction from the center. l must be odd.\n\nIf σ is provided as a single number, a symmetric 2d DoG kernel is returned.\n\nSee also: KernelFactors.IIRGaussian.\n\n\n\n"
},

{
    "location": "function_reference.html#ImageFiltering.Kernel.LoG",
    "page": "Function reference",
    "title": "ImageFiltering.Kernel.LoG",
    "category": "Function",
    "text": "LoG((σ1, σ2, ...)) -> k\nLoG(σ)             -> k\n\nConstruct a Laplacian-of-Gaussian kernel k. σd is the gaussian width along dimension d.  If σ is supplied as a single number, a symmetric 2d kernel is returned.\n\nSee also: KernelFactors.IIRGaussian and Kernel.Laplacian.\n\n\n\n"
},

{
    "location": "function_reference.html#ImageFiltering.Kernel.Laplacian",
    "page": "Function reference",
    "title": "ImageFiltering.Kernel.Laplacian",
    "category": "Type",
    "text": "Laplacian((true,true,false,...))\nLaplacian(dims, N)\nLacplacian()\n\nLaplacian kernel in N dimensions, taking derivatives along the directions marked as true in the supplied tuple. Alternatively, one can pass dims, a listing of the dimensions for differentiation. (However, this variant is not inferrable.)\n\nLaplacian() is the 2d laplacian, equivalent to Laplacian((true,true)).\n\nThe kernel is represented as an opaque type, but you can use convert(AbstractArray, L) to convert it into array format.\n\n\n\n"
},

{
    "location": "function_reference.html#Kernel-1",
    "page": "Function reference",
    "title": "Kernel",
    "category": "section",
    "text": "Kernel.sobel\nKernel.prewitt\nKernel.ando3\nKernel.ando4\nKernel.ando5\nKernel.gaussian\nKernel.DoG\nKernel.LoG\nKernel.Laplacian"
},

{
    "location": "function_reference.html#ImageFiltering.KernelFactors.sobel",
    "page": "Function reference",
    "title": "ImageFiltering.KernelFactors.sobel",
    "category": "Function",
    "text": "kern1, kern2 = sobel()\n\nFactored Sobel filters for dimensions 1 and 2 of a two-dimensional image. Each is a 2-tuple of one-dimensional filters.\n\n\n\nkern = sobel(extended::NTuple{N,Bool}, d)\n\nReturn a factored Sobel filter for computing the gradient in N dimensions along axis d. If extended[dim] is false, kern will have size 1 along that dimension.\n\n\n\n"
},

{
    "location": "function_reference.html#ImageFiltering.KernelFactors.prewitt",
    "page": "Function reference",
    "title": "ImageFiltering.KernelFactors.prewitt",
    "category": "Function",
    "text": "kern1, kern2 = prewitt() returns factored Prewitt filters for dimensions 1 and 2 of your image\n\n\n\nkern = prewitt(extended::NTuple{N,Bool}, d)\n\nReturn a factored Prewitt filter for computing the gradient in N dimensions along axis d. If extended[dim] is false, kern will have size 1 along that dimension.\n\n\n\n"
},

{
    "location": "function_reference.html#ImageFiltering.KernelFactors.ando3",
    "page": "Function reference",
    "title": "ImageFiltering.KernelFactors.ando3",
    "category": "Function",
    "text": "kern1, kern2 = ando3() returns optimal 3x3 gradient filters for dimensions 1 and 2 of your image, as defined in Ando Shigeru, IEEE Trans. Pat. Anal. Mach. Int., vol. 22 no 3, March 2000.\n\nSee also: ando4, ando5.\n\n\n\nkern = ando3(extended::NTuple{N,Bool}, d)\n\nReturn a factored Ando filter (size 3) for computing the gradient in N dimensions along axis d.  If extended[dim] is false, kern will have size 1 along that dimension.\n\n\n\n"
},

{
    "location": "function_reference.html#ImageFiltering.KernelFactors.ando4",
    "page": "Function reference",
    "title": "ImageFiltering.KernelFactors.ando4",
    "category": "Function",
    "text": "kern1, kern2 = ando4() returns separable approximations of the optimal 4x4 filters for dimensions 1 and 2 of your image, as defined in Ando Shigeru, IEEE Trans. Pat. Anal. Mach. Int., vol. 22 no 3, March 2000.\n\nSee also: Kernel.ando4.\n\n\n\nkern = ando4(extended::NTuple{N,Bool}, d)\n\nReturn a factored Ando filter (size 4) for computing the gradient in N dimensions along axis d.  If extended[dim] is false, kern will have size 1 along that dimension.\n\n\n\n"
},

{
    "location": "function_reference.html#ImageFiltering.KernelFactors.ando5",
    "page": "Function reference",
    "title": "ImageFiltering.KernelFactors.ando5",
    "category": "Function",
    "text": "kern1, kern2 = ando5_sep() returns separable approximations of the optimal 5x5 gradient filters for dimensions 1 and 2 of your image, as defined in Ando Shigeru, IEEE Trans. Pat. Anal. Mach. Int., vol. 22 no 3, March 2000.\n\nSee also: Kernel.ando5.\n\n\n\nkern = ando5(extended::NTuple{N,Bool}, d)\n\nReturn a factored Ando filter (size 5) for computing the gradient in N dimensions along axis d.  If extended[dim] is false, kern will have size 1 along that dimension.\n\n\n\n"
},

{
    "location": "function_reference.html#ImageFiltering.KernelFactors.gaussian",
    "page": "Function reference",
    "title": "ImageFiltering.KernelFactors.gaussian",
    "category": "Function",
    "text": "gaussian(σ::Real, [l]) -> g\n\nConstruct a 1d gaussian kernel g with standard deviation σ, optionally providing the kernel length l. The default is to extend by two σ in each direction from the center. l must be odd.\n\n\n\ngaussian((σ1, σ2, ...), [l]) -> (g1, g2, ...)\n\nConstruct a multidimensional gaussian filter as a product of single-dimension factors, with standard deviation σd along dimension d. Optionally provide the kernel length l, which must be a tuple of the same length.\n\n\n\n"
},

{
    "location": "function_reference.html#ImageFiltering.KernelFactors.IIRGaussian",
    "page": "Function reference",
    "title": "ImageFiltering.KernelFactors.IIRGaussian",
    "category": "Function",
    "text": "IIRGaussian([T], σ; emit_warning::Bool=true)\n\nConstruct an infinite impulse response (IIR) approximation to a Gaussian of standard deviation σ. σ may either be a single real number or a tuple of numbers; in the latter case, a tuple of such filters will be created, each for filtering a different dimension of an array.\n\nOptionally specify the type T for the filter coefficients; if not supplied, it will match σ (unless σ is not floating-point, in which case Float64 will be chosen).\n\nCitation\n\nI. T. Young, L. J. van Vliet, and M. van Ginkel, \"Recursive Gabor Filtering\". IEEE Trans. Sig. Proc., 50: 2798-2805 (2002).\n\n\n\n"
},

{
    "location": "function_reference.html#ImageFiltering.KernelFactors.TriggsSdika",
    "page": "Function reference",
    "title": "ImageFiltering.KernelFactors.TriggsSdika",
    "category": "Type",
    "text": "TriggsSdika(a, b, scale, M)\n\nDefines a kernel for one-dimensional infinite impulse response (IIR) filtering. a is a \"forward\" filter, b a \"backward\" filter, M is a matrix for matching boundary conditions at the right edge, and scale is a constant scaling applied to each element at the conclusion of filtering.\n\nCitation\n\nB. Triggs and M. Sdika, \"Boundary conditions for Young-van Vliet recursive filtering\". IEEE Trans. on Sig. Proc. 54: 2365-2367 (2006).\n\n\n\nTriggsSdika(ab, scale)\n\nCreate a symmetric Triggs-Sdika filter (with a = b = ab). M is calculated for you. Only length 3 filters are currently supported.\n\n\n\n"
},

{
    "location": "function_reference.html#KernelFactors-1",
    "page": "Function reference",
    "title": "KernelFactors",
    "category": "section",
    "text": "KernelFactors.sobel\nKernelFactors.prewitt\nKernelFactors.ando3\nKernelFactors.ando4\nKernelFactors.ando5\nKernelFactors.gaussian\nKernelFactors.IIRGaussian\nKernelFactors.TriggsSdika"
},

{
    "location": "function_reference.html#ImageFiltering.centered",
    "page": "Function reference",
    "title": "ImageFiltering.centered",
    "category": "Function",
    "text": "centered(kernel) -> shiftedkernel\n\nShift the origin-of-coordinates to the center of kernel. The center-element of kernel will be accessed by shiftedkernel[0, 0, ...].\n\nThis function makes it easy to supply kernels using regular Arrays, and provides compatibility with other languages that do not support arbitrary indices.\n\nSee also: imfilter.\n\n\n\n"
},

{
    "location": "function_reference.html#ImageFiltering.KernelFactors.kernelfactors",
    "page": "Function reference",
    "title": "ImageFiltering.KernelFactors.kernelfactors",
    "category": "Function",
    "text": "kernelfactors(factors::Tuple)\n\nPrepare a factored kernel for filtering. If passed a 2-tuple of vectors of lengths m and n, this will return a 2-tuple of ReshapedVectors that are effectively of sizes m×1 and 1×n. In general, each successive factor will be reshaped to extend along the corresponding dimension.\n\nIf passed a tuple of general arrays, it is assumed that each is shaped appropriately along its \"leading\" dimensions; the dimensionality of each is \"extended\" to N = length(factors), appending 1s to the size as needed.\n\n\n\n"
},

{
    "location": "function_reference.html#ImageFiltering.Kernel.reflect",
    "page": "Function reference",
    "title": "ImageFiltering.Kernel.reflect",
    "category": "Function",
    "text": "reflect(kernel) --> reflectedkernel\n\nCompute the pointwise reflection around 0, 0, ... of the kernel kernel.  Using imfilter with a reflectedkernel performs convolution, rather than correlation, with respect to the original kernel.\n\n\n\n"
},

{
    "location": "function_reference.html#Kernel-utilities-1",
    "page": "Function reference",
    "title": "Kernel utilities",
    "category": "section",
    "text": "centered\nkernelfactors\nreflect"
},

{
    "location": "function_reference.html#ImageFiltering.padarray",
    "page": "Function reference",
    "title": "ImageFiltering.padarray",
    "category": "Function",
    "text": "padarray([T], img, border) --> imgpadded\n\nGenerate a padded image from an array img and a specification border of the boundary conditions and amount of padding to add. border can be a Pad, Fill, or Inner object.\n\nOptionally provide the element type T of imgpadded.\n\n\n\n"
},

{
    "location": "function_reference.html#ImageFiltering.Pad",
    "page": "Function reference",
    "title": "ImageFiltering.Pad",
    "category": "Type",
    "text": "Pad is a type that stores choices about padding. Instances must set style, a Symbol specifying the boundary conditions of the image, one of:\n\n:replicate (repeat edge values to infinity)\n:circular (image edges \"wrap around\")\n:symmetric (the image reflects relative to a position between pixels)\n:reflect (the image reflects relative to the edge itself)\n\nThe default value is :replicate.\n\nIt's worth emphasizing that padding is most straightforwardly specified as a string,\n\nimfilter(img, kernel, \"replicate\")\n\nrather than\n\nimfilter(img, kernel, Pad(:replicate))\n\n\n\n"
},

{
    "location": "function_reference.html#ImageFiltering.Fill",
    "page": "Function reference",
    "title": "ImageFiltering.Fill",
    "category": "Type",
    "text": "Fill(val)\nFill(val, lo, hi)\n\nPad the edges of the image with a constant value, val.\n\nOptionally supply the extent of the padding, see Pad.\n\nExample:\n\nimfilter(img, kernel, Fill(zero(eltype(img))))\n\n\n\n"
},

{
    "location": "function_reference.html#ImageFiltering.Inner",
    "page": "Function reference",
    "title": "ImageFiltering.Inner",
    "category": "Type",
    "text": "Inner()\nInner(lo, hi)\n\nIndicate that edges are to be discarded in filtering, only the interior of the result it to be returned.\n\nExample:\n\nimfilter(img, kernel, Inner())\n\n\n\n"
},

{
    "location": "function_reference.html#ImageFiltering.NA",
    "page": "Function reference",
    "title": "ImageFiltering.NA",
    "category": "Type",
    "text": "NA()\nNA(lo, hi)\n\nChoose filtering using \"NA\" (Not Available) boundary conditions. This is most appropriate for filters that have only positive weights, such as blurring filters. Effectively, the output pixel value is normalized in the following way:\n\n          filtered img with Fill(0) boundary conditions\noutput =  ---------------------------------------------\n          filtered 1   with Fill(0) boundary conditions\n\nAs a consequence, filtering has the same behavior as nanmean. Indeed, invalid pixels in img can be marked as NaN and then they are effectively omitted from the filtered result.\n\n\n\n"
},

{
    "location": "function_reference.html#ImageFiltering.NoPad",
    "page": "Function reference",
    "title": "ImageFiltering.NoPad",
    "category": "Type",
    "text": "NoPad()\nNoPad(border)\n\nIndicates that no padding should be applied to the input array, or that you have already pre-padded the input image. Passing a border object allows you to preserve \"memory\" of a border choice; it can be retrieved by indexing with [].\n\nExample\n\nnp = NoPad(Pad(:replicate))\nimfilter!(out, img, kernel, np)\n\nruns filtering directly, skipping any padding steps.  Every entry of out must be computable using in-bounds operations on img and kernel.\n\n\n\n"
},

{
    "location": "function_reference.html#Boundaries-and-padding-1",
    "page": "Function reference",
    "title": "Boundaries and padding",
    "category": "section",
    "text": "padarray\nPad\nFill\nInner\nNA\nNoPad"
},

{
    "location": "function_reference.html#ImageFiltering.Algorithm.FIR",
    "page": "Function reference",
    "title": "ImageFiltering.Algorithm.FIR",
    "category": "Type",
    "text": "Filter using a direct algorithm\n\n\n\n"
},

{
    "location": "function_reference.html#ImageFiltering.Algorithm.FFT",
    "page": "Function reference",
    "title": "ImageFiltering.Algorithm.FFT",
    "category": "Type",
    "text": "Filter using the Fast Fourier Transform\n\n\n\n"
},

{
    "location": "function_reference.html#ImageFiltering.Algorithm.IIR",
    "page": "Function reference",
    "title": "ImageFiltering.Algorithm.IIR",
    "category": "Type",
    "text": "Filter with an Infinite Impulse Response filter\n\n\n\n"
},

{
    "location": "function_reference.html#ImageFiltering.Algorithm.Mixed",
    "page": "Function reference",
    "title": "ImageFiltering.Algorithm.Mixed",
    "category": "Type",
    "text": "Filter with a cascade of mixed types (IIR, FIR)\n\n\n\n"
},

{
    "location": "function_reference.html#Algorithms-1",
    "page": "Function reference",
    "title": "Algorithms",
    "category": "section",
    "text": "Algorithm.FIR\nAlgorithm.FFT\nAlgorithm.IIR\nAlgorithm.Mixed"
},

{
    "location": "function_reference.html#ImageFiltering.KernelFactors.ReshapedOneD",
    "page": "Function reference",
    "title": "ImageFiltering.KernelFactors.ReshapedOneD",
    "category": "Type",
    "text": "ReshapedOneD{N,Npre}(data)\n\nReturn an object of dimensionality N, where data must have dimensionality 1. The indices are 0:0 for the first Npre dimensions, have the indices of data for dimension Npre+1, and are 0:0 for the remaining dimensions.\n\ndata must support eltype and ndims, but does not have to be an AbstractArray.\n\nReshapedOneDs allow one to specify a \"filtering dimension\" for a 1-dimensional filter.\n\n\n\n"
},

{
    "location": "function_reference.html#Internal-machinery-1",
    "page": "Function reference",
    "title": "Internal machinery",
    "category": "section",
    "text": "KernelFactors.ReshapedOneD"
},

]}
