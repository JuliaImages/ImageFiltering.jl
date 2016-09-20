function padarray(img::AbstractArray, prepad::Union{Vector{Int},Dims}, postpad::Union{Vector{Int},Dims}, border::AbstractString, value)
    if border == "value"
        depwarn("string-valued borders are deprecated, use `padarray(img, Fill(value, prepad, postpad))` instead, where the padding entries are Dims-tuples", :padarray)
        return padarray(img, Fill(value, (prepad...,), (postpad...,)))
    end
    padarray(img, prepad, postpad, border)
end

function padarray(img::AbstractArray, prepad::Union{Vector{Int},Dims}, postpad::Union{Vector{Int},Dims}, border::AbstractString)
    if border ∈ ["replicate", "circular", "reflect", "symmetric"]
        depwarn("string-valued borders are deprecated for `padarray`; use `padarray(img, Pad(:$border, prepad, postpad))` instead, where the padding entries are Dims-tuples", :padarray)
        return padarray(img, Pad(Symbol(border), (prepad...,), (postpad...,)))
    elseif border == "inner"
        depwarn("string-valued \"inner\" border is deprecated, use `padarray(img, Inner())` instead", :padarray)
        return padarray(img, Inner())
    else
        throw(ArgumentError("$border not a recognized border"))
    end
end

padarray(img::AbstractArray, padding::Union{Vector{Int},Dims}, border::AbstractString = "replicate") = padarray(img, padding, padding, border)
padarray{T<:Number}(img::AbstractArray{T}, padding::Union{Vector{Int},Dims}, value::T) = padarray(img, padding, padding, "value", value)

function padarray{T,n}(img::AbstractArray{T,n}, padding::Union{Vector{Int},Dims}, border::AbstractString, direction::AbstractString)
    if direction == "both"
        return padarray(img, padding, padding, border)
    elseif direction == "pre"
        return padarray(img, padding, zeros(Int, n), border)
    elseif direction == "post"
        return padarray(img, zeros(Int, n), padding, border)
    end
end

function padarray{T<:Number,n}(img::AbstractArray{T,n}, padding::Vector{Int}, value::T, direction::AbstractString)
    if direction == "both"
        return padarray(img, padding, padding, "value", value)
    elseif direction == "pre"
        return padarray(img, padding, zeros(Int, n), "value", value)
    elseif direction == "post"
        return padarray(img, zeros(Int, n), padding, "value", value)
    end
end

function imfilter(img::AbstractArray, kern, border::AbstractString, value)
    if border == "value"
        depwarn("string-valued borders are deprecated, use `imfilter(img, kern, Fill(value))` instead", :imfilter)
        return imfilter(img, kern, Fill(value))
    end
    imfilter(img, kern, border)
end

export imfilter_fft
function imfilter_fft(img, kern, border::AbstractString, value)
    if border == "value"
        depwarn("string-valued 'fill' borders are deprecated, use `imfilter(img, kern, Fill(value), Algorithm.FFT())` instead", :imfilter_fft)
        return imfilter(img, kern, Fill(value), Algorithm.FFT())
    elseif border ∈ ["replicate", "circular", "reflect", "symmetric"]
        return imfilter(img, kern, border, Algorithm.FFT())
    elseif border == "inner"
        depwarn("specifying \"inner\" as a string is deprecated, use `imfilter(img, kern, Inner(), Algorithm.FFT())` instead", :imfilter_fft)
        return imfilter(img, kern, Inner(), Algorithm.FFT())
    else
        throw(ArgumentError("$border not a recognized border"))
    end
end

imfilter_fft(img, filter) = imfilter_fft(img, filter, "replicate", 0)
imfilter_fft(img, filter, border) = imfilter_fft(img, filter, border, 0)

export imfilter_gaussian
function imfilter_gaussian(img, sigma; emit_warning=true, astype=nothing)
    if astype != nothing
        depwarn("imfilter_gaussian(img, sigma; astype=$astype, kwargs...) is deprecated; use `imfilter($astype, img, IIRGaussian(sigma; kwargs...))` instead, possibly with `NA()`", :imfilter_gaussian)
        factkernel = KernelFactors.IIRGaussian(astype, sigma; emit_warning=emit_warning)
        return imfilter(astype, img, factkernel, NA())
    end
    depwarn("imfilter_gaussian(img, sigma; kwargs...) is deprecated; use `imfilter(img, IIRGaussian(sigma; kwargs...))` instead, possibly with `NA()`", :imfilter_gaussian)
    factkernel = KernelFactors.IIRGaussian(sigma; emit_warning=emit_warning)
    imfilter(_eltype(Float64, eltype(img)), img, factkernel, NA())
end

_eltype{T,C<:Colorant}(::Type{T}, ::Type{C}) = base_colorant_type(C){T}
_eltype{T,R<:Real}(::Type{T}, ::Type{R}) = T

@deprecate imfilter_LoG(img, σ, border="replicate") imfilter(img, Kernel.LoG(σ), border)

function imgradients(img::AbstractArray, method::AbstractString)
    depwarn("imgradients requires the kernel to be specified as a function, e.g., KernelFactors.ando3. Note also that the order of outputs has switched.", :imgradients)
    imgradients(img, kernelfunc_lookup(method))
end

function kernelfunc_lookup(method::AbstractString)
    if method=="sobel"
        return KernelFactors.sobel
    elseif method=="prewitt"
        return KernelFactors.prewitt
    elseif method=="ando3"
        return KernelFactors.ando3
    elseif method=="ando4"
        return KernelFactors.ando4
    elseif method=="ando5"
        return KernelFactors.ando5
    else
        error("$method does not correspond to a known gradient method")
    end
end

function extrema_filter(A, window::AbstractVector{Int})
    depwarn("extrema_filter(A, window) has been replaced by rankfilter(extrema, A, window). Note that rankfilter returns a single array, rather than a pair Amin, Amax; it also preserves the input size.", :extrema_filter)
    Aminmax = rankfilter(extrema, A, window)
    minval, maxval = map(first, Aminmax), map(last, Aminmax)
    # For backwards-compatability, discard the edges
    halfsz = map(n->n>>1, window)
    inds = map((ind,h,w)->first(ind)+h:last(ind)-(w-h-1), indices(A), halfsz, window)
    minval[inds...], maxval[inds...]
end

# This was an internal---but exported---method
function extrema_filter(A, window::Int)
    depwarn("extrema_filter(A, window::Int) was an internal method and is being eliminated. Please see `rankfilter`.", :extrema_filter)
    Aminmax = rankfilter(extrema, vec(A), window)
    minval, maxval = map(first, Aminmax), map(last, Aminmax)
    ind1 = indices(Aminmax,1)
    halfsz = window>>1
    ind = first(ind1)+halfsz:last(ind1)-(window-halfsz-1)
    minval[ind], maxval[ind]
end

export extrema_filter
