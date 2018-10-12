"""
    otf = psf2otf(psf, [outsize = size(psf)])
    otf = psf2otf(psf, dims...)

    Convert point-spread function to optical transfer function

    The tuple `outsize` can be used to specify the size of the `otf` array

    See also: [`otf2psf`](@ref).
"""
function psf2otf(psf::AbstractArray{T,N}, outsize::Tuple{Vararg{Integer}}=size(psf)) where {T,N}
    psf = repeat(psf, ones(Int,length(outsize))...) # make dimension consistent
    psfsize = size(psf)

    if psfsize != outsize
        pad = outsize .- psfsize
        all(pad .>= 0) || throw(DimensionMismatch("psf $psfsize too large for output otf $outsize."))
        lo_dim = tuple(zeros(Int,length(pad))...)
        hi_dim = pad
        psf = padarray(psf,Fill(zero(T),lo_dim,hi_dim))
    end

    shift = @. -floor(Int, psfsize/2)
    fft(circshift(psf, shift))
end
psf2otf(psf::AbstractArray, outsize::Integer...) = psf2otf(psf, tuple(outsize...))


"""
    psf = otf2psf(otf, [outsize = size(otf)])
    psf = otf2psf(otf, dims...)

    Convert optical transfer function to point-spread function

    See also: [`psf2otf`](@ref).
"""
function otf2psf(otf::AbstractArray{T,N}, outsize::Tuple{Vararg{Integer}}=size(otf)) where {T,N}
    otfsize = size(otf)
    outsize = (outsize..., 
        tuple(ones(Int, length(otfsize) - length(outsize))...)...) # make dimension consistent
    all(otfsize >= outsize ) || throw(DimensionMismatch("output psf $outsize too large for otf $otfsize."))

    psf = ifft(otf)
    shift = @. floor(Int, outsize/2)
    psf = circshift(psf,shift)
    psf[map(x -> 1:x, outsize)...]
end
otf2psf(otf::AbstractArray, outsize::Integer...) = otf2psf(otf, tuple(outsize...))
