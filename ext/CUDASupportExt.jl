module CUDASupportExt

using CUDA
using Base
using ImageFiltering

export findlocalextrema, findlocalmaxima, findlocalminima

import ImageFiltering: findlocalextrema, findlocalmaxima, findlocalminima


findlocalextrema(f, img::CuArray{T,N}, window, edges::Bool) where {T,N} =
    findlocalextrema(f, img, window, ntuple(_ -> edges, N))

function findlocalextrema(
    f,
    img::CuArray{T,N},
    window::Dims{N},
    edges::NTuple{N,Bool},
) where {T,N}

    @assert all(isodd, window) "window entries must be odd"

    mode = _extrema_mode(f)

    mask = similar(img, Bool)
    _localextrema_mask!(mask, img, window, edges, mode)

    return findall(Array(mask))
end

function findlocalmaxima(img::CuArray; window=_default_window_cuda(img), edges=true)
    findlocalextrema(>, img, window, edges)
end

function findlocalminima(img::CuArray; window=_default_window_cuda(img), edges=true)
    findlocalextrema(<, img, window, edges)
end

# --------------------------------------------------
# Default window from coords_spatial
# --------------------------------------------------

function _default_window_cuda(img)
    spatial = ImageFiltering.coords_spatial(img)
    spatial_set = Set(spatial)
    return ntuple(d -> (d in spatial_set ? 3 : 1), ndims(img))
end

# --------------------------------------------------
# Comparison mode
# --------------------------------------------------

@inline function _extrema_mode(f)
    if f === >
        return Int32(1)
    elseif f === <
        return Int32(-1)
    else
        throw(ArgumentError("CUDA findlocalextrema currently supports only `>` and `<`"))
    end
end

@inline _cmp(mode::Int32, a, b) = mode == 1 ? (a > b) : (a < b)

# --------------------------------------------------
# Metadata preparation on host
# --------------------------------------------------

function _compute_strides(dims::NTuple{N,Int}) where {N}
    s = Vector{Int}(undef, N)
    stride = 1
    for d in 1:N
        s[d] = stride
        stride *= dims[d]
    end
    return s
end

function _localextrema_mask!(
    mask::CuArray{Bool,N},
    img::CuArray{T,N},
    window::Dims{N},
    edges::NTuple{N,Bool},
    mode::Int32,
) where {T,N}

    dims_h    = collect(Int, size(img))
    strides_h = _compute_strides(size(img))
    halfw_h = [window[d] ÷ 2 for d in 1:N]
    clip_h  = [edges[d] ? 0 : max(1, halfw_h[d]) for d in 1:N]

    dims_d    = CuArray(dims_h)
    strides_d = CuArray(strides_h)
    halfw_d   = CuArray(halfw_h)
    clip_d    = CuArray(clip_h)

    len = length(img)
    threads = 256
    blocks  = cld(len, threads)

    @cuda threads=threads blocks=blocks _localextrema_nd!(
        mask, img, dims_d, strides_d, halfw_d, clip_d, Int32(N), mode, len
    )

    return mask
end

# --------------------------------------------------
# ND kernel with runtime dimension metadata
# --------------------------------------------------

function _localextrema_nd!(
    mask,
    A,
    dims,
    strides,
    halfw,
    clip,
    nd::Int32,
    mode::Int32,
    len::Int,
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if idx > len
        return
    end

    # Recover coordinates from linear index
    # coord[d] is computed on the fly; no tuple creation.
    center_ok = true
    linear0 = idx - 1

    # First pass: check whether this center is eligible
    for d in 1:nd
        coord_d = (linear0 ÷ strides[d]) % dims[d] + 1
        if !(1 + clip[d] <= coord_d <= dims[d] - clip[d])
            center_ok = false
            break
        end
    end

    if !center_ok
        @inbounds mask[idx] = false
        return
    end

    c = @inbounds A[idx]

    # Total number of offsets in the window
    total_offsets = 1
    for d in 1:nd
        total_offsets *= (2 * halfw[d] + 1)
    end

    isext = true

    # Enumerate offsets in mixed radix
    for k in 0:(total_offsets - 1)
        t = k
        neigh_idx = idx
        allzero = true
        inbounds = true

        for d in 1:nd
            width_d = 2 * halfw[d] + 1
            off_d = (t % width_d) - halfw[d]
            t ÷= width_d

            allzero &= (off_d == 0)

            coord_d = (linear0 ÷ strides[d]) % dims[d] + 1
            neigh_coord_d = coord_d + off_d

            if !(1 <= neigh_coord_d <= dims[d])
                inbounds = false
                break
            end

            neigh_idx += off_d * strides[d]
        end

        if !allzero && inbounds
            v = @inbounds A[neigh_idx]
            if !_cmp(mode, c, v)
                isext = false
                break
            end
        end
    end

    @inbounds mask[idx] = isext
    return
end

end # module