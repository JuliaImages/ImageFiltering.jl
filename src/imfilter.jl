# Step 1: if necessary, determine the output's element type
@inline function imfilter(img::AbstractArray, kernel, args...)
    imfilter(filter_type(img, kernel), img, kernel, args...)
end

# Step 2: if necessary, put the kernel into cannonical (factored) form
@inline function imfilter(::Type{T}, img::AbstractArray, kernel::Union{ArrayLike,Laplacian}, args...) where {T}
    imfilter(T, img, factorkernel(kernel), args...)
end
@inline function imfilter(::Type{T}, img::AbstractArray{TI}, kernel::AbstractArray{TK}, args...) where {T<:Integer,TI<:Integer,TK<:Integer}
    imfilter(T, img, (kernel,), args...)
end

# Step 3: if necessary, fill in the default border
function imfilter(::Type{T}, img::AbstractArray, kernel::ProcessedKernel, args...) where {T}
    imfilter(T, img, kernel, "replicate", args...)
end

function imfilter(::Type{T}, img::AbstractArray, kernel::ProcessedKernel, border::AbstractString, args...) where {T}
    imfilter(T, img, kernel, borderinstance(border), args...)
end

# Step 4: if necessary, allocate the ouput
@inline function imfilter(::Type{T}, img::AbstractArray, kernel::ProcessedKernel, border::AbstractBorder, args...) where {T}
    imfilter!(allocate_output(T, img, kernel, border), img, kernel, border, args...)
end

# Now do the same steps for the case where the user supplies a Resource
@inline function imfilter(r::AbstractResource, img::AbstractArray, kernel, args...)
    imfilter(r, filter_type(img, kernel), img, kernel, args...)
end

@inline function imfilter(r::AbstractResource, ::Type{T}, img::AbstractArray, kernel::ArrayLike, args...) where {T}
    imfilter(r, T, img, factorkernel(kernel), args...)
end

# For steps 3 & 4, we make args... explicit as a means to prevent
# specifying both r and an algorithm
function imfilter(r::AbstractResource, ::Type{T}, img::AbstractArray, kernel::ProcessedKernel) where {T}
    imfilter(r, T, img, kernel, Pad(:replicate))  # supply the default border
end

function imfilter(r::AbstractResource, ::Type{T}, img::AbstractArray, kernel::ProcessedKernel, border::AbstractString) where {T}
    imfilter(r, T, img, kernel, borderinstance(border))
end

function imfilter(r::AbstractResource, ::Type{T}, img::AbstractArray, kernel::ProcessedKernel, border::AbstractBorder) where {T}
    imfilter!(r, allocate_output(T, img, kernel, border), img, kernel, border)
end

"""
    imfilter([T], img, kernel, [border="replicate"], [alg]) --> imgfilt
    imfilter([r], img, kernel, [border="replicate"], [alg]) --> imgfilt
    imfilter(r, T, img, kernel, [border="replicate"], [alg]) --> imgfilt

Filter a one, two or multidimensional array `img` with a `kernel` by computing
their correlation.

# Extended help

### Choices for `r`

Optionally, you can dispatch to different implementations by passing in a resource `r`
as defined by the [ComputationalResources](https://github.com/timholy/ComputationalResources.jl) package.

For example:

```julia
imfilter(ArrayFireLibs(), img, kernel)
```

would request that the computation be performed on the GPU using the
ArrayFire libraries.

### Choices for `T`

Optionally, you can control the element type of the output image by
passing in a type `T` as the first argument.

### Choices for `img`

You can specify a one, two, or multidimensional array defining your image.

### Choices for `kernel`

The `kernel[0, 0,..]` parameter corresponds to the origin (zero displacement) of
the kernel; you can use `centered` to place the origin at the array center, or
use the OffsetArrays package to set `kernel`'s indices manually. For example, to
filter with a random *centered* 3x3 kernel, you could use either of the
following:

```julia
kernel = centered(rand(3,3))
kernel = OffsetArray(rand(3,3), -1:1, -1:1)
```

The `kernel` parameter can be specified as an array or as a "factored kernel", a
tuple `(filt1, filt2, ...)` of filters to apply along each axis of the image. In
cases where you know your kernel is separable, this format can speed processing.
Each of these should have the same dimensionality as the image itself, and be
shaped in a manner that indicates the filtering axis, e.g., a 3x1 filter for
filtering the first dimension and a 1x3 filter for filtering the second
dimension. In two dimensions, any kernel passed as a single matrix is checked
for separability; if you want to eliminate that check, pass the kernel as a
single-element tuple, `(kernel,)`.

### Choices for `border`

At the image edge, `border` is used to specify the padding which will be used
to extrapolate the image beyond its original bounds. 

#### `"replicate"` (default)

The border pixels extend beyond the image boundaries.

```plain
   ╭────┏━━━━━━┓────╮
   │aaaa┃abcdef┃ffff│
   ╰────┗━━━━━━┛────╯
```

#### `"circular"`

The border pixels wrap around. For instance, indexing beyond the left border
returns values starting from the right border.

```plain

   ╭────┏━━━━━━┓────╮
   │cdef┃abcdef┃abcd│
   ╰────┗━━━━━━┛────╯

```

#### `"reflect"`

The border pixels reflect relative to a position between pixels. That is, the
border pixel is omitted when mirroring.

```plain

   ╭────┏━━━━━━┓────╮
   │dcba┃abcdef┃fedc│
   ╰────┗━━━━━━┛────╯

```

#### `"symmetric"`

The border pixels reflect relative to the edge itself.

```plain

   ╭────┏━━━━━━┓────╮
   │edcb┃abcdef┃edcb│
   ╰────┗━━━━━━┛────╯

```

#### `Fill(m)`

The border pixels are filled with a specified value ``m``.

```plain

   ╭────┏━━━━━━┓────╮
   │mmmm┃abcdef┃mmmm│
   ╰────┗━━━━━━┛────╯

```

#### `Inner()`

Indicate that edges are to be discarded in filtering, only the interior of the
result is to be returned.

#### `NA()`

Choose filtering using "NA" (Not Available) boundary conditions. This
is most appropriate for filters that have only positive weights, such
as blurring filters.

See also: [`Pad`](@ref), [`padarray`](@ref), [`Inner`](@ref), [`NA`](@ref)  and
[`NoPad`](@ref)

### Choices for `alg`

The `alg` parameter allows you to choose the particular algorithm: `Algorithm.FIR()`
(finite impulse response, aka traditional digital filtering) or `Algorithm.FFT()`
(Fourier-based filtering). If no choice is specified, one will be chosen based
on the size of the image and kernel in a way that strives to deliver good
performance. Alternatively you can use a custom filter type, like
[`KernelFactors.IIRGaussian`](@ref).

"""
imfilter

# see below for imfilter! docstring

# imfilter! can be called directly, so we take steps 2&3 here too. We
# have to be a little more cautious to make sure that later methods
# don't inadvertently call back to these: in methods that take an
# AbstractResource argument, exclude `NoPad()` as a border option.
function imfilter!(out::AbstractArray, img::AbstractArray, kernel::Union{ArrayLike,Laplacian}, args...)
    imfilter!(out, img, factorkernel(kernel), args...)
end

function imfilter!(r::AbstractResource, out::AbstractArray, img::AbstractArray, kernel, args...)
    imfilter!(r, out, img, factorkernel(kernel), args...)
end

function imfilter!(out::AbstractArray, img::AbstractArray, kernel::ProcessedKernel)
    imfilter!(out, img, kernel, Pad(:replicate))
end

function imfilter!(r::AbstractResource, out::AbstractArray, img::AbstractArray, kernel::ProcessedKernel)
    imfilter!(r, out, img, kernel, Pad(:replicate))
end

function imfilter!(out::AbstractArray, img::AbstractArray, kernel::ProcessedKernel, border::AbstractString, args...)
    imfilter!(out, img, kernel, borderinstance(border), args...)
end

function imfilter!(r::AbstractResource, out::AbstractArray, img::AbstractArray, kernel::ProcessedKernel, border::AbstractString)
    imfilter!(r, out, img, kernel, borderinstance(border))
end

# Step 5: if necessary, pick an algorithm
function imfilter!(out::AbstractArray, img::AbstractArray, kernel::ProcessedKernel, border::AbstractBorder)
    imfilter!(out, img, kernel, border, filter_algorithm(out, img, kernel))
end

function imfilter!(out::AbstractArray, img::AbstractArray, kernel::ProcessedKernel, border::AbstractBorder, alg::Alg)
    local ret
    try
        ret = imfilter!(default_resource(alg_defaults(alg, out, kernel)), out, img, kernel, border)
    catch err
        if isa(err, InexactError)
            Tw = Float64
            if eltype(img) <: Integer
                try
                    # If a type doesn't support widen, it would be bad
                    # if our attempt to be helpful triggered a
                    # completely different error...
                    Tw = widen(eltype(img))
                catch
                end
            end
            @warn "Likely overflow or conversion error detected. Consider specifying the output type, e.g., `imfilter($Tw, img, kernel, ...)`"
        end
        rethrow(err)
    end
    ret
end

"""
    imfilter!(imgfilt, img, kernel, [border="replicate"], [alg])
    imfilter!(r, imgfilt, img, kernel, border::Pad)
    imfilter!(r, imgfilt, img, kernel, border::NoPad, [inds=axes(imgfilt)])

Filter an array `img` with kernel `kernel` by computing their
correlation, storing the result in `imgfilt`.

The indices of `imgfilt` determine the region over which the filtered
image is computed---you can use this fact to select just a specific
region of interest, although be aware that the input `img` might still
get padded.  Alteratively, explicitly provide the indices `inds` of
`imgfilt` that you want to calculate, and use `NoPad` boundary
conditions. In such cases, you are responsible for supplying
appropriate padding: `img` must be indexable for all of the locations
needed for calculating the output. This syntax is best-supported for
FIR filtering; in particular, that that IIR filtering can lead to
results that are inconsistent with respect to filtering the entire
array.

See also: [`imfilter`](@ref).
"""
imfilter!

# Step 6: pad the input
# NA "padding": normalizing by the number of available values (similar to nanmean)
function imfilter!(r::AbstractResource,
    out::AbstractArray{S,N},
    img::AbstractArray{T,N},
    kernel::ProcessedKernel,
    border::NA{na}) where {T,S,N,na}
    _imfilter_na!(r, out, img, kernel, na)
end

function _imfilter_na!(r::AbstractResource,
    out::AbstractArray{S,N},
    img::AbstractArray{T,N},
    kernel::ProcessedKernel,
    na) where {T,S,N}
    naflag = na.(img)
    hasna = any(naflag)
    if isseparable(kernel) && !hasna
        imfilter_na_separable!(r, out, img, kernel)
    else
        imfilter_na_inseparable!(r, out, img, naflag, kernel)
    end
    out
end

# for types that can't have NaNs, we can skip the isnan check
function _imfilter_na!(r::AbstractResource,
    out::AbstractArray{S,N},
    img::AbstractArray{T,N},
    kernel::ProcessedKernel,
    na::typeof(isnan)) where {T<:Union{Integer,FixedColorant},S,N}
    if !isseparable(kernel)
        naflag = fill(false, axes(img))
        imfilter_na_inseparable!(r, out, img, naflag, kernel)
    else
        imfilter_na_separable!(r, out, img, kernel)
    end
    out
end

# Any other kind of not-fully-specified padding
function imfilter!(r::AbstractResource,
    out::AbstractArray{S,N},
    img::AbstractArray{T,N},
    kernel::ProcessedKernel,
    border::BorderSpecAny) where {S,T,N}
    bord = border(kernel, img, Alg(r))  # if it's FFT, the size of img is also relevant
    imfilter!(r, out, img, kernel, bord)
end

# Any fully-specified padding
function imfilter!(r::AbstractResource,
    out::AbstractArray{S,N},
    img::AbstractArray{T,N},
    kernel::ProcessedKernel,
    border::AbstractBorder) where {S,T,N}
    A = padarray(S, img, border)
    # By specifying NoPad(), we ensure that dispatch will never
    # accidentally "go back" to an earlier routine and apply more
    # padding
    imfilter!(r, out, A, kernel, NoPad(border))
end

# # An optimized case that performs only "virtual padding"
# function imfilter!{S,T,N,A<:Union{FIR,FIRTiled}}(r::AbstractCPU{A},
#                                                  out::AbstractArray{S,N},
#                                                  img::AbstractArray{T,N},
#                                                  kernel::ProcessedKernel,
#                                                  border::Pad{0})
#     # The fast path: handle the points that don't need padding
#     iinds = map(intersect, interior(img, kernel), axes(out))
#     imfilter!(r, out, img, kernel, NoPad(border), iinds)
#     # The not-so-fast path: handle the edges
#     # TODO: when the kernel is factored, move this logic in to each factor
#     # This is especially important for bigger kernels, where the product pkernel is larger
#     padded = view(img, padindices(img, border(kernel))...)
#     pkernel = kernelconv(kernel...)
#     _imfilter_iter!(r, out, padded, pkernel, EdgeIterator(axes(out), iinds))
# end

### "Scheduler" methods (all with NoPad)

# These methods handle much of what Halide calls "the schedule."
# Together they handle the order-of-operations for separable and/or
# cascaded kernels, and even implement multithreadable tiling for FIR
# filtering.

function imfilter!(::AbstractResource, ::AbstractArray, ::AbstractArray, ::ProcessedKernel, border::AbstractBorder, ::Indices)
    error("Invalid border strategy `$border`: only `NoPad()` is supported.\n(You have called a stage of `imfilter`'s dispatch hierarchy after border-handling.)")
end

# Trivial kernel (a copy operation)
function imfilter!(r::AbstractResource, out::AbstractArray, A::AbstractArray, kernel::Tuple{}, ::NoPad, inds::Indices=axes(out))
    R = CartesianIndices(inds)
    copyto!(out, R, A, R)
end

# A single kernel
function imfilter!(r::AbstractResource, out::AbstractArray, A::AbstractArray, kernel::Tuple{Any}, border::NoPad, inds::Indices=axes(out))
    kern = kernel[1]
    iscopy(kern) && return imfilter!(r, out, A, (), border, inds)
    imfilter!(r, out, A, samedims(out, kern), border, inds)
end

# A filter cascade (2 or more filters)
function imfilter!(r::AbstractResource, out::AbstractArray, A::AbstractArray, kernel::Tuple{Any,Any,Vararg{Any}}, border::NoPad, inds::Indices=axes(out))
    kern = kernel[1]
    iscopy(kern) && return imfilter!(r, out, A, tail(kernel), border, inds)
    # For multiple stages of filtering, we introduce a second buffer
    # and swap them at each stage. The first of the two is the one
    # that holds the most recent result.
    A2 = tempbuffer(A, eltype(out), kernel)
    indsstep = shrink(expand(inds, calculate_padding(kernel)), kern)
    _imfilter!(r, out, A, A2, kernel, border, indsstep)
    return out
end

### Use a tiled algorithm for the cascaded case
function imfilter!(r::AbstractCPU{FIRTiled{N}}, out::AbstractArray{S,N}, A::AbstractArray{T,N}, kernel::Tuple{Any,Any,Vararg{Any}}, border::NoPad, inds::Indices=axes(out)) where {S,T,N}
    kern = kernel[1]
    iscopy(kern) && return imfilter!(r, out, A, tail(kernel), border, inds)
    tmp = tile_allocate(filter_type(A, kernel), r.settings.tilesize, kernel)
    _imfilter_tiled!(r, out, A, kernel, border, tmp, inds)
    out
end

### Scheduler support methods

## No tiling, filter cascade

# For these (internal) methods, `indsstep` refers to `inds` for this
# step of filtering, not the indices of `out` that we want to finally
# target.

# When `kernel` is (originally) a tuple that has both TriggsSdika and
# FIR filters, the overall padding gets doubled, yet we only trim off
# the minimum at each stage. Consequently, `indsstep` might be
# optimistic about the range available in `out`; therefore we use
# `intersect`.
function _imfilter!(r, out::AbstractArray, A1, A2, kernel::Tuple{}, border::NoPad, indsstep::Indices)
    imfilter!(r, out, A1, kernel, border, map(intersect, indsstep, axes(out)))
end

function _imfilter!(r, out::AbstractArray, A1, A2, kernel::Tuple{Any}, border::NoPad, indsstep::Indices)
    imfilter!(r, out, A1, kernel[1], border, map(intersect, indsstep, axes(out)))
end

# For IIR, it's important to filter over the whole passed-in range,
# and then copy! to out
function _imfilter!(r, out::AbstractArray, A1, A2, kernel::Tuple{AnyIIR}, border::NoPad, indsstep::Indices)
    if indsstep != axes(out)
        imfilter!(r, A2, A1, kernel[1], border, indsstep)
        R = CartesianIndices(map(intersect, indsstep, axes(out)))
        return copyto!(out, R, A2, R)
    end
    imfilter!(r, out, A1, kernel[1], border, indsstep)
end

function _imfilter!(r, out::AbstractArray, A1, A2::AbstractArray, kernel::Tuple{Any,Any,Vararg{Any}}, border::NoPad, indsstep::Indices)
    kern = kernel[1]
    iscopy(kern) && return _imfilter!(r, out, A1, A2, tail(kernel), border, indsstep)
    kernN = samedims(A2, kern)
    imfilter!(r, A2, A1, kernN, border, indsstep)  # store result in A2
    kernelt = tail(kernel)
    newinds = next_shrink(indsstep, kernelt)
    _imfilter!(r, out, A2, A1, tail(kernel), border, newinds)          # swap the buffers
end

function _imfilter!(r, out::AbstractArray, A1, A2::Tuple{AbstractArray,AbstractArray}, kernel::Tuple{Any,Any,Vararg{Any}}, border::NoPad, indsstep::Indices)
    kern = kernel[1]
    iscopy(kern) && return _imfilter!(r, out, A1, A2, tail(kernel), border, indsstep)
    A2_1, A2_2 = A2
    kernN = samedims(A2_1, kern)
    imfilter!(r, A2_1, A1, kernN, border, indsstep)  # store result in A2
    kernelt = tail(kernel)
    newinds = next_shrink(indsstep, kernelt)
    _imfilter!(r, out, A2_1, A2_2, tail(kernel), border, newinds)          # swap the buffers
end

# Single-threaded, pair of kernels (with only one temporary buffer required)
function _imfilter_tiled!(r::CPU1, out, A, kernel::Tuple{Any,Any}, border::NoPad, tiles::Vector{AA}, indsout) where {AA<:AbstractArray}
    k1, k2 = kernel
    tile = tiles[1]
    indsk2, indstile = axes(k2), axes(tile)
    sz = map(length, indstile)
    chunksz = map(length, shrink(indstile, indsk2))
    for tinds in TileIterator(indsout, chunksz)
        tileinds = expand(tinds, k2)
        tileb = TileBuffer(tile, tileinds)
        imfilter!(r, tileb.view, A, samedims(tileb, k1), border, tileinds)
        imfilter!(r, out, tileb.view, samedims(out, k2), border, tinds)
    end
    out
end

# Multithreaded, pair of kernels
function _imfilter_tiled!(r::CPUThreads, out, A, kernel::Tuple{Any,Any}, border::NoPad, tiles::Vector{AA}, indsout) where {AA<:AbstractArray}
    k1, k2 = kernel
    tile = tiles[1]
    indsk2, indstile = axes(k2), axes(tile)
    sz = map(length, indstile)
    chunksz = map(length, shrink(indstile, indsk2))
    tileinds_all = collect(expand(inds, k2) for inds in TileIterator(indsout, chunksz))
    _imfilter_tiled_threads!(CPU1(r), out, A, samedims(out, k1), samedims(out, k2), border, tileinds_all, tiles)
end
# This must be in a separate function due to #15276
@noinline function _imfilter_tiled_threads!(r1, out, A, k1, k2, border, tileinds_all, tile::Vector{AA}) where {AA<:AbstractArray}
    Threads.@threads for i = 1:length(tileinds_all)
        id = Threads.threadid()
        tileinds = tileinds_all[i]
        tileb = TileBuffer(tile[id], tileinds)
        imfilter!(r1, tileb, A, k1, border, tileinds)
        imfilter!(r1, out, tileb, k2, border, shrink(tileinds, k2))
    end
    out
end

# Single-threaded, multiple kernels (requires two tile buffers, swapping on each iteration)
function _imfilter_tiled!(r::CPU1, out, A, kernel::Tuple{Any,Any,Vararg{Any}}, border::NoPad, tiles::Vector{Tuple{AA,AA}}, indsout) where {AA<:AbstractArray}
    k1, kt = kernel[1], tail(kernel)
    tilepair = tiles[1]
    indstile = axes(tilepair[1])
    sz = map(length, indstile)
    chunksz = map(length, shrink(indstile, kt))
    for tinds in TileIterator(indsout, chunksz)
        tileinds = expand(tinds, kt)
        tileb1 = TileBuffer(tilepair[1], tileinds)
        imfilter!(r, tileb1, A, samedims(tileb1, k1), border, tileinds)
        _imfilter_tiled_swap!(r, out, kt, border, (tileb1, tilepair[2]))
    end
end

# Multithreaded, multiple kernels
function _imfilter_tiled!(r::CPUThreads, out, A, kernel::Tuple{Any,Any,Vararg{Any}}, border::NoPad, tiles::Vector{Tuple{AA,AA}}, indsout) where {AA<:AbstractArray}
    k1, kt = kernel[1], tail(kernel)
    tilepair = tiles[1]
    indstile = axes(tilepair[1])
    sz = map(length, indstile)
    chunksz = map(length, shrink(indstile, kt))
    tileinds_all = collect(expand(inds, kt) for inds in TileIterator(indsout, chunksz))
    _imfilter_tiled_threads!(CPU1(r), out, A, samedims(out, k1), kt, border, tileinds_all, tiles)
end
# This must be in a separate function due to #15276
@noinline function _imfilter_tiled_threads!(r1, out, A, k1, kt, border, tileinds_all, tiles::Vector{Tuple{AA,AA}}) where {AA<:AbstractArray}
    Threads.@threads for i = 1:length(tileinds_all)
        tileinds = tileinds_all[i]
        id = Threads.threadid()
        tile1, tile2 = tiles[id]
        tileb1 = TileBuffer(tile1, tileinds)
        imfilter!(r1, tileb1, A, k1, border, tileinds)
        _imfilter_tiled_swap!(r1, out, kt, border, (tileb1, tile2))
    end
    out
end

# The first of the pair in `tmp` has the current data. We also make
# the second a plain array so there's no doubt about who's holding the
# proper indices.
function _imfilter_tiled_swap!(r, out, kernel::Tuple{Any,Any,Vararg{Any}}, border, tmp::Tuple{TileBuffer,Array})
    tileb1, tile2 = tmp
    k1, kt = kernel[1], tail(kernel)
    parentinds = axes(tileb1)
    tileinds = shrink(parentinds, k1)
    tileb2 = TileBuffer(tile2, tileinds)
    imfilter!(r, tileb2, tileb1, samedims(tileb2, k1), border, tileinds)
    _imfilter_tiled_swap!(r, out, kt, border, (tileb2, tilebuf_parent(tileb1)))
end

# on the last call we write to `out` instead of one of the buffers
function _imfilter_tiled_swap!(r, out, kernel::Tuple{Any}, border, tmp::Tuple{TileBuffer,Array})
    tileb1 = tmp[1]
    k1 = kernel[1]
    parentinds = axes(tileb1)
    tileinds = shrink(parentinds, k1)
    imfilter!(r, out, tileb1, samedims(out, k1), border, tileinds)
end

### FIR filtering

"""
    imfilter!(::AbstractResource, imgfilt, img, kernel, NoPad(), [inds=axes(imgfilt)])

Filter an array `img` with kernel `kernel` by computing their
correlation, storing the result in `imgfilt`, defaulting to a finite-impulse
response (FIR) algorithm. Any necessary padding must have already been
supplied to `img`. If you want padding applied, instead call

    imfilter!([r::AbstractResource,] imgfilt, img, kernel, border)

with a specific `border`, or use

    imfilter!(imgfilt, img, kernel, [Algorithm.FIR()])

for default padding.

If `inds` is supplied, only the elements of `imgfilt` with indices in
the domain of `inds` will be calculated. This can be particularly
useful for "cascaded FIR filters" where you pad over a larger area and
then calculate the result over just the necessary/well-defined region
at each successive stage.

See also: [`imfilter`](@ref).
"""
function imfilter!(r::AbstractResource,
    out::AbstractArray{S,N},
    A::AbstractArray{T,N},
    kern::NDimKernel{N},
    border::NoPad,
    inds::Indices{N}=axes(out)) where {S,T,N}
    (isempty(A) || isempty(kern)) && return out
    indso, indsA, indsk = axes(out), axes(A), axes(kern)
    if iscopy(kern)
        R = CartesianIndices(inds)
        return copyto!(out, R, A, R)
    end
    for i = 1:N
        # Check that inds is inbounds for out
        indsi, indsoi, indsAi, indski = inds[i], indso[i], indsA[i], indsk[i]
        if first(indsi) < first(indsoi) || last(indsi) > last(indsoi)
            throw(DimensionMismatch("output indices $indso disagree with requested indices $inds"))
        end
        # Check that input A is big enough not to throw a BoundsError
        if first(indsAi) > first(indsi) + first(indski) ||
           last(indsA[i]) < last(indsi) + last(indski)
            throw(DimensionMismatch("requested indices $inds and kernel indices $indsk do not agree with indices of padded input, $indsA"))
        end
    end
    _imfilter_inbounds!(r, out, A, kern, border, inds)
end

function _imfilter_inbounds!(r::AbstractResource, out, A::AbstractArray, kern::ReshapedIIR, border::NoPad, inds)
    indspre, ind, indspost = iterdims(inds, kern)
    _imfilter_dim!(r, out, A, kern.data, indspre, ind, indspost, border[])
end

function _imfilter_inbounds!(r::AbstractResource, out, A::AbstractArray, kern, border::NoPad, inds)
    indsk = axes(kern)
    R, Rk = CartesianIndices(inds), CartesianIndices(indsk)
    if isempty(R) || isempty(Rk)
        return out
    end
    p = accumfilter(A[first(R)+first(Rk)], first(kern))
    z = zero(typeof(p + p))
    __imfilter_inbounds!(r, out, A, kern, border, R, z)
end

function __imfilter_inbounds!(r, out, A, kern, border, R, z)
    Rk = CartesianIndices(axes(kern))
    for I in safetail(R), i in safehead(R)
        tmp = z
        @inbounds for J in safetail(Rk), j in safehead(Rk)
            tmp += safe_for_prod(A[i+j, I+J], tmp) * kern[j, J]
        end
        @inbounds out[i, I] = tmp
    end
    out
end

# This is unfortunate, but specializing this saves an add in the inner
# loop and results in a modest performance improvement. It would be
# nice if LLVM did this automatically. (@polly?)
function __imfilter_inbounds!(r, out, A::OffsetArray, kern::OffsetArray, border, R, z)
    off, k = CartesianIndex(kern.offsets), parent(kern)
    o, O = safehead(off), safetail(off)
    Rnew = CartesianIndices(map((x, y) -> x .+ y, R.indices, Tuple(off)))
    Rk = CartesianIndices(axes(k))
    offA, pA = CartesianIndex(A.offsets), parent(A)
    oA, OA = safehead(offA), safetail(offA)
    for I in safetail(Rnew)
        IA = I - OA
        for i in safehead(Rnew)
            tmp = z
            iA = i - oA
            @inbounds for J in safetail(Rk), j in safehead(Rk)
                tmp += safe_for_prod(pA[iA+j, IA+J], tmp) * k[j, J]
            end
            @inbounds out[i-o, I-O] = tmp
        end
    end
    out
end

function _imfilter_inbounds!(r::AbstractResource, out, A::AbstractArray, kern::ReshapedOneD, border::NoPad, inds)
    Rpre, ind, Rpost = iterdims(inds, kern)
    k = kern.data
    R, Rk = CartesianIndices(inds), CartesianIndices(axes(kern))
    if isempty(R) || isempty(Rk)
        return out
    end
    p = accumfilter(A[first(R)+first(Rk)], first(k))
    z = zero(typeof(p + p))
    _imfilter_inbounds!(r, z, out, A, k, Rpre, ind, Rpost)
end

# Many of the following are unfortunate specializations
function _imfilter_inbounds!(r::AbstractResource, z, out, A::AbstractArray, k::OffsetVector, Rpre::CartesianIndices, ind, Rpost::CartesianIndices)
    _imfilter_inbounds!(r, z, out, A, parent(k), Rpre, ind, Rpost, k.offsets[1])
end

function _imfilter_inbounds!(r::AbstractResource, z, out, A::AbstractArray, k::AbstractVector, Rpre::CartesianIndices, ind, Rpost::CartesianIndices, koffset=0)
    indsk = axes(k, 1)
    for Ipost in Rpost
        for i in ind
            ik = i + koffset
            for Ipre in Rpre
                tmp = z
                for j in indsk
                    @inbounds tmp += safe_for_prod(A[Ipre, ik+j, Ipost], tmp) * k[j]
                end
                @inbounds out[Ipre, i, Ipost] = tmp
            end
        end
    end
    out
end

function _imfilter_inbounds!(r::AbstractResource, out, A::OffsetArray, kern::ReshapedVector, border::NoPad, inds)
    Rpre, ind, Rpost = iterdims(inds, kern)
    k = kern.data
    R, Rk = CartesianIndices(inds), CartesianIndices(axes(kern))
    if isempty(R) || isempty(Rk)
        return out
    end
    p = accumfilter(A[first(R)+first(Rk)], first(k))
    z = zero(typeof(p + p))
    Opre, o, Opost = KernelFactors.indexsplit(CartesianIndex(A.offsets), kern)
    _imfilter_inbounds!(r, z, out, parent(A), k, Rpre, ind, Rpost, Opre, o, Opost)
end

function _imfilter_inbounds!(r::AbstractResource, z, out, A::AbstractArray, k::OffsetVector, Rpre::CartesianIndices, ind, Rpost::CartesianIndices, Opre, o, Opost)
    _imfilter_inbounds!(r, z, out, A, parent(k), Rpre, ind, Rpost, Opre, o, Opost, k.offsets[1])
end

function _imfilter_inbounds!(r::AbstractResource, z, out, A::AbstractArray, k::AbstractVector, Rpre::CartesianIndices, ind, Rpost::CartesianIndices, Opre, o, Opost, koffset=0)
    indsk = axes(k, 1)
    for Ipost in Rpost
        IOpost = Ipost - Opost
        for i in ind
            io = i - o + koffset
            for Ipre in Rpre
                IOpre = Ipre - Opre
                tmp = z
                for j in indsk
                    @inbounds tmp += safe_for_prod(A[IOpre, io+j, IOpost], tmp) * k[j]
                end
                @inbounds out[Ipre, i, Ipost] = tmp
            end
        end
    end
    out
end
# end unfortunate specializations

## commented out because "virtual padding" is commented out
# function _imfilter_iter!(r::AbstractResource, out, padded, kernel::AbstractArray, iter)
#     p = padded[first(iter)] * first(kernel)
#     z = zero(typeof(p+p))
#     Rk = CartesianIndices(axes(kernel))
#     for I in iter
#         tmp = z
#         for J in Rk
#             @inbounds tmp += safe_for_prod(padded[I+J], tmp)*kernel[J]
#         end
#         out[I] = tmp
#     end
#     out
# end

# function _imfilter_iter!(r::AbstractResource, out, padded, kern::ReshapedOneD, iter)
#     Rpre, ind, Rpost = iterdims(axes(out), kern)
#     k = kern.data
#     indsk = axes(k, 1)
#     p = padded[first(iter)] * first(k)
#     TT = typeof(p+p)
#     for I in iter
#         Ipre, i, Ipost = KernelFactors.indexsplit(I, kern)
#         tmp = zero(TT) error("probably needs fixing")
#         @inbounds for j in indsk
#             tmp += safe_for_prod(padded[Ipre,i+j,Ipost], tmp)*k[j]
#         end
#         out[I] = tmp
#     end
#     out
# end



### FFT filtering

"""
    imfilter!(::AbstractResource{FFT}, imgfilt, img, kernel, NoPad())

Filter an array `img` with kernel `kernel` by computing their
correlation, storing the result in `imgfilt`, using a fast Fourier
transform (FFT) algorithm. Any necessary padding must have already
been applied to `img`. If you want padding applied, instead call

    imfilter!(::AbstractResource{FFT}, imgfilt, img, kernel, border)

with a specific `border`, or use

    imfilter!(imgfilt, img, kernel, Algorithm.FFT())

for default padding.

See also: [`imfilter`](@ref).
"""
function imfilter!(r::AbstractCPU{FFT},
    out::AbstractArray{S,N},
    img::AbstractArray{T,N},
    kernel::AbstractArray{K,N},
    border::NoPad) where {S,T,K,N}
    imfilter!(r, out, img, (kernel,), border)
end

function imfilter!(r::AbstractCPU{FFT},
    out::AbstractArray{S,N},
    A::AbstractArray{T,N},
    kernel::Tuple{AbstractArray},
    border::NoPad) where {S,T,N}
    _imfilter_fft!(r, out, A, kernel, border)  # ambiguity resolution
end
function imfilter!(r::AbstractCPU{FFT},
    out::AbstractArray{S,N},
    A::AbstractArray{T,N},
    kernel::Tuple{AbstractArray,Vararg{AbstractArray}},
    border::NoPad) where {S,T,N}
    _imfilter_fft!(r, out, A, kernel, border)
end

function _imfilter_fft!(r::AbstractCPU{FFT},
    out::AbstractArray{S,N},
    A::AbstractArray{T,N},
    kernel::Tuple{AbstractArray,Vararg{AbstractArray}},
    border::NoPad) where {S,T,N}
    kern = samedims(A, kernelconv(kernel...))
    krn = FFTView(zeros(eltype(kern), map(length, axes(A))))
    for I in CartesianIndices(axes(kern))
        krn[I] = kern[I]
    end
    Af = filtfft(A, krn)
    if map(first, axes(out)) == map(first, axes(Af))
        R = CartesianIndices(axes(out))
        copyto!(out, R, Af, R)
    else
        # Exploit the periodic boundary conditions of FFTView
        dest = FFTView(out)
        # src = OffsetArray(view(FFTView(Af), axes(dest)...), axes(dest))
        src = view(FFTView(Af), axes(dest)...)
        copyto!(dest, src)
    end
    out
end

function filtfft(A, krn)
    B = rfft(A)
    B .*= conj!(rfft(krn))
    irfft(B, length(axes(A, 1)))
end
function filtfft(A::AbstractArray{C}, krn) where {C<:Colorant}
    Av, dims = channelview_dims(A)
    kernrs = kreshape(C, krn)
    B = rfft(Av, dims)
    B .*= conj!(rfft(kernrs, dims))
    Avf = irfft(B, length(axes(Av, dims[1])), dims)
    colorview(base_colorant_type(C){eltype(Avf)}, Avf)
end
channelview_dims(A::AbstractArray{C,N}) where {C<:Colorant,N} = channelview(A), ntuple(d -> d + 1, Val(N))
channelview_dims(A::AbstractArray{C,N}) where {C<:ImageCore.Color1,N} = channelview(A), ntuple(identity, Val(N))

function kreshape(::Type{C}, krn::FFTView) where {C<:Colorant}
    kern = parent(krn)
    kernrs = FFTView(reshape(kern, 1, size(kern)...))
end
kreshape(::Type{C}, krn::FFTView) where {C<:ImageCore.Color1} = krn

### Triggs-Sdika (modified Young-van Vliet) recursive filtering
# B. Triggs and M. Sdika, "Boundary conditions for Young-van Vliet
# recursive filtering". IEEE Trans. on Sig. Proc. 54: 2365-2367
# (2006).

# Note this is safe for inplace use, i.e., out === img

function imfilter!(r::AbstractResource{IIR},
    out::AbstractArray{S,N},
    img::AbstractArray{T,N},
    kernel::Tuple{TriggsSdika,Vararg{TriggsSdika}},
    border::BorderSpec) where {S,T,N}
    isa(border, Pad) && border.style != :replicate && throw(ArgumentError("only \"replicate\" is supported"))
    length(kernel) <= N || throw(DimensionMismatch("cannot have more kernels than dimensions"))
    inds = axes(img)
    _imfilter_inplace_tuple!(r, out, img, kernel, CartesianIndices(()), inds, CartesianIndices(tail(inds)), border)
end

"""
    imfilter!(r::AbstractResource, imgfilt, img, kernel::Tuple{TriggsSdika...}, border)
    imfilter!(r::AbstractResource, imgfilt, img, kernel::TriggsSdika, dim::Integer, border)

Filter an array `img` with a Triggs-Sdika infinite impulse response
(IIR) `kernel`, storing the result in `imgfilt`. Unlike the `FIR` and
`FFT` algorithms, this version is safe for inplace operations, i.e.,
`imgfilt` can be the same array as `img`.

Either specify one kernel per dimension (as a tuple), or a particular
dimension `dim` along which to filter. If you exhaust `kernel`s before
you run out of array dimensions, the remaining dimension(s) will not
be filtered.

With Triggs-Sdika filtering, the only border options are `NA()`,
`"replicate"`, or `Fill(value)`.

See also: [`imfilter`](@ref), [`KernelFactors.TriggsSdika`](@ref), [`KernelFactors.IIRGaussian`](@ref).
"""
function imfilter!(r::AbstractResource, out::AbstractArray, img::AbstractArray, kernel::TriggsSdika, dim::Integer, border::BorderSpec)
    inds = axes(img)
    k, l = length(kernel.a), length(kernel.b)
    # This next part is not type-stable, which is why _imfilter_dim! has a @noinline
    Rbegin = CartesianIndices(inds[1:dim-1])
    Rend = CartesianIndices(inds[dim+1:end])
    _imfilter_dim!(r, out, img, kernel, Rbegin, inds[dim], Rend, border)
end
function imfilter!(r::AbstractResource, out::AbstractArray, img::AbstractArray, kernel::TriggsSdika, dim::Integer, border::AbstractString)
    imfilter!(r, out, img, kernel, dim, Pad(Symbol(border)))
end


function imfilter!(r::AbstractResource, out::AbstractArray, A::AbstractVector, kern::TriggsSdika, border::NoPad, inds::Indices=axes(out))
    indspre, ind, indspost = iterdims(inds, kern)
    _imfilter_dim!(r, out, A, kern, CartesianIndices(indspre), ind, CartesianIndices(indspost), border[])
end

# Lispy and type-stable inplace (currently just Triggs-Sdika) filtering over each dimension
function _imfilter_inplace_tuple!(r, out, img, kernel, Rbegin, inds, Rend, border)
    ind = first(inds)
    _imfilter_dim!(r, out, img, first(kernel), Rbegin, ind, Rend, border)
    _imfilter_inplace_tuple!(r,
        out,
        out,
        tail(kernel),
        CartesianIndices((Rbegin.indices..., ind)),
        tail(inds),
        _tail(Rend),
        border)
end
# When the final kernel has been used, return the output
_imfilter_inplace_tuple!(r, out, img, ::Tuple{}, Rbegin, inds, Rend, border) = out

# This is the "workhorse" function that performs Triggs-Sdika IIR
# filtering along a particular dimension. The "pre" dimensions are
# encoded in Rbegin, the "post" dimensions in Rend, and the dimension
# we're filtering is sandwiched between these. This design is
# type-stable and cache-friendly for any dimension---we update values
# in memory-order rather than along the chosen dimension. Nor does it
# require that the arrays have efficient linear indexing. For more
# information, see http://julialang.org/blog/2016/02/iteration.
@noinline function _imfilter_dim!(r::AbstractResource,
    out, img, kernel::TriggsSdika{T,k,l},
    Rbegin::CartesianIndices, ind::AbstractUnitRange,
    Rend::CartesianIndices, border::AbstractBorder) where {T,k,l}

    @noinline function throw_imfilter_dim(R, n, l)
        dim = ndims(R) + 1
        throw(DimensionMismatch("size $n of img along dimension $dim is too small for filtering with IIR kernel of length $l"))
    end

    if iscopy(kernel)
        if !(out === img)
            copyto!(out, img)
        end
        return out
    end
    if length(ind) <= max(k, l)
        throw_imfilter_dim(Rbegin, length(ind), max(k, l))
    end
    indleft = ind[firstindex(ind):firstindex(ind)+k-1]
    indright = ind[end-l+1:end]
    for Iend in Rend
        # Initialize the left border
        for Ibegin in Rbegin
            leftborder!(out, img, kernel, Ibegin, indleft, Iend, border)
        end
        # Propagate forwards. We omit the final point in case border
        # is "replicate", so that the original value is still
        # available. rightborder! will handle that point.
        for i = range(first(ind) + k, stop=ind[end-1])
            @inbounds for Ibegin in Rbegin
                tmp = accumfilter(img[Ibegin, i, Iend], one(T))
                for j = 1:k
                    tmp += kernel.a[j] * safe_for_prod(out[Ibegin, i-j, Iend], tmp)
                end
                out[Ibegin, i, Iend] = tmp
            end
        end
        # Initialize the right border
        for Ibegin in Rbegin
            rightborder!(out, img, kernel, Ibegin, indright, Iend, border)
        end
        # Propagate backwards
        for i = ind[end-l]:-1:first(ind)
            @inbounds for Ibegin in Rbegin
                tmp = accumfilter(out[Ibegin, i, Iend], one(T))
                for j = 1:l
                    tmp += kernel.b[j] * safe_for_prod(out[Ibegin, i+j, Iend], tmp)
                end
                out[Ibegin, i, Iend] = tmp
            end
        end
        # Final scaling
        for i in ind
            @inbounds for Ibegin in Rbegin
                out[Ibegin, i, Iend] *= kernel.scale
            end
        end
    end
    out
end

# Implements the initialization in the first paragraph of Triggs & Sdika, section II
function leftborder!(out, img, kernel, Ibegin, indleft, Iend, border::Fill)
    _leftborder!(out, img, kernel, Ibegin, indleft, Iend, convert(eltype(img), border.value))
end
function leftborder!(out, img, kernel, Ibegin, indleft, Iend, border::Pad)
    _leftborder!(out, img, kernel, Ibegin, indleft, Iend, img[Ibegin, indleft[1], Iend])
end
function _leftborder!(out, img, kernel::TriggsSdika{T,k,l}, Ibegin, indleft, Iend, iminus) where {T,k,l}
    uminus = iminus / (1 - kernel.asum)
    n = 0
    for i in indleft
        n += 1
        tmp = accumfilter(img[Ibegin, i, Iend], one(T))
        for j = 1:n-1
            tmp += kernel.a[j] * safe_for_prod(out[Ibegin, i-j, Iend], tmp)
        end
        for j = n:k
            tmp += kernel.a[j] * uminus
        end
        out[Ibegin, i, Iend] = tmp
    end
    out
end

# Implements Triggs & Sdika, Eqs 14-15
function rightborder!(out, img, kernel, Ibegin, indright, Iend, border::Fill)
    _rightborder!(out, img, kernel, Ibegin, indright, Iend, convert(eltype(img), border.value))
end
function rightborder!(out, img, kernel, Ibegin, indright, Iend, border::Pad)
    _rightborder!(out, img, kernel, Ibegin, indright, Iend, img[Ibegin, indright[end], Iend])
end
function _rightborder!(out, img, kernel::TriggsSdika{T,k,l}, Ibegin, indright, Iend, iplus) where {T,k,l}
    # The final value from forward-filtering was not calculated, so do that here
    i = last(indright)
    tmp = accumfilter(img[Ibegin, i, Iend], one(T))
    for j = 1:k
        tmp += kernel.a[j] * safe_for_prod(out[Ibegin, i-j, Iend], tmp)
    end
    out[Ibegin, i, Iend] = tmp
    # Initialize the v values at and beyond the right edge
    uplus = iplus / (1 - kernel.asum)
    vplus = uplus / (1 - kernel.bsum)
    vright = kernel.M * rightΔu(out, uplus, Ibegin, last(indright), Iend, kernel) .+ vplus
    out[Ibegin, last(indright), Iend] = vright[1]
    # Propagate inward
    n = 1
    for i in last(indright)-1:-1:first(indright)
        n += 1
        tmp = accumfilter(out[Ibegin, i, Iend], one(T))
        for j = 1:n-1
            tmp += kernel.b[j] * safe_for_prod(out[Ibegin, i+j, Iend], tmp)
        end
        for j = n:l
            tmp += kernel.b[j] * safe_for_prod(vright[j-n+2], tmp)
        end
        out[Ibegin, i, Iend] = tmp
    end
    out
end

# Part of Triggs & Sdika, Eq. 14
function rightΔu(img, uplus, Ibegin, i, Iend, kernel::TriggsSdika{T,3,l}) where {T,l}
    @inbounds ret = SVector(img[Ibegin, i, Iend] - uplus,
        img[Ibegin, i-1, Iend] - uplus,
        img[Ibegin, i-2, Iend] - uplus)
    ret
end

### NA boundary conditions

function imfilter_na_inseparable!(r, out::AbstractArray{T}, img, naflag, kernel::Tuple{Vararg{AnyIIR}}) where {T}
    fc, fn = Fill(zero(T)), Fill(zero(eltype(T)))  # color, numeric
    copyto!(out, img)
    out[naflag] .= zero(T)
    validpixels = copyto!(similar(Array{eltype(T)}, axes(img)), mappedarray(!, naflag))
    # TriggsSdika is safe for inplace operations
    imfilter!(r, out, out, kernel, fc)
    imfilter!(r, validpixels, validpixels, kernel, fn)
    for I in eachindex(out)
        out[I] /= validpixels[I]
    end
    out
end

function imfilter_na_inseparable!(r, out::AbstractArray{T}, img, naflag, kernel::Tuple) where {T}
    fc, fn = Fill(zero(T)), Fill(zero(eltype(T)))  # color, numeric
    imgtmp = copyto!(similar(out, axes(img)), img)
    imgtmp[naflag] .= Ref(zero(T))
    validpixels = copyto!(similar(Array{eltype(T)}, axes(img)), mappedarray(x -> !x, naflag))
    imfilter!(r, out, imgtmp, kernel, fc)
    vp = imfilter(r, validpixels, kernel, fn)
    for I in eachindex(out)
        out[I] /= vp[I]
    end
    out
end

function imfilter_na_separable!(r, out::AbstractArray{T}, img, kernel::Tuple) where {T}
    fc, fn = Fill(zero(T)), Fill(zero(eltype(T)))  # color, numeric
    imfilter!(r, out, img, kernel, fc)
    normalize_separable!(r, out, kernel, fn)
end

### Utilities

filter_type(img::AbstractArray{S}, kernel) where {S} = filter_type(S, kernel)

filter_type(::Type{S}, kernel::ArrayLike{T}) where {S,T} = typeof(zero(S) * zero(T) + zero(S) * zero(T))
filter_type(::Type{S}, ::Laplacian) where {S<:Union{Normed,FixedColorant}} = float32(S)
filter_type(::Type{S}, kernel::Laplacian) where {S<:Colorant} = S
filter_type(::Type{S}, ::Laplacian) where {S<:AbstractFloat} = S
filter_type(::Type{S}, ::Laplacian) where {S<:Signed} = S
filter_type(::Type{S}, ::Laplacian) where {S<:Unsigned} = signed_type(S)
filter_type(::Type{Bool}, ::Laplacian) = Int8

signed_type(::Type{UInt8}) = Int16
signed_type(::Type{UInt16}) = Int32
signed_type(::Type{UInt32}) = Int64
signed_type(::Type{T}) where {T<:Integer} = Int

@inline function filter_type(::Type{S}, kernel::Tuple{Any,Vararg{Any}}) where {S}
    T = filter_type(S, kernel[1])
    filter_type(T, S, tail(kernel))
end
@inline function filter_type(::Type{T}, ::Type{S}, kernel::Tuple{Any,Vararg{Any}}) where {T,S}
    Tnew = promote_type(T, filter_type(S, kernel[1]))
    filter_type(Tnew, S, tail(kernel))
end
filter_type(::Type{T}, ::Type{S}, kernel::Tuple{}) where {T,S} = T

factorkernel(kernel::AbstractArray) = (kernelshift(axes(kernel), kernel),)
factorkernel(L::Laplacian) = (L,)

function factorkernel(kernel::AbstractMatrix{T}) where {T}
    inds = axes(kernel)
    m, n = map(length, inds)
    kern = Array{T}(undef, m, n)
    copyto!(kern, 1:m, 1:n, kernel, inds[1], inds[2])
    factorstridedkernel(inds, kern)
end

function factorstridedkernel(inds, kernel::StridedMatrix)
    SVD = svd(kernel)
    U, S, Vt = SVD.U, SVD.S, SVD.Vt
    separable = true
    EPS = sqrt(eps(eltype(S)))
    for i = 2:length(S)
        separable &= (abs(S[i]) < EPS)
    end
    if !separable
        ks = kernelshift(inds, kernel)
        return (dummykernel(axes(ks)), ks)
    end
    s = S[1]
    u, v = U[:, 1:1], Vt[1:1, :]
    ss = sqrt(s)
    (kernelshift((inds[1], dummyind(inds[1])), ss * u),
        kernelshift((dummyind(inds[2]), inds[2]), ss * v))
end

kernelshift(inds::NTuple{N,Base.OneTo}, A::StridedArray) where {N} = _kernelshift(inds, A)
kernelshift(inds::NTuple{N,Base.OneTo}, A) where {N} = _kernelshift(inds, A)
function _kernelshift(inds, A)
    Base.depwarn("assuming that the origin is at the center of the kernel; to avoid this warning, call `centered(kernel)` or use an OffsetArray", :_kernelshift)  # this may be necessary long-term?
    centered(A)
end
kernelshift(inds::Indices, A::StridedArray) = OffsetArray(A, inds...)
function kernelshift(inds::Indices, A)
    @assert axes(A) == inds
    A
end

# Note this is not type-stable. Fortunately, all the outputs are
# allocated by the time this gets called.
function filter_algorithm(out, img, kernel::Union{ArrayType,Tuple{Vararg{ArrayType}}})
    L = maxlen(kernel)
    if L > 30 && eltype(img) <: Union{Number,Colorant} && all(isfinite, img)
        return FFT()
    end
    sz = map(length, calculate_padding(kernel))
    isa(kernel, Tuple) && length(kernel) > 1 ? FIRTiled(padded_tilesize(eltype(out), sz)) : FIR()
end
filter_algorithm(out, img, kernel::Tuple{AnyIIR,Vararg{AnyIIR}}) = IIR()
filter_algorithm(out, img, kernel) = Mixed()

maxlen(A::AbstractArray) = length(A)
@inline maxlen(kernel::Tuple) = _maxlen(0, kernel...)
_maxlen(len, kernel1, kernel...) = _maxlen(max(len, length(kernel1)), kernel...)
_maxlen(len) = len

Base.length(A::ReshapedVector) = length(A.data)

isseparable(kernels::Tuple{Vararg{AnyIIR}}) = true
isseparable(kernels::Tuple) = all(x -> nextendeddims(x) == 1, kernels)

normalize_separable!(r::AbstractResource, A, ::Tuple{}, border) = error("this shouldn't happen")
function normalize_separable!(r::AbstractResource, A, kernels::NTuple{N,TriggsSdika}, border) where {N}
    inds = axes(A)
    function imfilter_inplace!(r, a, kern, border)
        imfilter!(r, a, a, (kern,), border)
    end
    filtdims = ntuple(d -> imfilter_inplace!(r, fill(1.0, inds[d]), kernels[d], border), Val(N))
    normalize_dims!(A, filtdims)
end
function normalize_separable!(r::AbstractResource, A, kernels::NTuple{N,ReshapedIIR}, border) where {N}
    normalize_separable!(r, A, map(_vec, kernels), border)
end

function normalize_separable!(r::AbstractResource, A, kernels::NTuple{N,Any}, border) where {N}
    inds = axes(A)
    # some kernels require floats here
    filtdims = ntuple(d -> imfilter(r, fill(1.0, inds[d]), _vec(kernels[d]), border), Val(N))
    normalize_dims!(A, filtdims)
end

function normalize_dims!(A::AbstractArray{T,N}, factors::NTuple{N}) where {T,N}
    for I in CartesianIndices(axes(A))
        tmp = A[I] / factors[1][I[1]]
        for d = 2:N
            tmp /= factors[d][I[d]]
        end
        A[I] = tmp
    end
    A
end

iscopy(kernel::AbstractArray) = all(x -> x == 0:0, axes(kernel)) && first(kernel) == 1
iscopy(kernel::Laplacian) = false
iscopy(kernel::TriggsSdika) = all(x -> x == 0, kernel.a) && all(x -> x == 0, kernel.b) && kernel.scale == 1
iscopy(kernel::ReshapedOneD) = iscopy(kernel.data)

kernelconv(kernel) = kernel
function kernelconv(k1, k2, kernels...)
    out = similar(Array{filter_type(eltype(k1), k2)}, calculate_padding((k1, k2)))
    fill!(out, zero(eltype(out)))
    k1N, k2N = samedims(out, k1), samedims(out, k2)
    R1, R2 = CartesianIndices(axes(k1N)), CartesianIndices(axes(k2N))
    ref = accumfilter(zero(eltype(k1)), zero(eltype(k2)))
    for I1 in R1
        for I2 in R2
            out[I1+I2] += safe_for_prod(k1N[I1], ref) * k2N[I2]
        end
    end
    kernelconv(out, kernels...)
end

default_resource(alg::FIRTiled) = Threads.nthreads() > 1 ? CPUThreads(alg) : CPU1(alg)
default_resource(alg) = CPU1(alg)

function alg_defaults(alg::FIRTiled{0}, out, kernel)
    sz = map(length, calculate_padding(kernel))
    FIRTiled(padded_tilesize(eltype(out), sz))
end
alg_defaults(alg::Alg, out, kernel) = alg


## Faster Cartesian iteration
# Splitting out the first dimension saves a branch
safetail(R::CartesianIndices) = CartesianIndices(tail(R.indices))
safetail(R::CartesianIndices{1}) = CartesianIndices(())
safetail(R::CartesianIndices{0}) = CartesianIndices(())
safetail(I::CartesianIndex) = CartesianIndex(tail(Tuple(I)))
safetail(::CartesianIndex{1}) = CartesianIndex(())
safetail(::CartesianIndex{0}) = CartesianIndex(())

safehead(R::CartesianIndices) = R.indices[1]
safehead(R::CartesianIndices{0}) = CartesianIndices(())
safehead(I::CartesianIndex) = I[1]
safehead(::CartesianIndex{0}) = CartesianIndex(())

## Tiling utilities

function tile_allocate(::Type{T}, kernel) where {T}
    sz = map(length, calculate_padding(kernel))
    tsz = padded_tilesize(T, sz)
    tile_allocate(T, tsz, kernel)
end

function tile_allocate(::Type{T}, tsz::Dims, kernel::Tuple{Any,Any}) where {T}
    # Allocate a single tile for a 2-stage filter
    [Array{T}(undef, tsz) for i = 1:Threads.nthreads()]
end

function tile_allocate(::Type{T}, tsz::Dims, kernel::Tuple{Any,Any,Vararg{Any}}) where {T}
    # Allocate a pair of tiles and swap buffers at each stage
    [(Array{T}(undef, tsz), Array{T}(undef, tsz)) for i = 1:Threads.nthreads()]
end

## Whole-image temporaries
const fillbuf_nan = Ref(false)  # used only for testing purposes

function tempbuffer(A::AbstractArray, ::Type{T}, kernel::Tuple{Any,Any}) where {T}
    A2 = similar(A, T)
    if fillbuf_nan[]
        fill!(A2, NaN)  # for testing purposes
    end
    A2
end

# If there are more than two stages of filtering, we need two
# temporaries to avoid overwriting the input image
function tempbuffer(A::AbstractArray, ::Type{T}, kernel::Tuple{Any,Any,Any,Vararg{Any}}) where {T}
    (similar(A, T), similar(A, T))
end
