module Models

using ImageBase
using ImageBase.ImageCore.MappedArrays: of_eltype
using ImageBase.FiniteDiff

"""
This submodule provides predefined image-related models and its solvers that can be reused
by many image processing tasks.

- solve the Rudin Osher Fatemi (ROF) model using the primal-dual method: [`solve_ROF_PD`](@ref) and [`solve_ROF_PD!`](@ref)
"""
Models

export solve_ROF_PD, solve_ROF_PD!


##### implementation details

"""
    solve_ROF_PD([T], img::AbstractArray, λ; kwargs...)

Return a smoothed version of `img`, using Rudin-Osher-Fatemi (ROF) filtering, more commonly
known as Total Variation (TV) denoising or TV regularization. This algorithm is based on the
primal-dual method.

This function applies to generic N-dimensional colorant array and is also CUDA-compatible.
See also [`solve_ROF_PD!`](@ref) for the in-place version.

# Arguments

- `T`: the output element type. By default it is `float32(eltype(img))`.
- `img`: the input image, usually a noisy image.
- `λ`: the regularization coefficient. Larger `λ` results in more smoothing.

# Parameters

- `num_iters::Int`: The number of iterations before stopping.

# Examples

```julia
using ImageFiltering
using ImageFiltering.Models: solve_ROF_PD
using ImageQualityIndexes
using TestImages

img_ori = float.(testimage("cameraman"))
img_noisy = img_ori .+ 0.1 .* randn(size(img_ori))
assess_psnr(img_noisy, img_ori) # ~20 dB

img_smoothed = solve_ROF_PD(img_noisy, 0.015, 50)
assess_psnr(img_smoothed, img_ori) # ~27 dB

# larger λ produces over-smoothed result
img_smoothed = solve_ROF_PD(img_noisy, 5, 50)
assess_psnr(img_smoothed, img_ori) # ~21 dB
```

# Extended help

Mathematically, this function solves the following ROF model using the primal-dual method:

```math
\\min_u \\lVert u - g \\rVert^2 + \\lambda\\lvert\\nabla u\\rvert
```

# References

- [1] Chambolle, A. (2004). "An algorithm for total variation minimization and applications". _Journal of Mathematical Imaging and Vision_. 20: 89–97
- [2] [Wikipedia: Total Variation Denoising](https://en.wikipedia.org/wiki/Total_variation_denoising)
"""
solve_ROF_PD(img::AbstractArray{T}, args...) where T = solve_ROF_PD(float32(T), img, args...)
function solve_ROF_PD(::Type{T}, img::AbstractArray, args...) where T
    u = similar(img, T)
    buffer = preallocate_solve_ROF_PD(T, img)
    solve_ROF_PD!(u, buffer, img, args...)
end

# non-exported helper
preallocate_solve_ROF_PD(img::AbstractArray{T}) where T = preallocate_solve_ROF_PD(float32(T), img)
function preallocate_solve_ROF_PD(::Type{T}, img) where T
    div_p = similar(img, T)
    p = ntuple(i->similar(img, T), ndims(img))
    ∇u = ntuple(i->similar(img, T), ndims(img))
    ∇u_mag = similar(img, eltype(T))
    return div_p, p, ∇u, ∇u_mag
end

"""
    solve_ROF_PD!(out, buffer, img, λ, num_iters)

The in-place version of [`solve_ROF_PD`](@ref).

It is not uncommon to use ROF solver in a higher-level loop, in which case it makes sense to
preallocate the output and intermediate arrays to make it faster.

!!! note "Buffer"
    The content and meaning of `buffer` might change without any notice if the internal
    implementation is changed. Use `preallocate_solve_ROF_PD` helper function to avoid
    potential changes.

# Examples

```julia
using ImageFiltering.Models: preallocate_solve_ROF_PD

out = similar(img)
buffer = preallocate_solve_ROF_PD(img)
solve_ROF_PD!(out, buffer, img, 0.2, 30)
```

"""
function solve_ROF_PD!(
        out::AbstractArray{T},
        buffer::Tuple,
        img::AbstractArray,
        λ::Real,
        num_iters::Integer) where T
    # seperate a stub method to reduce latency
    FT = float32(T)
    if FT == T
        solve_ROF_PD!(out, buffer, img, Float32(λ), Int(num_iters))
    else
        solve_ROF_PD!(out, buffer, FT.(img), Float32(λ), Int(num_iters))
    end
end
function solve_ROF_PD!(
        out::AbstractArray,
        (div_p, p, ∇u, ∇u_mag)::Tuple,
        img::AbstractArray,
        λ::Float32,
        num_iters::Int)
    # Total Variation regularized image denoising using the primal dual algorithm
    # Implement according to reference [1]
    τ = 1//4 # see 2nd remark after proof of Theorem 3.1.

    # use the same symbol in the paper
    u, g = out, img

    fgradient!(p, g)
    # This iterates Eq. (9) of [1]
    # TODO(johnnychen94): set better stop criterion
    for _ in 1:num_iters
        fdiv!(div_p, p)
        # multiply term inside ∇ by -λ. Thm. 3.1 relates this to `u` via Eq. 7.
        @. u = g - λ*div_p
        fgradient!(∇u, u)
        _l2norm_vec!(∇u_mag, ∇u) # |∇(g - λdiv p)|
        # Eq. (9): update p
        for i in 1:length(p)
            @. p[i] = (p[i] - (τ/λ)*∇u[i])/(1 + (τ/λ) * ∇u_mag)
        end
    end
    return u
end

function _l2norm_vec!(out, Vs::Tuple)
    all(v->axes(out) == axes(v), Vs) || throw(ArgumentError("All axes of input data should be the same."))
    @. out = abs2(Vs[1])
    for v in Vs[2:end]
        @. out += abs2(v)
    end
    @. out = sqrt(out)
    return out
end


end # module
