module Models

using ImageBase
using ImageBase.ImageCore.MappedArrays: of_eltype
using ImageBase.FiniteDiff

# Introduced in ColorVectorSpace v0.9.3
# https://github.com/JuliaGraphics/ColorVectorSpace.jl/pull/172
using ImageBase.ImageCore.ColorVectorSpace.Future: abs2

"""
This submodule provides predefined image-related models and its solvers that can be reused
by many image processing tasks.

- solve the Rudin Osher Fatemi (ROF) model using the primal-dual method: [`solve_ROF_PD`](@ref)
"""
Models

export solve_ROF_PD


##### implementation details

"""
    solve_ROF_PD(img：：AbstractArray, λ; kwargs...)

Perform Rudin-Osher-Fatemi (ROF) filtering, more commonly known as Total Variation (TV)
denoising or TV regularization. This algorithm is based on the primal-dual method.

This function applies to generic n-dimensional colorant array and is also CUDA-compatible.

# Arguments

- `img`: the input image, usually is a noisy image.
- `λ`: the regularization coefficient. Larger `λ` would produce more smooth image.

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
- [2] https://en.wikipedia.org/wiki/Total_variation_denoising
"""
function solve_ROF_PD(img::AbstractArray, λ::Real, num_iters::Integer)
    # Total Variation regularized image denoising using the primal dual algorithm
    # Implement according to reference [1]
    τ = 1/4   # see 2nd remark after proof of Theorem 3.1.

    g = of_eltype(floattype(eltype(img)), img) # use the same symbol in the paper
    u = similar(g)
    p = fgradient(g)
    div_p = similar(g)
    ∇u = map(similar, p)
    ∇u_mag = similar(g, eltype(eltype(g)))

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
