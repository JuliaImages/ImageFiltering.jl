module Models

using ImageBase
using ImageBase.ImageCore.MappedArrays: of_eltype
using ImageBase.FiniteDiff: fdiv!, fdiff!, fgradient!

# Introduced in ColorVectorSpace v0.9.3
# https://github.com/JuliaGraphics/ColorVectorSpace.jl/pull/172
using ImageBase.ImageCore.ColorVectorSpace.Future: abs2

using ..ImageFiltering: ffteltype, freqkernel
using FFTW

"""
This submodule provides predefined image-related models and its solvers that can be reused
by many image processing tasks.

- solve the Rudin Osher Fatemi (ROF) model using the primal-dual method: [`solve_ROF_PD`](@ref) and [`solve_ROF_PD!`](@ref)
- solve the Rudin Osher Fatemi (ROF) model using the ADMM method: [`solve_ROF_ADMM`](@ref) and [`solve_ROF_ADMM!`](@ref)
"""
Models

export solve_ROF_PD, solve_ROF_PD!,
    solve_ROF_ADMM, solve_ROF_ADMM!


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

"""
    solve_ROF_ADMM([T], img::AbstractArray, λ; kwargs...)

Return a smoothed version of `img`, using Rudin-Osher-Fatemi (ROF) filtering, more commonly
known as Total Variation (TV) denoising or TV regularization. This algorithm is based on the
alternating direction method of multipliers (ADMM) method; it is also called the split
bregman method.

See also [`solve_ROF_PD`](@ref) for the primal dual method.

# References

- [1] Goldstein, Tom, and Stanley Osher. "The split Bregman method for L1-regularized problems." _SIAM journal on imaging sciences_ 2.2 (2009): 323-343.
- [2] Getreuer, Pascal. "Rudin-Osher-Fatemi total variation denoising using split Bregman." _Image Processing On Line 2_ (2012): 74-95.
"""
solve_ROF_ADMM(img::AbstractArray{T}, args...; kwargs...) where T = solve_ROF_ADMM(float32(T), img, args...; kwargs...)

function solve_ROF_ADMM(::Type{T}, img, args...; kwargs...) where T
    # TODO(johnnychen94): support generic images
    out = similar(img, T)
    buffer = preallocate_solve_ROF_ADMM(T, img)
    solve_ROF_ADMM!(out, buffer, img, args...; kwargs...)
end

# non-exported helper
preallocate_solve_ROF_ADMM(img::AbstractArray{T}) where T = preallocate_solve_ROF_ADMM(float32(T), img)
function preallocate_solve_ROF_ADMM(::Type{T}, img) where T
    # Use similar to allow construct CuArray when `img` is a CuArray

    # split variable for ∇
    d = ntuple(i->fill!(similar(img, T), zero(T)), ndims(img))
    # dual variable
    b = map(copy, d)

    # precomputed the negative Laplacian kernel in frequency space
    DTD = _negative_flaplacian_freqkernel(img)

    # buffer for u subproblem
    #   minᵤ 0.5μ||u-f||₂² + 0.5∑ᵢ(λ||dᵢ - ∇ᵢu - bᵢ||₂²)
    # For generality to arbitrary dimension, fft-based solver is applied
    #   u = real.(ifft(fft(RHS)./fft(LHS)))
    # where LHS is actually a convolution operator thus we can apply convolution theorem.
    Δbd = similar(first(b))
    LHS = similar(first(d)) # left-hand side: μI + Δ
    RHS = similar(LHS) # right-hand side: μf + λ∑ᵢ(∇ᵢ'(dᵢ - bᵢ))
    fft_tmp = similar(LHS, Complex{eltype(T)})

    return d, b, DTD, Δbd, LHS, RHS, fft_tmp
end

"""
    solve_ROF_ADMM!(out, buffer, img, λ, μ; kwargs...)

The in-place version of [`solve_ROF_ADMM`](@ref).
"""
function solve_ROF_ADMM!(out::AbstractArray{T}, buffer, img, λ, μ; kwargs...) where T
    # seperate a stub method to reduce latency
    FT = float32(T)
    if FT == T
        _solve_ROF_ADMM_anisotropic!(out, buffer, img, Float32(λ), Float32(μ); kwargs...)
    else
        _solve_ROF_ADMM_anisotropic!(out, buffer, FT.(img), Float32(λ), Float32(μ); kwargs...)
    end
end

function _solve_ROF_ADMM_anisotropic!(
        out::AbstractArray,
        (d, b, DTD, Δbd, LHS, RHS, fft_tmp)::Tuple,
        img::AbstractArray,
        λ::Float32, μ::Float32; num_iters::Integer=200)
    # The notation and algorithm follows reference [1] with one modification:
    u, f = out, img

    # u-subproblem is solved using direct fft-based method instead of using Gauss–Seidel
    # method. This is because Gauss–Seidel method itself is a solver for 2-dimensional case,
    # while FFT-based solver support arbitrary dimensional. However, we might still want to
    # provide a specialization solver for the 2D case using Gauss-Seidel method and see if
    # it helps speaking of performance.

    # soft-thresholding: d-subproblem is a simplified lasso problem
    S(x, λ) = sign(x) * max(abs(x)-λ, zero(x))

    # initialization
    foreach(x->fill!(x, zero(eltype(x))), d)
    foreach(x->fill!(x, zero(eltype(x))), b)
    @. LHS = λ + μ * DTD
    # apply the algorithm below Eq. (4.2)
    for _ in 1:num_iters
        @. RHS = λ/μ * f
        for i in 1:length(d)
            # negative adjoint gradient is the reverse finite difference
            @. Δbd = b[i] - d[i]
            # reuse d here as buffer as it will be updated later in d-subproblem
            RHS .+= fdiff!(d[i], Δbd; dims=i, rev=true)
        end
        # TODO(johnnychen94): optimize the memory allocation
        fft_tmp .= μ .* fft(RHS)./LHS
        u .= real.(ifft(fft_tmp))

        # for anisotropic problem, the update of d[i] is decoupled
        du = RHS # reuse RHS as buffer for d-subproblem
        for i in 1:length(d)
            fdiff!(du, u; dims=i)
            # 2) solve d subproblem
            @. d[i] = S(du + b[i], 1/μ)
            # 3) update dual variable b
            @. b[i] = b[i] + du - d[i]
        end
    end
    return u
end

# generates -fft(Δ) = -∑fft(∇ᵢ')(∇ᵢ) of size `size(X)` in the frequency domain
function _negative_flaplacian_freqkernel(X::AbstractArray{T}) where T<:Real
    FΔ = fill!(similar(X), zero(ffteltype(T)))
    for i in 1:ndims(X)
        # forward diff
        ∇ᵢ = reshape([1, -1], ntuple(j->j==i ? 2 : 1, ndims(X)))
        # TODO(johnnychen94): reuse the allocated memory
        F∇ᵢ = freqkernel(∇ᵢ, size(X))
        @. FΔ += Base.abs2(F∇ᵢ)
    end
    return FΔ
end

end # module
