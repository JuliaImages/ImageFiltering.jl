using ImageFiltering, ImageCore, OffsetArrays, ComputationalResources, StaticArrays
using ImageFiltering.KernelFactors: SlidingKernel, SlidingDCT, SlidingDST, SeparableSum, ReshapedOneD,
    DCTGaussian, slidinggaussian, DCTLoG, DCTDoG
using Random, Test

# The same kernel as a plain FIR array, to compare against the sliding algorithm
firkernel(k::SlidingKernel) = OffsetArray(collect(k), axes(k))
firkernel(k::ReshapedOneD{T,N,Npre}) where {T,N,Npre} = ReshapedOneD{N,Npre}(firkernel(k.data))
firkernel(ks::Tuple) = map(firkernel, ks)
firkernel(L::SeparableSum) = SeparableSum(map(firkernel, L.terms), L.weights)
firkernel(k) = k

## Tolerances for comparing two computations of the same filter (sliding vs. FIR)
#
# We require |y - y_ref| ≤ atol + rtol |y_ref| for every pixel and color channel, with
# rtol = 4eps(T) in the precision T of the computation (the lower of the precisions of
# the result and of the kernel coefficients). The absolute term is the round-off
# of the computation, which does not scale with |y_ref| (derivative outputs are small
# because large terms cancel). For an input of magnitude X and a kernel
# h[r] = c₀ + Σₖ cₖ cos(ωₖ r) on -N:N, each resonator of the recurrence carries a windowed
# sum of size |cₖ|(2N+1)X and amplifies its rounding errors by up to 1/|sin ωₖ| (by up to
# n for ωₖ = π, a double pole), and these errors accumulate as a random walk, ∝ √n,
# along a line of length n. The FIR reference adds (2N+1)‖h‖₁X. In a cascade, the error
# of each stage is scaled by the ℓ¹ norms of the other stages. With atol = 4eps(T) X
# times this growth factor, the observed errors are at least 4 times below the bound
# (checked up to n = 10⁶ for Float32 and Float64). Fixed-point results may in addition
# round differently, by one quantum.
kerneldata(k::ReshapedOneD) = k.data
kerneldata(k) = k
filterdim(::ReshapedOneD{T,N,Npre}) where {T,N,Npre} = Npre + 1
filterdim(_) = 1
function stagegrowth(k::SlidingKernel, n)
    N = k.N
    resonators = abs(KernelFactors.dccoef(k)) * (2N + 1) +
                 sum(((c, ω),) -> abs(c) * (2N + 1) * min(1 / abs(sin(ω)), n), zip(k.c, k.ω); init = 0.0)
    return resonators * sqrt(n) + (2N + 1) * sum(abs, k)
end
stagegrowth(k::AbstractArray, n) = length(k) * sum(abs, k)   # an FIR stage
function growth(img, ks::Tuple)
    stages = map(kerneldata, ks)
    norms = map(k -> sum(abs, k), stages)
    return sum(eachindex(stages)) do s
        n = size(img, filterdim(ks[s])) + length(stages[s])   # the (virtually) padded line
        stagegrowth(stages[s], n) * prod(i -> i == s ? 1.0 : norms[i], eachindex(norms))
    end
end
growth(img, k::SlidingKernel) = growth(img, (k,))
growth(img, L::SeparableSum) = sum(((t, w),) -> abs(w) * growth(img, t), zip(L.terms, L.weights))
kerneltypes(ks::Tuple) = map(k -> eltype(kerneldata(k)), ks)
kerneltypes(L::SeparableSum) = reduce((a, b) -> (a..., b...), map(kerneltypes, L.terms))
kerneltypes(k) = (eltype(k),)
# (all the element types in these tests are exactly representable as Float64)
magnitude(x) = maximum(v -> isfinite(Float64(v)) ? abs(Float64(v)) : 0.0, channelview(x); init = 0.0)
function filtertol(img, ks, border, out)
    S = eltype(eltype(out))
    ϵ = maximum(T -> eps(float(T)), (S, kerneltypes(ks)...))
    X = max(magnitude(img), border isa Fill ? magnitude([border.value]) : 0.0)
    atol = 4 * ϵ * X * growth(img, ks)
    # NA() divides by the filtered mask (≥ 2^-D near corners) and filters twice
    border isa NA && (atol *= 2^(ndims(img) + 1))
    S <: FixedPoint && (atol += eps(S))
    return atol, 4ϵ
end
function withintol(out, ref, atol, rtol)
    axes(out) == axes(ref) || return false
    return all(((a, b),) -> abs(Float64(a) - Float64(b)) <= atol + rtol * abs(Float64(b)), zip(channelview(out), channelview(ref)))
end
# `out` is the result of filtering `img` with `ks`, up to rounding, if `ref` is the exact one
filtersmatch(out, ref, img, ks, border = "replicate") = withintol(out, ref, filtertol(img, ks, border, out)...)
# Filtering `img` with the sliding kernels `ks` gives the result of direct (FIR) filtering
# with the same kernels; if FIR filtering throws (e.g., "reflect" on an axis of length 1,
# or Inner() with a kernel longer than the image), it throws the same error
function agrees(img, ks, border = "replicate")
    ref = try
        imfilter(img, firkernel(ks), border, Algorithm.FIR())
    catch err
        return try imfilter(img, ks, border); false catch err2 typeof(err2) == typeof(err) end
    end
    out = imfilter(img, ks, border)
    return eltype(out) == eltype(ref) && filtersmatch(out, ref, img, ks, border)
end
fir(img, ks, args...) = imfilter(img, firkernel(ks), args..., Algorithm.FIR())
# The floating-point type of the pixels of an image (Float32 for N0f8); kernels of this type
# keep the computation in it, whereas Float64 kernels (the default) promote it to Float64
pixelfloat(img) = float(eltype(eltype(img)))

# Peak signal-to-noise ratio for images with values in [0, 1]
psnr(a, ref) = 10 * log10(length(ref) / sum(abs2, a .- ref))
# The untruncated sampled Gaussian, normalized
function sampledgaussian(σ, M = ceil(Int, 12σ))
    g = OffsetArray([exp(-r^2 / (2σ^2)) for r in -M:M], -M:M)
    return g ./ sum(g)
end
# ... and its derivatives, as correlation kernels (imfilter computes the derivative)
function sampledderivative(σ, order::Integer, M = ceil(Int, 12σ))
    g = sampledgaussian(σ, M)
    order == 0 && return g
    order == 1 && return OffsetArray([r / σ^2 * g[r] for r in -M:M], -M:M)
    return OffsetArray([(r^2 - σ^2) / σ^4 * g[r] for r in -M:M], -M:M)
end
# relative L2 error of a kernel with respect to a reference
l2err(k, ref) = sqrt(sum(r -> abs2((r in axes(k, 1) ? k[r] : 0.0) - ref[r]), axes(ref, 1)) / sum(abs2, ref))

@testset "Sliding DCT/DST" begin
    Random.seed!(1234)
    @testset "kernel" begin
        for σ in (1.0, 2.5, 7.3, 30.0), K in 1:6, kind in (1, 3, 5, 7)
            k = DCTGaussian(σ; K = K, kind = kind)
            N = k.N
            @test axes(k) == (-N:N,)
            @test size(k) == (2N + 1,) && length(k) == 2N + 1
            @test eltype(k) == Float64
            @test all(r -> k[r] ≈ k[-r], 1:N)
            @test_throws BoundsError k[N+1]
            # the sum is preserved; and so is the variance for K ≥ 2
            @test sum(k) ≈ 1
            if K >= 2
                @test sum(r -> r^2 * k[r], -N:N) ≈ σ^2 rtol = 1e-6
            end
            # The DC term is stored separately
            @test (kind ∈ (1, 5)) == !iszero(k.c0)
            @test length(k.c) == (kind ∈ (1, 5) ? K - 1 : K)
        end
        # Accuracy improves with K (relative to the peak of an untruncated sampled Gaussian)
        σ = 10.0
        g = sampledgaussian(σ)
        kerr(k, p) = (r = axes(g, 1); e = [(abs(i) <= k.N ? k[i] : 0.0) - g[i] for i in r]; p == Inf ? maximum(abs, e) / g[0] : sqrt(sum(abs2, e) / sum(abs2, g)))
        errs = [kerr(DCTGaussian(σ; K = K), Inf) for K in 3:7]
        @test issorted(errs; rev = true)
        @test errs[2] < 3e-3      # K = 4, the default
        @test errs[end] < 3e-5    # K = 7
        @test kerr(DCTGaussian(σ), Inf) == errs[2]
        # The default is DCT-5: 3 cosines and the constant term
        @test DCTGaussian(3.0) isa SlidingDCT{Float64,3,true}
        # The default radius minimizes the L2 error
        for σ in (1.5, 3.0, 10.0), K in (2, 3, 5), kind in (5, 7)
            g = sampledgaussian(σ)
            k = DCTGaussian(σ; K = K, kind = kind, emit_warning = false)
            @test all(N -> kerr(DCTGaussian(σ; K = K, N = N, kind = kind), 2) >= kerr(k, 2), max(K, k.N - 5):k.N+5)
        end

        # Element types
        @test eltype(DCTGaussian(3)) == Float64
        @test eltype(DCTGaussian(3.0f0)) == Float32
        @test eltype(DCTGaussian(Float32, 3.0)) == Float32
        @test DCTGaussian(Float32, 3.0) ≈ DCTGaussian(3.0)
        # Tuples
        kx, ky = DCTGaussian((2.0, 5.0))
        @test kx isa ReshapedOneD{Float64,2,0} && ky isa ReshapedOneD{Float64,2,1}
        @test kx.data == DCTGaussian(2.0) && ky.data == DCTGaussian(5.0)
        @test DCTGaussian([2.0, 5.0]) == (kx, ky)
        @test DCTGaussian((2.0, 5.0); N = (9, 20))[2].data.N == 20
        @test DCTGaussian((2.0, 5.0); N = 20)[1].data.N == 20
        @test DCTGaussian((2.0, 5.0); K = 5)[1].data == DCTGaussian(2.0; K = 5)
        @test eltype(DCTGaussian(Float32, (2, 5))[1]) == Float32
        # A 1-d kernel can only filter vectors: n-d arrays need one kernel per dimension
        @test_throws DimensionMismatch imfilter(rand(5, 6), DCTGaussian(3.0))
        @test_throws DimensionMismatch imfilter(rand(5, 6), (DCTGaussian(3.0),))
        @test_throws DimensionMismatch imfilter(rand(5, 6, 7), slidinggaussian(3.0; order = 1))
        # (forcing FIR, the error comes from the padding, as for `KernelFactors.gaussian(3)`)
        @test_throws ArgumentError imfilter(rand(5, 6), DCTGaussian(3.0), "reflect", Algorithm.FIR())
        @test imfilter(rand(5, 6), (ReshapedOneD{2,0}(DCTGaussian(3.0)),)) isa Matrix{Float64}   # explicit is fine
        # σ = 0 is the identity
        k0 = DCTGaussian(0)
        @test axes(k0) == (0:0,) && k0[0] == 1
        @test ImageFiltering.iscopy(k0)
        img = rand(5, 6)
        @test imfilter(img, DCTGaussian((0, 0))) == img
        # Warnings and errors
        @test_logs (:warn, "σ is too small for accuracy") DCTGaussian(0.5)
        @test_logs DCTGaussian(0.5; emit_warning = false)
        @test_throws ArgumentError DCTGaussian(-1)
        @test_throws ArgumentError DCTGaussian(3; kind = 2)
        @test_throws ArgumentError DCTGaussian(3; K = 0)
        @test_throws ArgumentError DCTGaussian(3; K = 5, N = 3)

        # Fitting an arbitrary even kernel
        h = centered([1, 4, 6, 4, 1] / 16)
        @test_throws ArgumentError SlidingDCT(OffsetArray(h.parent, 0:4))
        @test_throws ArgumentError SlidingDCT(centered([1.0, 2.0, 3.0, 1.0, 1.0]))
        @test_throws ArgumentError SlidingDCT(h; K = 3, moments = (0, 2, 4, 6))
        # With K = N + 1 the representation is exact, whatever the DCT type
        for kind in (1, 3, 5, 7)
            k = SlidingDCT(h; K = 3, kind = kind)
            @test collect(k) ≈ collect(h)
        end
        k = SlidingDCT(h; K = 2)
        @test sum(k) ≈ 1
        @test sum(r -> r^2 * k[r], -2:2) ≈ sum(r -> r^2 * h[r], -2:2)
        @test eltype(SlidingDCT(centered(Float32[1, 2, 1]); K = 1)) == Float32
        @test sum(SlidingDCT(centered(Float32[1, 2, 1]); K = 1)) ≈ 4
        # A kernel that is itself a sum of cosines is recovered exactly, without moment constraints
        N = 12
        ω = [π * (2k + 1) / (2N + 1) for k in 0:2]
        hc = OffsetArray([0.3cos(ω[1] * r) - 0.2cos(ω[2] * r) + 0.1cos(ω[3] * r) for r in -N:N], -N:N)
        kc = SlidingDCT(hc; K = 3, kind = 7, moments = ())
        @test kc.c ≈ [0.3, -0.2, 0.1]
        @test collect(kc) ≈ collect(hc)
    end

    @testset "agreement with FIR filtering" begin
        borders = ("replicate", "circular", "symmetric", "reflect", Fill(0), Fill(0.25), Inner(), NA())
        # 1d
        x = rand(150)
        for kind in (1, 3, 5, 7), K in (1, 3), σ in (1.5, 6.0)
            k = DCTGaussian(σ; K = K, kind = kind)
            for border in borders
                @test agrees(x, k, border)
            end
        end
        # n-d with various element types, computing in the precision of the pixels (Float32
        # for all but the first image); see the edge cases for promotion to Float64
        imgs = (rand(37, 41), rand(Float32, 64, 20), rand(Gray{N0f8}, 30, 25),
                rand(RGB{N0f8}, 20, 22, 9), rand(RGB{Float32}, 3, 4, 5, 6))
        for img in imgs
            σs = ntuple(d -> 1 + 2d / 3, ndims(img))
            for (kind, K) in ((7, 3), (5, 2))
                ks = DCTGaussian(pixelfloat(img), σs; K = K, kind = kind)
                @test eltype(eltype(imfilter(img, ks))) == pixelfloat(img)
                for border in borders
                    border isa Fill && (border = Fill(zero(eltype(img))))
                    @test agrees(img, ks, border)
                end
            end
        end
        # Filtering only some dimensions, and arrays smaller than the kernel
        img = rand(50, 3, 2)
        k = DCTGaussian(4.0)
        k3 = ntuple(d -> ReshapedOneD{3,d-1}(k), 3)
        for ks in (k3[1:1], k3[2:2], k3[1:2], k3)
            for border in borders
                # NA() requires one kernel per dimension
                border isa NA && length(ks) < ndims(img) && continue
                @test agrees(img, ks, border)
            end
        end
        @test imfilter(Float64[], DCTGaussian(3.0)) == Float64[]
        @test agrees([2.0], DCTGaussian(3.0)) && imfilter([2.0], DCTGaussian(3.0)) ≈ [2.0]
        # Offset arrays
        img = OffsetArray(rand(20, 30), -5:14, 3:32)
        ks = DCTGaussian((2.0, 3.0))
        @test axes(imfilter(img, ks, "reflect")) == axes(img)
        @test agrees(img, ks, "reflect")
        # Restricted output region and preallocated output
        img = rand(40, 40)
        ref = fir(img, ks, "symmetric")
        out = similar(img)
        @test imfilter!(out, img, ks, "symmetric") === out
        @test filtersmatch(out, ref, img, ks)
        out = OffsetArray(zeros(10, 12), 11:20, 5:16)
        imfilter!(out, img, ks, "symmetric")
        @test filtersmatch(parent(out), ref[11:20, 5:16], img, ks)
        # In-place filtering (one and several stages)
        for kk in (ks, ks[1:1], (ks..., ks...))
            img2 = copy(img)
            @test imfilter!(img2, img2, kk, "reflect") === img2
            @test filtersmatch(img2, fir(img, kk, "reflect"), img, kk)
        end
        img3 = rand(15, 16, 17)
        k3 = DCTGaussian((2.0, 1.5, 2.5))
        for kk in (k3, k3[1:2], k3[2:3], k3[2:2])   # odd and even numbers of stages
            img4 = copy(img3)
            @test imfilter!(img4, img4, kk, "symmetric") === img4
            @test filtersmatch(img4, fir(img3, kk, "symmetric"), img3, kk)
        end
        # Filtering along a single dimension
        out = similar(img)
        for border in ("replicate", Pad(:circular), Fill(0.5))
            @test imfilter!(CPU1(Algorithm.DCT()), out, img, ks[2].data, 2, border) === out
            @test filtersmatch(out, fir(img, (ks[2],), border), img, (ks[2],), border)
        end
        img2 = copy(img)
        imfilter!(CPU1(Algorithm.DCT()), img2, img2, ks[1].data, 1, "symmetric")
        @test filtersmatch(img2, fir(img, (ks[1],), "symmetric"), img, (ks[1],))
        @test_throws ArgumentError imfilter!(CPU1(Algorithm.DCT()), out, img, ks[1].data, 3, "symmetric")
        @test_throws DimensionMismatch imfilter!(CPU1(Algorithm.DCT()), zeros(3, 3), img, ks[1].data, 1, "symmetric")
        # Algorithm selection; all algorithms agree
        @test ImageFiltering.filter_algorithm(img, img, ks) == Algorithm.DCT()
        @test ImageFiltering.filter_algorithm(img, img, (ks[1], KernelFactors.gaussian((1.0, 1.0))[2])) != Algorithm.DCT()
        @test filtersmatch(imfilter(img, ks, "symmetric", Algorithm.DCT()), ref, img, ks)
        @test filtersmatch(imfilter(CPU1(Algorithm.DCT()), img, ks, Fill(0)), fir(img, ks, Fill(0)), img, ks)
        @test filtersmatch(imfilter(CPU1(Algorithm.FIR()), img, ks), fir(img, ks), img, ks)
        # (the round-off of the FFT is of the same order as that of the recurrence here)
        @test filtersmatch(imfilter(img, ks, "replicate", Algorithm.FFT()), fir(img, ks), img, ks)
        # Cascades mixing FIR and SlidingDCT kernels (and multiple DCT kernels along a dimension)
        kf = KernelFactors.gaussian((1.0, 2.0))
        ksd = DCTGaussian((3.0, 1.5))
        mixed = (kf[1], ksd[2], ksd[1], kf[2])
        @test agrees(img, mixed, "reflect")
        for border in ("replicate", "reflect", Fill(0.3))
            @test agrees(img, (ksd[1], ksd[1]), border)
            @test agrees(img, (ksd[2], ksd[1], ksd[2]), border)
        end
        # A Fill border must account for the sum of the kernels of previous stages
        kn = SlidingDCT(centered([1.0, 2.0, 3.0, 2.0, 1.0]); K = 3)   # sum(kn) == 9
        kn2 = kernelfactors((kn, kn))
        @test agrees(img, kn2, Fill(0.7))
        @test agrees(img, (kn2[2], kn2[1]), Fill(0.7))
        @test agrees(img, (ksd..., ksd...), "circular")
        # Integer images
        @test agrees(rand(1:100, 30, 30), ksd)
        # NaN handling: with NA() missing values are ignored
        imgn = rand(60, 60); imgn[30, 30] = NaN
        @test !any(isnan, imfilter(imgn, ksd, NA()))
        @test agrees(imgn, ksd, NA())
        # Otherwise, like for IIR filters, a NaN contaminates the remainder of each line
        # (unlike FIR filtering, which only contaminates the neighborhood)
        out, ref = imfilter(imgn, ksd), imfilter(imgn, firkernel(ksd), Algorithm.FIR())
        @test !isnan(out[1, 1]) && isnan(out[end, end])
        @test !isnan(ref[1, 1]) && !isnan(ref[end, end])
    end

    @testset "properties" begin
        # constant images are unchanged
        for (img, σs) in ((fill(0.7, 50, 60), (5.0, 12.0)), (fill(RGB{N0f8}(0.2, 0.4, 0.6), 30, 30), (3.0, 3.0)))
            ks = DCTGaussian(σs)
            @test filtersmatch(imfilter(img, ks), img, img, ks)
        end
        # Accuracy with respect to an (essentially) exact Gaussian filter: a ±8σ window
        img = rand(200, 200)
        for σ in (2.0, 8.0, 25.0)
            l = 2 * ceil(Int, 8σ) + 1
            ref = imfilter(img, KernelFactors.gaussian((σ, σ), (l, l)), "replicate", Algorithm.FFT())
            p4 = psnr(imfilter(img, DCTGaussian((σ, σ)), "replicate"), ref)   # the default, K = 4
            p6 = psnr(imfilter(img, DCTGaussian((σ, σ); K = 6), "replicate"), ref)
            piir = psnr(imfilter(img, KernelFactors.IIRGaussian((σ, σ)), "replicate"), ref)
            @test p4 > 72
            @test p6 > 105
            @test p4 > piir + 25   # substantially more accurate than IIRGaussian
        end
        # the input is not modified
        img = rand(20, 20); img0 = copy(img)
        imfilter(img, DCTGaussian((2.0, 2.0)))
        @test img == img0
    end

    @testset "numerical stability" begin
        # Long signals: the round-off errors of the recurrence must grow as a random walk
        # (∝ √n, as assumed by `filtertol`), not linearly or exponentially
        x = rand(200_000)
        for (σ, K, kind, order) in ((3.0, 3, 7, 0), (50.0, 6, 7, 0), (20.0, 4, 5, 0), (20.0, 4, 1, 0), (500.0, 3, 3, 0),
                                    (10.0, 4, 5, 1), (10.0, 5, 7, 2))
            k = slidinggaussian(σ; K = K, kind = kind, order = order)
            @test agrees(x, k, "reflect")
        end
        # Float32: compare with the exact (Float64) filtering by the same Float32 kernel
        for k32 in (DCTGaussian(10.0f0), slidinggaussian(10.0f0; order = 1), slidinggaussian(10.0f0; order = 2))
            x32 = rand(Float32, 200_000)
            out = imfilter(x32, k32)
            @test eltype(out) == Float32
            @test filtersmatch(out, fir(Float64.(x32), convert(SlidingKernel{Float64}, k32)), x32, k32)
        end
    end

    @testset "derivatives" begin
        for σ in (2.0, 7.5), order in (1, 2), kind in (1, 3, 5, 7)
            k = slidinggaussian(σ; order = order, kind = kind)
            N = k.N
            @test k isa (isodd(order) ? SlidingDST : SlidingDCT)
            @test all(r -> k[-r] ≈ (-1)^order * k[r], 0:N)
            # moments: exact on low-degree polynomials
            if order == 1
                @test sum(r -> r * k[r], -N:N) ≈ 1
            else
                @test abs(sum(k)) < 1e-12
                @test sum(r -> r^2 * k[r], -N:N) ≈ 2
            end
        end
        # accuracy with the default parameters, and improvement with K
        for σ in (3.0, 10.0), order in (1, 2)
            ref = sampledderivative(σ, order)
            # (the default K gives an accuracy similar to that of the Gaussian)
            @test l2err(slidinggaussian(σ; order = order), ref) < 6e-3
            errs = [l2err(slidinggaussian(σ; order = order, K = K), ref) for K in 3:6]
            @test issorted(errs; rev = true)
        end
        @test slidinggaussian(3.0) == DCTGaussian(3.0)
        @test slidinggaussian(3.0; order = 1) isa SlidingDST{Float64,4}
        @test slidinggaussian(3.0; order = 2) isa SlidingDCT{Float64,5,true}   # K = 6: 5 cosines and the constant term
        @test eltype(slidinggaussian(3.0f0; order = 1)) == Float32
        # derivatives of polynomials, away from the borders
        t = collect(1.0:300.0)
        for σ in (2.0, 10.0)
            k1, k2 = slidinggaussian(σ; order = 1), slidinggaussian(σ; order = 2)
            inner = 6ceil(Int, σ):300-6ceil(Int, σ)
            lin, quad = 3 .* t .- 7, 0.5 .* t .^ 2 .+ t
            @test filtersmatch(imfilter(lin, k1)[inner], fill(3, length(inner)), lin, k1)
            @test filtersmatch(imfilter(quad, k2)[inner], fill(1, length(inner)), quad, k2)
            @test filtersmatch(imfilter(quad, k1)[inner], t[inner] .+ 1, quad, k1)
        end
        # Tuples of kernels: gradients
        ks = slidinggaussian((2.0, 3.0); order = (1, 0))
        @test ks[1].data isa SlidingDST && ks[2].data isa SlidingDCT
        @test ks[2].data == DCTGaussian(3.0)
        @test slidinggaussian((2.0, 3.0); order = (0, 2), K = (3, 5))[2].data isa SlidingDCT{Float64,4,true}   # 4 cosines + constant
        @test_throws DimensionMismatch slidinggaussian((2.0, 3.0); order = (1, 0, 0))
        @test_throws ArgumentError slidinggaussian(3.0; order = 3)
        @test_throws ArgumentError slidinggaussian(0; order = 1)
        @test slidinggaussian(0) == DCTGaussian(0)
        # Fitting arbitrary odd kernels
        h = centered([-1.0, -2.0, 0.0, 2.0, 1.0])
        @test_throws ArgumentError SlidingDST(centered([1.0, 2.0, 0.0, 2.0, 1.0]))
        @test collect(SlidingDST(h; K = 2)) ≈ collect(h)   # K = N: exact
        N = 12
        ω = [2π * k / (2N + 1) for k in 1:3]
        hs = OffsetArray([0.3sin(ω[1] * r) - 0.2sin(ω[2] * r) + 0.1sin(ω[3] * r) for r in -N:N], -N:N)
        @test SlidingDST(hs; K = 3, moments = ()).c ≈ [0.3, -0.2, 0.1]
        @test_throws ArgumentError SlidingDST(h; K = 3)   # K must not exceed N

        # Agreement with FIR filtering
        x = rand(150)
        for kind in (1, 5), σ in (1.5, 6.0)
            k = slidinggaussian(σ; order = 1, kind = kind)
            for border in ("replicate", "circular", "symmetric", "reflect", Fill(0), Fill(0.25), Inner())
                @test agrees(x, k, border)
            end
        end
        for img in (rand(40, 50), rand(Gray{N0f8}, 30, 25), rand(RGB{Float32}, 20, 22, 9))
            σs = ntuple(d -> 1 + d / 2, ndims(img))
            for order in ((1, ntuple(_ -> 0, ndims(img) - 1)...), (0, 2, ntuple(_ -> 0, ndims(img) - 2)...))
                ks = slidinggaussian(pixelfloat(img), σs; order = order)
                for border in ("replicate", "reflect", Fill(zero(eltype(img))), Fill(oneunit(eltype(img))))
                    @test agrees(img, ks, border)
                end
            end
        end
    end

    @testset "SeparableSum" begin
        kx, ky = DCTGaussian((2.0, 3.0))
        dx, dy = slidinggaussian((2.0, 3.0); order = (2, 2))
        L = SeparableSum(((dx, ky), (kx, dy)))
        @test L.weights == (true, true)
        @test ndims(L) == 2
        @test eltype(L) == Float64
        @test axes(L) == (-max(dx.data.N, kx.data.N):max(dx.data.N, kx.data.N), -max(dy.data.N, ky.data.N):max(dy.data.N, ky.data.N))
        @test occursin("SeparableSum of 2", sprint(show, L))
        @test_throws DimensionMismatch SeparableSum(((dx, ky), (kx, dy)), (1,))
        @test_throws DimensionMismatch SeparableSum(((dx, ky), (kx.data,)))
        @test DCTLoG((2.0, 3.0)).terms == L.terms
        @test DCTLoG((2.0, 3.0); K2 = 4).terms[1][1].data == slidinggaussian(2.0; order = 2, K = 4)
        # Filtering is the sum of the filtered terms, for all borders and algorithms
        img = rand(40, 45)
        ref(border) = fir(img, (dx, ky), border) + fir(img, (kx, dy), border)
        @test fir(img, L, "reflect") ≈ ref("reflect")
        # (NA() is not meaningful for kernels that sum to zero, such as second derivatives)
        for border in ("replicate", "circular", "symmetric", "reflect", Fill(0.5), Inner())
            @test agrees(img, L, border)
        end
        @test filtersmatch(imfilter(img, L), ref("replicate"), img, L)
        out = imfilter(img, L, Inner())
        @test axes(out) == map(intersect, axes(imfilter(img, (dx, ky), Inner())), axes(imfilter(img, (kx, dy), Inner())))
        @test imfilter(Float32, img, L) isa Matrix{Float32}
        @test filtersmatch(imfilter(CPU1(Algorithm.DCT()), img, L, "reflect"), ref("reflect"), img, L)
        @test filtersmatch(imfilter(CPU1(Algorithm.FIR()), Float64, img, L, "reflect"), ref("reflect"), img, L)
        out = similar(img)
        @test imfilter!(out, img, L, "reflect") === out
        @test filtersmatch(out, ref("reflect"), img, L)
        @test filtersmatch(imfilter!(CPU1(Algorithm.DCT()), out, img, L, Pad(:reflect)), ref("reflect"), img, L)
        img2 = copy(img)
        imfilter!(img2, img2, L, "reflect")
        @test filtersmatch(img2, ref("reflect"), img, L)
        # Weights
        D = SeparableSum(((kx, ky), (dx, dy)), (2, -0.5))
        @test agrees(img, D, "reflect")
        @test fir(img, D, "reflect") ≈ 2 * fir(img, (kx, ky), "reflect") - 0.5 * fir(img, (dx, dy), "reflect")
        @test imfilter(rand(Gray{N0f8}, 20, 20), L) isa Matrix{Gray{Float64}}

        # Laplacian of Gaussian
        img = rand(80, 90)
        for σ in (2.0, 5.0)
            ref = imfilter(img, Kernel.LoG(σ), "reflect")
            @test maximum(abs, imfilter(img, DCTLoG(σ), "reflect") - ref) < 3e-2 * maximum(abs, ref)
            # exact on quadratic images
            q = [0.5i^2 + 2j^2 + i for i in 1:80, j in 1:90]
            m = 6ceil(Int, σ)
            @test filtersmatch(imfilter(q, DCTLoG(σ))[m:end-m, m:end-m], fill(5, 81 - 2m, 91 - 2m), q, DCTLoG(σ))
        end
        # Accuracy with respect to the LoG of the untruncated sampled Gaussian
        for σ in (2.0, 8.0)
            d2, g0 = centered(sampledderivative(σ, 2).parent), centered(sampledgaussian(σ).parent)
            exact = imfilter(img, kernelfactors((d2, g0)), "reflect") + imfilter(img, kernelfactors((g0, d2)), "reflect")
            relerr(out) = sqrt(sum(abs2, out - exact) / sum(abs2, exact))
            @test relerr(imfilter(img, DCTLoG(σ), "reflect")) < 1e-2
            σ > 5 && @test relerr(imfilter(img, DCTLoG(σ), "reflect")) < relerr(imfilter(img, Kernel.LoG(σ), "reflect"))
        end
        # The terms are accumulated without (type-unstable) per-pixel overhead
        imgf = rand(Float32, 200, 200)
        imfilter(imgf, DCTLoG(3.0))
        @test @allocated(imfilter(imgf, DCTLoG(3.0))) < 8 * sizeof(similar(imgf, Float64))
        img3 = rand(30, 31, 32)
        ref = imfilter(img3, Kernel.LoG((2.0, 2.5, 3.0)), "reflect")
        @test maximum(abs, imfilter(img3, DCTLoG((2.0, 2.5, 3.0)), "reflect") - ref) < 3e-2 * maximum(abs, ref)
        @test length(DCTLoG((2.0, 2.5, 3.0)).terms) == 3
        # Difference of Gaussians, compared with an (essentially) exact one using ±8σ windows
        g(σ) = (l = 2ceil(Int, 8σ) + 1; KernelFactors.gaussian((σ, σ), (l, l)))
        for σ in (2.0, 5.0)
            exact = imfilter(img, g(σ), "reflect", Algorithm.FFT()) - imfilter(img, g(√2 * σ), "reflect", Algorithm.FFT())
            err(D) = maximum(abs, imfilter(img, D, "reflect") - exact)
            @test err(DCTDoG(σ)) < err(Kernel.DoG(σ)) / 2   # more accurate than the (truncated) dense kernel
            @test err(DCTDoG(σ; K = 5)) < 2e-4
        end
        @test DCTDoG((2.0, 3.0)).terms[2] == DCTGaussian((2 * √2, 3 * √2))
        @test DCTDoG((2.0, 3.0), (4.0, 5.0)).weights == (true, -1)
    end

    @testset "edge cases" begin
        borders = ("replicate", "circular", "symmetric", "reflect", Fill(0), Fill(0.25), Inner(), NA())
        k0, k1, k2 = DCTGaussian(2.5), slidinggaussian(2.5; order = 1), slidinggaussian(2.5; order = 2)
        N = k0.N
        # Axes shorter than (or close to) the kernel length 2N+1
        for k in (k0, k1, k2), n in unique((1, 2, 3, N - 1, N, N + 1, 2N, 2N + 1, 2N + 2)), border in borders
            border isa NA && k !== k0 && continue   # (NA() normalizes by the sum of the kernel)
            @test agrees(rand(n), k, border)
            @test agrees(rand(n, 7), kernelfactors((k, DCTGaussian(1.5))), border)
        end
        # Impulses at and near the boundaries expose off-by-one and sign errors in the
        # virtual padding
        n = 3N + 4
        for k in (k0, k1, k2), p in unique((1, 2, N, N + 1, N + 2, n - N - 1, n - N, n - 1, n)), border in borders
            border isa NA && continue
            x = zeros(n); x[p] = 1
            @test agrees(x, k, border)
            img = zeros(n, n); img[p, n+1-p] = 1
            @test agrees(img, kernelfactors((k, k)), border)
        end
        # An interior impulse gives the reversed kernel (imfilter computes a correlation),
        # and with Fill(0) an impulse at the edge gives the truncated kernel
        for k in (k0, k1, k2)
            Nk = k.N
            x = zeros(4Nk + 1); x[2Nk+1] = 1
            @test filtersmatch(imfilter(x, k, Fill(0)), [zeros(Nk); [k[-r] for r in -Nk:Nk]; zeros(Nk)], x, k)
            x = zeros(2Nk); x[1] = 1
            @test filtersmatch(imfilter(x, k, Fill(0))[1:Nk+1], [k[-r] for r in 0:Nk], x, k)
        end
        # Kernels that are not Gaussians: arbitrary coefficients and (non-harmonic)
        # frequencies, and smooth kernels fitted by SlidingDCT/SlidingDST
        kd = SlidingDCT(9, SVector(0.3, -0.2, 0.05), SVector(0.21, 0.83, 1.7), 0.1, true)
        ks = SlidingDST(9, SVector(0.4, 0.1), SVector(0.35, 2.2))
        kd32 = convert(SlidingKernel{Float32}, kd)
        hhat = centered([(1 - r^2 / 36) * exp(-r^2 / 72) for r in -24.0:24.0])  # "Mexican hat"
        hgab = centered([sin(0.4r) * exp(-r^2 / 128) for r in -24.0:24.0])     # odd Gabor kernel
        hat, gab = SlidingDCT(hhat; K = 6), SlidingDST(hgab; K = 6)
        @test kd[3] ≈ 0.1 + 0.3cos(0.63) - 0.2cos(2.49) + 0.05cos(5.1)
        @test ks[-3] ≈ -(0.4sin(1.05) + 0.1sin(6.6))
        @test l2err(hat, hhat) < 2e-3 && l2err(gab, hgab) < 4e-3   # (smooth kernels fit well)
        x = rand(60)
        img = rand(Float32, 30, 40)   # (with Float64 kernels: promoted to Float64)
        for k in (kd, ks, kd32, hat, gab), border in borders
            border isa NA && continue
            @test agrees(x, k, border)
            @test agrees(img, kernelfactors((k, kd)), border)
        end
        @test agrees(img, (kernelfactors((kd, ks))..., kernelfactors((hat, gab))...), "reflect")
        # SubArrays (contiguous and strided), as input and as output
        A = rand(50, 60, 3)
        ks2 = slidinggaussian((2.0, 3.5); order = (1, 0))
        for v in (view(A, :, :, 2), view(A, 3:45, 10:50, 1), view(A, 1:2:49, 60:-2:1, 3), view(A, :, 2, :)')
            for border in ("replicate", "reflect", Fill(0.5), Inner())
                @test agrees(v, ks2, border)
                @test agrees(v, DCTGaussian((3.0, 2.0)), border)
            end
            @test agrees(v, DCTLoG(2.0))
            @test imfilter(v, DCTLoG(2.0)) == imfilter(copy(v), DCTLoG(2.0))
        end
        @test agrees(view(A, 1:2:50, :, :), DCTGaussian((1.5, 2.0, 1.0)), "symmetric")
        @test agrees(view(rand(100), 2:3:100), k1, "reflect")
        B = zeros(70, 80)
        out = view(B, 11:60, 11:70)
        img = rand(50, 60)
        @test imfilter!(out, img, ks2, "reflect") === out
        @test filtersmatch(out, fir(img, ks2, "reflect"), img, ks2)
        @test all(iszero, B[1:10, :]) && all(iszero, B[:, 71:80])
        # in place, on a view
        C = rand(60, 60); C0 = copy(C)
        v = view(C, 5:54, 6:55)
        imfilter!(v, v, ks2, "symmetric")
        @test filtersmatch(v, fir(C0[5:54, 6:55], ks2, "symmetric"), v, ks2)
        @test C[1:4, :] == C0[1:4, :]
        # NA() on a view
        V = view(rand(60, 60), 5:54, :); V[10, 10] = NaN
        @test agrees(V, DCTGaussian((2.0, 2.0)), NA())
    end

    @testset "multithreading" begin
        # The blocks of lines given to the tasks partition the lines
        for (Rpre, Rpost) in ((CartesianIndices(()), CartesianIndices((1:7, 0:4))), (CartesianIndices((1:100,)), CartesianIndices(())),
                              (CartesianIndices((1:5, 2:9)), CartesianIndices((3,))), (CartesianIndices((1:10,)), CartesianIndices((2,))),
                              (CartesianIndices(()), CartesianIndices(())))
            for ntasks in (1, 2, 3, 8)
                blocks = ImageFiltering._dct_blocks(Rpre, Rpost, ntasks)
                @test 1 <= length(blocks) <= ntasks
                lines = [(Ipre, Ipost) for (Bpre, Bpost) in blocks for Ipre in Bpre for Ipost in Bpost]
                @test sort(lines) == sort(vec([(Ipre, Ipost) for Ipre in Rpre, Ipost in Rpost]))
                # blocks along the (contiguous) first dimension keep at least 16 lines
                if ndims(Rpre) == 1 && length(unique(first.(blocks))) > 1
                    @test all(b -> length(b[1]) >= 16, blocks)
                end
            end
        end
        # Filtering by several tasks gives exactly the serial result. CI runs on a single
        # thread, so we call the task-based function directly with several tasks.
        img = rand(Float32, 70, 45, 3)
        for ks in (DCTGaussian(Float32, (2.0, 3.0, 1.5)), slidinggaussian((2.0, 3.0, 1.5); order = (1, 0, 2)))
            for k in ks
                Rpre, ind, Rpost = KernelFactors.iterdims(axes(img), k)
                N = k.data.N
                pidx = ImageFiltering.padindex(Pad(:reflect), N, UnitRange(ind), N)
                O0pre, O0post = zero(eltype(Rpre)), zero(eltype(Rpost))
                serial = ImageFiltering._imfilter_dct!(similar(img), img, k.data, Rpre, ind, Rpost, pidx, nothing, O0pre, O0post)
                for ntasks in (2, 4, 7)
                    @test ImageFiltering._imfilter_dct_tasks!(ntasks, similar(img), img, k.data, Rpre, ind, Rpost, pidx, nothing, O0pre, O0post) == serial
                end
            end
        end
        # Through the API (with several threads if Julia was started with them),
        # CPUThreads gives exactly the CPU1 result (or the same error), along every path
        rt, r1 = CPUThreads(Algorithm.DCT()), CPU1(Algorithm.DCT())
        result(r, args...) = try imfilter(r, args...) catch err typeof(err) end
        for img in (rand(300, 250), rand(RGB{N0f8}, 30, 40), rand(40, 41, 42), rand(20, 21, 22), rand(70_000))
            ks = ndims(img) == 1 ? slidinggaussian(3.0; order = 1) : slidinggaussian(ntuple(d -> 1.0 + d, ndims(img)); order = ntuple(d -> d == 1 ? 1 : 0, ndims(img)))
            for border in ("replicate", "symmetric", Fill(zero(eltype(img))), Inner(), NA())
                @test result(rt, img, ks, border) == result(r1, img, ks, border)
            end
        end
        img = rand(300, 250)
        L = DCTLoG(2.0)
        @test imfilter(rt, img, L) == imfilter(r1, img, L)
        ks = DCTGaussian((2.0, 3.0))
        @test imfilter!(rt, similar(img), img, ks[2].data, 2, "reflect") == imfilter!(r1, similar(img), img, ks[2].data, 2, "reflect")
        img2, img3 = copy(img), copy(img)
        @test imfilter!(rt, img2, img2, ks, Pad(:reflect)) == imfilter!(r1, img3, img3, ks, Pad(:reflect))
        # As for FIRTiled, multiple threads are used by default when available
        @test ImageFiltering.default_resource(Algorithm.DCT()) == (Threads.nthreads() > 1 ? rt : r1)
    end

    @testset "inference" begin
        img = rand(Float32, 20, 20)
        @inferred imfilter(img, DCTGaussian((2.0, 3.0)))
        @test @inferred(imfilter(img, DCTGaussian(Float32, (2.0, 3.0)))) isa Matrix{Float32}
        @test @inferred(imfilter(rand(Gray{N0f8}, 10, 10), DCTLoG(Float32, 2.0))) isa Matrix{Gray{Float32}}
        @inferred imfilter(rand(RGB{N0f8}, 10, 10), DCTGaussian((2.0, 3.0)))
        @inferred imfilter(img, slidinggaussian((2.0, 3.0); order = (1, 0)))
        @inferred imfilter(img, DCTLoG(2.0))
    end
end
