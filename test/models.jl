using ImageFiltering.Models

@testset "solve_ROF_PD" begin
    # Note: random seed really matters a lot

    @testset "Numerical" begin
        # 2D Gray
        img = restrict(testimage("cameraman"))
        img_noisy = img .+ 0.05randn(MersenneTwister(0), size(img))
        img_smoothed = solve_ROF_PD(img_noisy, 0.05, 20)
        @test ndims(img_smoothed) == 2
        @test eltype(img_smoothed) <: Gray
        @test assess_psnr(img_smoothed, img) > 31.67
        @test assess_ssim(img_smoothed, img) > 0.90

        # 2D RGB
        img = restrict(testimage("lighthouse"))
        img_noisy = img .+ colorview(RGB, ntuple(i->0.05.*randn(MersenneTwister(i), size(img)), 3)...)
        img_smoothed = solve_ROF_PD(img_noisy, 0.03, 20)
        @test ndims(img_smoothed) == 2
        @test eltype(img_smoothed) <: RGB
        @test assess_psnr(img_smoothed, img) > 32.15
        @test assess_ssim(img_smoothed, img) > 0.90

        # 3D Gray
        img = Gray.(restrict(testimage("mri"), (1, 2)))
        img_noisy = img .+ 0.05randn(MersenneTwister(0), size(img))
        img_smoothed = solve_ROF_PD(img_noisy, 0.02, 20)
        @test ndims(img_smoothed) == 3
        @test eltype(img_smoothed) <: Gray
        @test assess_psnr(img_smoothed, img) > 31.78
        @test assess_ssim(img_smoothed, img) > 0.85

        # 3D RGB
        img = RGB.(restrict(testimage("mri"), (1, 2)))
        img_noisy = img .+ colorview(RGB, ntuple(i->0.05.*randn(MersenneTwister(i), size(img)), 3)...)
        img_smoothed = solve_ROF_PD(img_noisy, 0.02, 20)
        @test ndims(img_smoothed) == 3
        @test eltype(img_smoothed) <: RGB
        @test assess_psnr(img_smoothed, img) > 31.17
        @test assess_ssim(img_smoothed, img) > 0.79
    end

    @testset "FixedPointNumbers" begin
        A = rand(N0f8, 20, 20)
        @test solve_ROF_PD(A, 0.01, 5) ≈ solve_ROF_PD(float32.(A), 0.01, 5)
    end

    @testset "OffsetArray" begin
        Ao = OffsetArray(rand(N0f8, 20, 20), -1, -1)
        out = solve_ROF_PD(Ao, 0.01, 5)
        @test axes(out) == axes(Ao)
    end
end
