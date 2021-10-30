using ImageFiltering.Models

@testset "solve_ROF_PD" begin
    # This testset is modified from its CPU version

    @testset "Numerical" begin
        # 2D Gray
        img = restrict(testimage("cameraman"))
        img_noisy = img .+ 0.05randn(MersenneTwister(0), size(img))
        img_smoothed = solve_ROF_PD(img_noisy, 0.05, 20)
        @test ndims(img_smoothed) == 2
        @test eltype(img_smoothed) <: Gray
        @test assess_psnr(img_smoothed, img) > 31.67
        @test assess_ssim(img_smoothed, img) > 0.90

        img_noisy_cu = CuArray(float32.(img_noisy))
        img_smoothed_cu = solve_ROF_PD(img_noisy_cu, 0.05, 20)
        @test img_smoothed_cu isa CuArray
        @test eltype(eltype(img_smoothed_cu)) == Float32
        @test Array(img_smoothed_cu) ≈ img_smoothed

        # 2D RGB
        img = restrict(testimage("lighthouse"))
        img_noisy = img .+ colorview(RGB, ntuple(i->0.05.*randn(MersenneTwister(i), size(img)), 3)...)
        img_smoothed = solve_ROF_PD(img_noisy, 0.03, 20)
        @test ndims(img_smoothed) == 2
        @test eltype(img_smoothed) <: RGB
        @test assess_psnr(img_smoothed, img) > 32.15
        @test assess_ssim(img_smoothed, img) > 0.90

        img_noisy_cu = CuArray(float32.(img_noisy))
        img_smoothed_cu = solve_ROF_PD(img_noisy_cu, 0.03, 20)
        @test img_smoothed_cu isa CuArray
        @test eltype(eltype(img_smoothed_cu)) == Float32
        @test Array(img_smoothed_cu) ≈ img_smoothed

        # 3D Gray
        img = Gray.(restrict(testimage("mri"), (1, 2)))
        img_noisy = img .+ 0.05randn(MersenneTwister(0), size(img))
        img_smoothed = solve_ROF_PD(img_noisy, 0.02, 20)
        @test ndims(img_smoothed) == 3
        @test eltype(img_smoothed) <: Gray
        @test assess_psnr(img_smoothed, img) > 31.78
        @test assess_ssim(img_smoothed, img) > 0.85

        img_noisy_cu = CuArray(float32.(img_noisy))
        img_smoothed_cu = solve_ROF_PD(img_noisy_cu, 0.02, 20)
        @test img_smoothed_cu isa CuArray
        @test eltype(eltype(img_smoothed_cu)) == Float32
        @test Array(img_smoothed_cu) ≈ img_smoothed

        # 3D RGB
        img = RGB.(restrict(testimage("mri"), (1, 2)))
        img_noisy = img .+ colorview(RGB, ntuple(i->0.05.*randn(MersenneTwister(i), size(img)), 3)...)
        img_smoothed = solve_ROF_PD(img_noisy, 0.02, 20)
        @test ndims(img_smoothed) == 3
        @test eltype(img_smoothed) <: RGB
        @test assess_psnr(img_smoothed, img) > 31.17
        @test assess_ssim(img_smoothed, img) > 0.79

        img_noisy_cu = CuArray(float32.(img_noisy))
        img_smoothed_cu = solve_ROF_PD(img_noisy_cu, 0.02, 20)
        @test img_smoothed_cu isa CuArray
        @test eltype(eltype(img_smoothed_cu)) == Float32
        @test Array(img_smoothed_cu) ≈ img_smoothed
    end
end
