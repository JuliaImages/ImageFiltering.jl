using ImageFiltering, Test, Statistics

@testset "gabor" begin
    σx = 8
    σy = 12
    size_x = 6*σx+1
    size_y = 6*σy+1
    γ = σx/σy
    # zero size forces default kernel width, with warnings

    @test_logs (:warn, r"The input parameter size_") match_mode=:any begin
        kernel = Kernel.gabor(0,0,σx,0,5,γ,0)
        @test isequal(size(kernel[1]),(size_x,size_y))
    end
    @test_logs (:warn, r"The input parameter size_") match_mode=:any begin
        kernel = Kernel.gabor(0,0,σx,π,5,γ,0)
        @test isequal(size(kernel[1]),(size_x,size_y))
    end

    for x in 0:4, y in 0:4, z in 0:4, t in 0:4
        σx1 = 2*x+1
        σy1 = 2*y+1
        λ = 2*z+1
        γ1 = σx1/σy1
        θ = 2*t+1
        kernel1 = Kernel.gabor(9,9,σx1,θ,λ,γ1,0)
        kernel2 = Kernel.gabor(9,9,σx1,θ+π,λ,γ1,0)
        @test abs(sum(kernel1[1] - kernel2[1])) < 1e-2
        @test abs(sum(kernel1[2] - kernel2[2])) < 1e-2
    end

    x1 = [j for i in 0:49,j in 0:49]
    wavelengths = (3, 10)
    images = [sin.(2*π*x1/λ) for λ in wavelengths]
    σx = 4
    σy = 5
    function match_score(image, λ)
        gabor_real = imfilter(image,centered(Kernel.gabor(6*σx+1,6*σy+1,σx,0,λ,σx/σy,0)[1]),"replicate")
        gabor_imag = imfilter(image,centered(Kernel.gabor(6*σx+1,6*σy+1,σx,0,λ,σx/σy,0)[2]),"replicate")
        gabor_result = sqrt.((gabor_real.*gabor_real) + (gabor_imag.*gabor_imag))
        return mean(gabor_result)
    end
    gabor_output = rand(Float64,2,2)
    for i = 1:2
        for j = 1:2
            gabor_output[i,j] = match_score(images[i],wavelengths[j])
        end
    end
    @test gabor_output[1,1] > gabor_output[1,2]
    @test gabor_output[2,2] > gabor_output[1,2]
    @test gabor_output[1,1] > gabor_output[2,1]
    @test gabor_output[2,2] > gabor_output[2,1]

    @test_throws ArgumentError Kernel.gabor(9,9,-2,0,5,0.1,0)
    @test_throws ArgumentError Kernel.gabor(9,9,2,0,-5,0.1,0)
    @test_throws ArgumentError Kernel.gabor(9,9,2,0,5,0,0)

end
