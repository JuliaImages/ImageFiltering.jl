using FFTW

@testset "fft" begin
    @testset "interface" begin
        @testset "1d" begin
            xs = ([1, -1],
                [1 -1])
            good_szs = ((4,),
                (4,4),
                (4,4,4))
            bad_szs = ((1,),
                (1,1))

            foreach(xs) do x
                otf = @test_nowarn psf2otf(x)
                psf = @test_nowarn otf2psf(otf)

                foreach(good_szs) do sz
                    @test_nowarn psf2otf(x, sz...)
                    otf = @test_nowarn psf2otf(x, sz)
                    @test_nowarn otf2psf(x, size(x)...)
                    psf = @test_nowarn otf2psf(otf, size(x))
                end

                foreach(bad_szs) do sz
                    @test_throws DimensionMismatch psf2otf(x, sz)
                end
            end
        end
    
        @testset "2d" begin
            x = [-1 0 1; -2 0 2; -1 0 1]
            good_szs = (
                (4,4),
                (4,4,4))
            bad_szs = ((1,),
                (1,1))

            otf = @test_nowarn psf2otf(x)
            psf = @test_nowarn otf2psf(otf)

            foreach(good_szs) do sz
                @test_nowarn psf2otf(x, sz...)
                otf = @test_nowarn psf2otf(x, sz)
                @test_nowarn otf2psf(x, size(x)...)
                psf = @test_nowarn otf2psf(otf, size(x))
            end

            foreach(bad_szs) do sz
                @test_throws DimensionMismatch psf2otf(x, sz)
            end
            
        end
    end

    @testset "numerical" begin
        x = [1, -1]
        otf_true = [0.0000 + 0.0000im   0.0000 + 0.0000im   0.0000 + 0.0000im   0.0000 + 0.0000im;
            -1.0000 + 1.0000im  -1.0000 + 1.0000im  -1.0000 + 1.0000im  -1.0000 + 1.0000im;
            -2.0000 + 0.0000im  -2.0000 + 0.0000im  -2.0000 + 0.0000im  -2.0000 + 0.0000im;
            -1.0000 - 1.0000im  -1.0000 - 1.0000im  -1.0000 - 1.0000im  -1.0000 - 1.0000im]
        otf = psf2otf(x, (4,4)) 
        @test otf ≈ otf_true
        @test otf2psf(otf, size(x)) ≈ x


        x = [-1 0 1; -2 0 2; -1 0 1]
        otf_true = [0.0000 + 0.0000im   0.0000 - 8.0000im   0.0000 + 0.0000im   0.0000 + 8.0000im;
            0.0000 + 0.0000im   0.0000 - 4.0000im   0.0000 + 0.0000im   0.0000 + 4.0000im;
            0.0000 + 0.0000im   0.0000 + 0.0000im   0.0000 + 0.0000im   0.0000 + 0.0000im;
            0.0000 + 0.0000im   0.0000 - 4.0000im   0.0000 + 0.0000im   0.0000 + 4.0000im]
        otf = psf2otf(x, (4,4)) 
        @test otf ≈ otf_true
        @test otf2psf(otf, size(x)) ≈ x
    end
end

nothing
