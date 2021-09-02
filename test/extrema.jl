@testset "extrema" begin
    @testset "local extrema" begin
        A = zeros(Int, 9, 9); A[[1:2;5],5].=1
        @test findlocalmaxima(A) == [CartesianIndex((5,5))]
        @test findlocalmaxima(A; window=(1,3)) == [CartesianIndex((1,5)),CartesianIndex((2,5)),CartesianIndex((5,5))]
        @test findlocalmaxima(A; window=(1,3), edges=false) == [CartesianIndex((2,5)),CartesianIndex((5,5))]
        A = zeros(Int, 9, 9, 9); A[[1:2;5],5,5].=1
        @test findlocalmaxima(A) == [CartesianIndex((5,5,5))]
        @test findlocalmaxima(A; window=(1,3,1)) == [CartesianIndex((1,5,5)),CartesianIndex((2,5,5)),CartesianIndex((5,5,5))]
        @test findlocalmaxima(A, window=(1,3,1), edges=false) == [CartesianIndex((2,5,5)),CartesianIndex((5,5,5))]
        A = zeros(Int, 9, 9); A[[1:2;5],5].=-1
        @test findlocalminima(A) == [CartesianIndex((5,5))]
    end

    @testset "blob_LoG" begin
        A = zeros(Int, 9, 9); A[5, 5] = 1
        blobs = blob_LoG(A, 2.0.^[0.5,0,1])
        @test length(blobs) == 1
        blob = blobs[1]
        @test blob.amplitude ≈ 0.3183098861837907
        @test blob.σ === (1.0, 1.0)
        @test blob.location == CartesianIndex((5,5))
        str = sprint(print, blob)
        @test occursin("σ=$((1.0, 1.0))", str)
        @test eval(Meta.parse(str)) == blob
        @test blob_LoG(A, [1.0]) == blobs
        @test blob_LoG(A, [1.0]; edges=(true, false, false)) == blobs
        @test isempty(blob_LoG(A, [1.0]; edges=false))
        A = zeros(Int, 9, 9); A[1, 5] = 1
        blobs = blob_LoG(A, 2.0.^[0,0.5,1])
        A = zeros(Int, 9, 9); A[1,5] = 1
        blobs = blob_LoG(A, 2.0.^[0.5,0,1])
        @test all(b.amplitude < 1e-16 for b in blobs)
        blobs = filter(b->b.amplitude > 0.1, blob_LoG(A, 2.0.^[0.5,0,1]; edges=true))
        @test length(blobs) == 1
        @test blobs[1].location == CartesianIndex((1,5))
        @test filter(b->b.amplitude > 0.1, blob_LoG(A, 2.0.^[0.5,0,1], edges=(true, true, false))) == blobs
        @test isempty(blob_LoG(A, 2.0.^[0,1], edges=(false, true, false)))
        blobs = blob_LoG(A, 2.0.^[0,0.5,1], edges=(true, false, true))
        @test all(b.amplitude < 1e-16 for b in blobs)
        # stub test for N-dimensional blob_LoG:
        A = zeros(Int, 9, 9, 9); A[5, 5, 5] = 1
        blobs = blob_LoG(A, 2.0.^[0.5, 0, 1])
        @test length(blobs) == 1
        @test blobs[1].location == CartesianIndex((5,5,5))
        # kinda anisotropic image
        A = zeros(Int,9,9,9); A[5,4:6,5] .= 1;
        blobs = blob_LoG(A,2 .^ [1.,0,0.5], σshape=(1.,3.,1.))
        @test length(blobs) == 1
        @test blobs[1].location == CartesianIndex((5,5,5))
        A = zeros(Int,9,9,9); A[1,1,4:6] .= 1;
        blobs = filter(b->b.amplitude > 0.1, blob_LoG(A, 2.0.^[0.5,0,1], edges=true, σshape=(1.,1.,3.)))
        @test length(blobs) == 1
        @test blobs[1].location == CartesianIndex((1,1,5))
        @test filter(b->b.amplitude > 0.1, blob_LoG(A, 2.0.^[0.5,0,1], edges=(true, true, true, false), σshape=(1.,1.,3.))) == blobs
        @test isempty(blob_LoG(A, 2.0.^[0,1], edges=(false, true, false, false), σshape=(1.,1.,3.)))
        @test length(blob_LoG([zeros(10); 1.0; 0.0], [4]; edges=true, rthresh=0)) > length(blob_LoG([zeros(10); 1.0; 0.0], [4]; edges=true))
    end
end
