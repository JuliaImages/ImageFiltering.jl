using ImageFiltering, OffsetArrays, Colors, FixedPointNumbers, Random
using Test

@testset "Border" begin
    borders = []
    push!(borders, Inner())
    push!(borders, Fill(0, (1,2), (3,4)))
    for _ in 1:10
        s = rand([:replicate, :circular, :symmetric, :reflect])
        pad = Pad(s, (rand(0:4),rand(0:4)), (rand(0:4), rand(0:4)))
        push!(borders, pad)
    end
    @testset "BorderArray consistent with padarray $b" for b in borders
        A = randn(5,5)
        B = BorderArray(A, b)
        @test padarray(A, b) == B
        for I in CartesianIndices(A)
            @test A[I] == B[I]
        end
    end

    @testset "padarray" begin
        A = reshape(1:25, 5, 5)
        @test @inferred(padarray(A, Fill(0,(2,2),(2,2)))) == OffsetArray(
            [0  0  0   0   0   0   0  0  0;
             0  0  0   0   0   0   0  0  0;
             0  0  1   6  11  16  21  0  0;
             0  0  2   7  12  17  22  0  0;
             0  0  3   8  13  18  23  0  0;
             0  0  4   9  14  19  24  0  0;
             0  0  5  10  15  20  25  0  0;
             0  0  0   0   0   0   0  0  0;
             0  0  0   0   0   0   0  0  0], (-2,-2))
        @test @inferred(padarray(A, Pad(:replicate,(2,2),(2,2)))) == OffsetArray(
            [ 1  1  1   6  11  16  21  21  21;
              1  1  1   6  11  16  21  21  21;
              1  1  1   6  11  16  21  21  21;
              2  2  2   7  12  17  22  22  22;
              3  3  3   8  13  18  23  23  23;
              4  4  4   9  14  19  24  24  24;
              5  5  5  10  15  20  25  25  25;
              5  5  5  10  15  20  25  25  25;
              5  5  5  10  15  20  25  25  25], (-2,-2))
        @test @inferred(padarray(A, Pad(:circular,(2,2),(2,2)))) == OffsetArray(
            [19  24  4   9  14  19  24  4   9;
             20  25  5  10  15  20  25  5  10;
             16  21  1   6  11  16  21  1   6;
             17  22  2   7  12  17  22  2   7;
             18  23  3   8  13  18  23  3   8;
             19  24  4   9  14  19  24  4   9;
             20  25  5  10  15  20  25  5  10;
             16  21  1   6  11  16  21  1   6;
             17  22  2   7  12  17  22  2   7], (-2,-2))
        @test @inferred(padarray(A, Pad(:symmetric,(2,2),(2,2)))) == OffsetArray(
            [ 7  2  2   7  12  17  22  22  17;
              6  1  1   6  11  16  21  21  16;
              6  1  1   6  11  16  21  21  16;
              7  2  2   7  12  17  22  22  17;
              8  3  3   8  13  18  23  23  18;
              9  4  4   9  14  19  24  24  19;
             10  5  5  10  15  20  25  25  20;
             10  5  5  10  15  20  25  25  20;
              9  4  4   9  14  19  24  24  19], (-2,-2))
        @test @inferred(padarray(A, Pad(:reflect,(2,2),(2,2)))) == OffsetArray(
            [13   8  3   8  13  18  23  18  13;
             12   7  2   7  12  17  22  17  12;
             11   6  1   6  11  16  21  16  11;
             12   7  2   7  12  17  22  17  12;
             13   8  3   8  13  18  23  18  13;
             14   9  4   9  14  19  24  19  14;
             15  10  5  10  15  20  25  20  15;
             14   9  4   9  14  19  24  19  14;
             13   8  3   8  13  18  23  18  13], (-2,-2))
        ret = @test_throws ArgumentError padarray(A, Fill(0))
        @test occursin("lacks the proper padding", ret.value.msg)
        for Style in (:replicate, :circular, :symmetric, :reflect)
            ret = @test_throws ArgumentError padarray(A, Pad(Style,(1,1,1),(1,1,1)))
            @test occursin("lacks the proper padding", ret.value.msg)
            ret = @test_throws ArgumentError padarray(A, Pad(Style))
            @test occursin("lacks the proper padding", ret.value.msg)
        end
        # same as above, but input arrays are of type OffsetArray
        Ao =  OffsetArray(reshape(1:25, 5, 5),-1,-1)
        @test @inferred(padarray(Ao, Fill(0,(2,2),(2,2)))) == OffsetArray(
            [0  0  0   0   0   0   0  0  0;
             0  0  0   0   0   0   0  0  0;
             0  0  1   6  11  16  21  0  0;
             0  0  2   7  12  17  22  0  0;
             0  0  3   8  13  18  23  0  0;
             0  0  4   9  14  19  24  0  0;
             0  0  5  10  15  20  25  0  0;
             0  0  0   0   0   0   0  0  0;
             0  0  0   0   0   0   0  0  0], (-3,-3))
        @test @inferred(padarray(Ao, Pad(:replicate,(2,2),(2,2)))) == OffsetArray(
            [ 1  1  1   6  11  16  21  21  21;
              1  1  1   6  11  16  21  21  21;
              1  1  1   6  11  16  21  21  21;
              2  2  2   7  12  17  22  22  22;
              3  3  3   8  13  18  23  23  23;
              4  4  4   9  14  19  24  24  24;
              5  5  5  10  15  20  25  25  25;
              5  5  5  10  15  20  25  25  25;
              5  5  5  10  15  20  25  25  25], (-3,-3))
        @test @inferred(padarray(Ao, Pad(:circular,(2,2),(2,2)))) == OffsetArray(
            [19  24  4   9  14  19  24  4   9;
             20  25  5  10  15  20  25  5  10;
             16  21  1   6  11  16  21  1   6;
             17  22  2   7  12  17  22  2   7;
             18  23  3   8  13  18  23  3   8;
             19  24  4   9  14  19  24  4   9;
             20  25  5  10  15  20  25  5  10;
             16  21  1   6  11  16  21  1   6;
             17  22  2   7  12  17  22  2   7], (-3,-3))
        @test @inferred(padarray(Ao, Pad(:symmetric,(2,2),(2,2)))) == OffsetArray(
            [ 7  2  2   7  12  17  22  22  17;
              6  1  1   6  11  16  21  21  16;
              6  1  1   6  11  16  21  21  16;
              7  2  2   7  12  17  22  22  17;
              8  3  3   8  13  18  23  23  18;
              9  4  4   9  14  19  24  24  19;
             10  5  5  10  15  20  25  25  20;
             10  5  5  10  15  20  25  25  20;
              9  4  4   9  14  19  24  24  19], (-3,-3))
        @test @inferred(padarray(Ao, Pad(:reflect,(2,2),(2,2)))) == OffsetArray(
            [13   8  3   8  13  18  23  18  13;
             12   7  2   7  12  17  22  17  12;
             11   6  1   6  11  16  21  16  11;
             12   7  2   7  12  17  22  17  12;
             13   8  3   8  13  18  23  18  13;
             14   9  4   9  14  19  24  19  14;
             15  10  5  10  15  20  25  20  15;
             14   9  4   9  14  19  24  19  14;
             13   8  3   8  13  18  23  18  13], (-3,-3))
        ret = @test_throws ArgumentError padarray(Ao, Fill(0))
        @test occursin("lacks the proper padding", ret.value.msg)
        for Style in (:replicate, :circular, :symmetric, :reflect)
            ret = @test_throws ArgumentError padarray(Ao, Pad(Style,(1,1,1),(1,1,1)))
            @test occursin("lacks the proper padding", ret.value.msg)
            ret = @test_throws ArgumentError padarray(Ao, Pad(Style))
            @test occursin("lacks the proper padding", ret.value.msg)
        end
        # arrays smaller than the padding
        A = [1 2; 3 4]
        @test @inferred(padarray(A, Pad(:replicate,(3,3),(3,3)))) == OffsetArray(
            [ 1  1  1  1  2  2  2  2;
              1  1  1  1  2  2  2  2;
              1  1  1  1  2  2  2  2;
              1  1  1  1  2  2  2  2;
              3  3  3  3  4  4  4  4;
              3  3  3  3  4  4  4  4;
              3  3  3  3  4  4  4  4;
              3  3  3  3  4  4  4  4], (-3,-3))
        @test @inferred(padarray(A, Pad(:circular,(3,3),(3,3)))) == OffsetArray(
            [ 4  3  4  3  4  3  4  3;
              2  1  2  1  2  1  2  1;
              4  3  4  3  4  3  4  3;
              2  1  2  1  2  1  2  1;
              4  3  4  3  4  3  4  3;
              2  1  2  1  2  1  2  1;
              4  3  4  3  4  3  4  3;
              2  1  2  1  2  1  2  1], (-3,-3))
        @test @inferred(padarray(A, Pad(:symmetric,(3,3),(3,3)))) == OffsetArray(
            [ 4  4  3  3  4  4  3  3;
              4  4  3  3  4  4  3  3;
              2  2  1  1  2  2  1  1;
              2  2  1  1  2  2  1  1;
              4  4  3  3  4  4  3  3;
              4  4  3  3  4  4  3  3;
              2  2  1  1  2  2  1  1;
              2  2  1  1  2  2  1  1], (-3,-3))
        @test @inferred(padarray(A, Pad(:reflect,(3,3),(3,3)))) == OffsetArray(
            [ 4  3  4  3  4  3  4  3;
              2  1  2  1  2  1  2  1;
              4  3  4  3  4  3  4  3;
              2  1  2  1  2  1  2  1;
              4  3  4  3  4  3  4  3;
              2  1  2  1  2  1  2  1;
              4  3  4  3  4  3  4  3;
              2  1  2  1  2  1  2  1], (-3,-3))
        for Style in (:replicate, :circular, :symmetric, :reflect)
            @test @inferred(padarray(A, Pad(Style,(0,0), (0,0)))) == A
        end
        @test @inferred(padarray(A, Fill(0, (0,0), (0,0)))) == A
        @test @inferred(padarray(A, Pad(:replicate,(1,2), (2,0)))) == OffsetArray([1 1 1 2; 1 1 1 2; 3 3 3 4; 3 3 3 4; 3 3 3 4], (-1,-2))
        @test @inferred(padarray(A, Pad(:circular,(2,1), (0,2)))) == OffsetArray([2 1 2 1 2; 4 3 4 3 4; 2 1 2 1 2; 4 3 4 3 4], (-2,-1))
        @test @inferred(padarray(A, Pad(:symmetric,(1,2), (2,0)))) == OffsetArray([2 1 1 2; 2 1 1 2; 4 3 3 4; 4 3 3 4; 2 1 1 2], (-1,-2))
        @test @inferred(padarray(A, Fill(-1,(1,2), (2,0)))) == OffsetArray([-1 -1 -1 -1; -1 -1 1 2; -1 -1 3 4; -1 -1 -1 -1; -1 -1 -1 -1], (-1,-2))
        A = [1 2 3; 4 5 6]
        @test @inferred(padarray(A, Pad(:reflect,(1,2), (2,0)))) == OffsetArray([6 5 4 5 6; 3 2 1 2 3; 6 5 4 5 6; 3 2 1 2 3; 6 5 4 5 6], (-1,-2))
        A = [1 2; 3 4]
        @test @inferred(padarray(A, Pad(1,1))) == OffsetArray([1 1 2 2; 1 1 2 2; 3 3 4 4; 3 3 4 4], (-1,-1))
        @test @inferred(padarray(A, Pad(:circular,(1,1), ()))) == OffsetArray([4 3 4; 2 1 2; 4 3 4], (-1,-1))
        @test @inferred(padarray(A, Pad(:symmetric,(), (1,1)))) == [1 2 2; 3 4 4; 3 4 4]
        A = ["a" "b"; "c" "d"]
        @test @inferred(padarray(A, Pad(1,1))) == OffsetArray(["a" "a" "b" "b"; "a" "a" "b" "b"; "c" "c" "d" "d"; "c" "c" "d" "d"], (-1,-1))
        @test_throws ErrorException padarray(A, Pad(:unknown,1,1))
        A = trues(3,3)
        @test isa(parent(padarray(A, Pad((1,2), (2,1)))), BitArray{2})
        A = falses(10,10,10)
        B = view(A,1:8,1:8,1:8)
        @test isa(parent(padarray(A, Pad((1,1,1)))), BitArray{3})
        @test isa(parent(padarray(B, Pad((1,1,1)))), BitArray{3})
        A = reshape(1:15, 3, 5)
        B = @inferred(padarray(A, Inner((1,1))))
        @test B == A
        # test that it's a copy
        B[1,1] = 0
        @test B != A
        A = rand(RGB{N0f8}, 3, 5)
        ret = @test_throws ErrorException padarray(A, Fill(0, (0,0), (0,0)))
        # FIXME: exact phrase depends on showarg in Interpolations
        # @test occursin("element type ColorTypes.RGB", ret.value.msg)
        # This is a temporary substitute:
        @test occursin("RGB", ret.value.msg)
        A = bitrand(3, 5)
        ret = @test_throws ErrorException padarray(A, Fill(7, (0,0), (0,0)))
        @test occursin("element type Bool", ret.value.msg)
        @test isa(parent(padarray(A, Fill(false, (1,1), (1,1)))), BitArray)
    end

    @testset "Pad" begin
        @test Pad(:replicate,[1,2], [5,3]) == Pad(:replicate,(1,2), (5,3))
        @test @inferred(Pad{2}(:replicate, [1,2], [5,3])) == Pad(:replicate,(1,2), (5,3))
        @eval @test_throws MethodError Pad{3}(:replicate, [1,2], [5,3])
        @test @inferred(Pad(:circular)(rand(3,5))) == Pad(:circular, (0,0),(3,5))
        @test @inferred(Pad(:circular)(centered(rand(3,5)))) == Pad{2}(:circular, (1,2),(1,2))
        @test @inferred(Pad(:symmetric)(Kernel.Laplacian())) == Pad{2}(:symmetric, (1,1),(1,1))
        @test @inferred(Pad(:symmetric)(Kernel.Laplacian(), rand(5,5), Algorithm.FIR())) == Pad(:symmetric,(1,1),(1,1))

        a = reshape(1:15, 3, 5)
        targetinds = (OffsetArray([1,1,2,3,3], 0:4), OffsetArray([1:5;], 1:5))
        ret = @test_throws ArgumentError ImageFiltering.padindices(rand(3,5), Pad(:replicate,(1,)))
        @test occursin("lacks the proper padding sizes", ret.value.msg)
    end


    @testset "Inner" begin
        @test @inferred(Inner((1,2))) == Inner((1,2), (1,2))
        @test @inferred(Inner((1,2), ())) == Inner((1,2), (0,0))
        @test @inferred(Inner((), (1,2))) == Inner((0,0), (1,2))
        @test @inferred(Inner{2}([1,2],[5,6])) == Inner((1,2), (5,6))
        @test Inner([1,2],[5,6]) == Inner((1,2), (5,6))
        @test @inferred(Inner()(rand(3,5))) == Inner((0,0),(3,5))
        @test @inferred(Inner()(centered(rand(3,5)))) == Inner((1,2),(1,2))
    end

    @testset "Fill" begin
        @test Fill(1,[1,2],[5,6]) == Fill(1, (1,2), (5,6))
        @test Fill(1,(3,2)) == Fill(1, (3,2), (3,2))
        @test @inferred(Fill(-1)(rand(3,5))) == Fill(-1, (0,0), (3,5))
        @test @inferred(Fill(-1)(centered(rand(3,5)))) == Fill(-1, (1,2), (1,2))
        @test @inferred(Fill(-1, rand(3,5))) == Fill(-1, (0,0), (3,5))
        @test @inferred(Fill(-1, centered(rand(3,5)))) == Fill(-1, (1,2), (1,2))
        @test @inferred(Fill(-1)(rand(3,5))) == Fill(-1, (0,0), (3,5))
        @test @inferred(Fill(-1)(centered(rand(3,5)))) == Fill(-1, (1,2), (1,2))
        @test @inferred(Fill(2)(Kernel.Laplacian(), rand(5,5), Algorithm.FIR())) == Fill(2, (1,1),(1,1))
    end

    @testset "misc" begin
        a0 = reshape([1])  # 0-dimensional
        @test ImageFiltering.accumulate_padding((), a0) == ()
        @test ImageFiltering.accumulate_padding((0:1, -1:1), a0) == (0:1, -1:1)
    end
end

nothing
