module TestSlidingWindow
using ImageFiltering
using ImageFiltering: CopyProvisioning, ViewProvisioning
using Test

@testset "sliding_window" begin
    base = collect(1:3)
    sw = @inferred sliding_window(base, (0:0,))
    @test sw == [[1],[2],[3]]

    sw = @inferred sliding_window(base, (0:1,))
    @test sw == [[1,2],[2,3],[3,3]]

    sw = @inferred sliding_window(base, (0:1,), border=:circular)
    @test sw == [[1,2],[2,3],[3,1]]
    @test eltype(sw) <: AbstractVector{Int}

    sw = @inferred sliding_window(base, (0:1,), border=:circular, provisioning=CopyProvisioning())
    @test sw == [[1,2],[2,3],[3,1]]
    @test eltype(sw) == Vector{Int}

    sw = @inferred sliding_window(base, (0:1,), border=:circular, provisioning=ViewProvisioning())
    @test sw == [[1,2],[2,3],[3,1]]
    @test eltype(sw) <: SubArray

    sw = @inferred sliding_window([1 2 3; 4 5 6], (0:0, -1:1), border=:circular, provisioning=CopyProvisioning())
    @test eltype(sw) == Matrix{Int}
    @test @inferred(sw[1,1]) == [3 1 2]
    @test sw[1,2] == [1 2 3]
    @test sw[1,3] == [2 3 1]
    @test sw[2,1] == [6 4 5]
    @test sw[2,2] == [4 5 6]
    @test sw[2,3] == [5 6 4]

    @test map(=>, 3:5, sliding_window(30:10:50, 0:1), ) == [
     3 => [30, 40],
     4 => [40, 50],
     5 => [50, 50],
    ]

    @testset "API" begin
        base = 1:3
        expected = [
         [2, 1, 2],
         [1, 2, 3],
         [2, 3, 2],
        ]
        for border in [:reflect, Fill(2), Fill(2, [1], [2])]
            for window in [(-1:1,), (3,), -1:1, 3]
                @test sliding_window(base, window, border=border) == expected
                for f in [padarray, BorderArray]
                    @test sliding_window(f, base, window, border=:reflect) == expected
                    for provisioning in [:copy, :view, ViewProvisioning(), CopyProvisioning()]
                        @test sliding_window(f, base, window, border=:reflect, provisioning=provisioning) == expected
                    end
                end
                for provisioning in [:copy, :view, ViewProvisioning(), CopyProvisioning()]
                    @test sliding_window(base, window, border=:reflect, provisioning=provisioning) == expected
                end
            end
        end
    end
end
end#module
