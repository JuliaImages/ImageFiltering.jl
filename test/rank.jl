using ImageFiltering, Base.Test

@testset "rankfilter" begin
    function groundtruth(f, A, window::Tuple)
        Aex = copy(A)
        hshift = map(x->x>>1+1, window)
        for Ishift in CartesianRange(window)
            for I in CartesianRange(size(A))
                Aex[I] = f(Aex[I], A[map(d->clamp(I[d]+Ishift[d]-hshift[d], 1, size(A,d)), 1:ndims(A))...])
            end
        end
        Aex
    end
    groundtruth(f, A, window) = groundtruth(f, A, (window...,))
    # 1d case
    A = [0.1,0.3,0.2,0.3,0.4]
    mm = rankfilter(extrema, A, 1)
    @test mm == [(a,a) for a in A]
    mm = rankfilter(extrema, A, 2)
    @test mm == [(0.1,0.1),(0.1,0.3),(0.2,0.3),(0.2,0.3),(0.3,0.4)]
    mm = rankfilter(extrema, A, 3)
    @test mm == [(0.1,0.3),(0.1,0.3),(0.2,0.3),(0.2,0.4),(0.3,0.4)]
    mm = rankfilter(extrema, A, 4)
    @test mm == [(0.1,0.3),(0.1,0.3),(0.1,0.3),(0.2,0.4),(0.2,0.4)]
    mm = rankfilter(extrema, A, 5)
    @test mm == [(0.1,0.3),(0.1,0.3),(0.1,0.4),(0.2,0.4),(0.2,0.4)]
    mm = rankfilter(extrema, A, 6)
    @test mm == [(0.1,0.3),(0.1,0.3),(0.1,0.4),(0.1,0.4),(0.2,0.4)]
    mm = rankfilter(extrema, A, 7)
    @test mm == [(0.1,0.3),(0.1,0.4),(0.1,0.4),(0.1,0.4),(0.2,0.4)]
    A = [0.1,0.3,0.5,0.4,0.2]
    mm = rankfilter(extrema, A, 1)
    @test mm == [(a,a) for a in A]
    mm = rankfilter(extrema, A, 2)
    @test mm == [(0.1,0.1),(0.1,0.3),(0.3,0.5),(0.4,0.5),(0.2,0.4)]
    mm = rankfilter(extrema, A, 3)
    @test mm == [(0.1,0.3),(0.1,0.5),(0.3,0.5),(0.2,0.5),(0.2,0.4)]
    mm = rankfilter(extrema, A, 4)
    @test mm == [(0.1,0.3),(0.1,0.5),(0.1,0.5),(0.2,0.5),(0.2,0.5)]
    mm = rankfilter(extrema, A, 5)
    @test mm == [(0.1,0.5),(0.1,0.5),(0.1,0.5),(0.2,0.5),(0.2,0.5)]
    mm = rankfilter(extrema, A, 6)
    @test mm == [(0.1,0.5),(0.1,0.5),(0.1,0.5),(0.1,0.5),(0.2,0.5)]
    mm = rankfilter(extrema, A, 7)
    @test mm == [(0.1,0.5),(0.1,0.5),(0.1,0.5),(0.1,0.5),(0.2,0.5)]
    # 2d case
    A = rand(5,5)/10
    A[2,2] = 0.8
    A[4,4] = 0.6
    for w in ((2,2), (2,3), (3,2), (3,3), (2,5))
        mm = rankfilter(extrema, A, w)
        maxval = last.(mm)
        Amax = groundtruth(max, A, w)
        @test maxval == Amax
        minval = first.(mm)
        Amin = groundtruth(min, A, w)
        @test minval == Amin
    end
    # 3d case
    A = rand(5,5,5)/10
    A[2,2,2] = 0.7
    A[4,4,2] = 0.4
    A[2,2,4] = 0.5
    for w in ((2,2,2), (2,3,2), (3,2,2), (2,2,3), (3,3,3), (2,5,3))
        mm = rankfilter(extrema, A, w)
        maxval = last.(mm)
        Amax = groundtruth(max, A, w)
        @test maxval == Amax
        minval = first.(mm)
        Amin = groundtruth(min, A, w)
        @test minval == Amin
    end
end

nothing
