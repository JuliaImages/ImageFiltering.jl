using ImageFiltering, Statistics, Test

@testset "mapwindow" begin
    function groundtruth(f, A, window::Tuple)
        Aex = copy(A)
        hshift = map(x->x>>1+1, window)
        for Ishift in CartesianIndices(window)
            for I in CartesianIndices(size(A))
                Aex[I] = f(Aex[I], A[map(d->clamp(I[d]+Ishift[d]-hshift[d], 1, size(A,d)), 1:ndims(A))...])
            end
        end
        Aex
    end
    groundtruth(f, A, window) = groundtruth(f, A, (window...,))
    # 1d case
    A = [0.1,0.3,0.2,0.3,0.4]
    mm = mapwindow(extrema, A, 1)
    @test mm == [(a,a) for a in A]
    mm = mapwindow(extrema, A, 2)
    @test mm == [(0.1,0.1),(0.1,0.3),(0.2,0.3),(0.2,0.3),(0.3,0.4)]
    mm = mapwindow(extrema, A, 3)
    @test mm == [(0.1,0.3),(0.1,0.3),(0.2,0.3),(0.2,0.4),(0.3,0.4)]
    mm = mapwindow(extrema, A, 4)
    @test mm == [(0.1,0.3),(0.1,0.3),(0.1,0.3),(0.2,0.4),(0.2,0.4)]
    mm = mapwindow(extrema, A, 5)
    @test mm == [(0.1,0.3),(0.1,0.3),(0.1,0.4),(0.2,0.4),(0.2,0.4)]
    mm = mapwindow(extrema, A, 6)
    @test mm == [(0.1,0.3),(0.1,0.3),(0.1,0.4),(0.1,0.4),(0.2,0.4)]
    mm = mapwindow(extrema, A, 7)
    @test mm == [(0.1,0.3),(0.1,0.4),(0.1,0.4),(0.1,0.4),(0.2,0.4)]
    A = [0.1,0.3,0.5,0.4,0.2]
    mm = mapwindow(extrema, A, 1)
    @test mm == [(a,a) for a in A]
    mm = mapwindow(extrema, A, 2)
    @test mm == [(0.1,0.1),(0.1,0.3),(0.3,0.5),(0.4,0.5),(0.2,0.4)]
    mm = mapwindow(extrema, A, 3)
    @test mm == [(0.1,0.3),(0.1,0.5),(0.3,0.5),(0.2,0.5),(0.2,0.4)]
    mm = mapwindow(extrema, A, 4)
    @test mm == [(0.1,0.3),(0.1,0.5),(0.1,0.5),(0.2,0.5),(0.2,0.5)]
    mm = mapwindow(extrema, A, 5)
    @test mm == [(0.1,0.5),(0.1,0.5),(0.1,0.5),(0.2,0.5),(0.2,0.5)]
    mm = mapwindow(extrema, A, 6)
    @test mm == [(0.1,0.5),(0.1,0.5),(0.1,0.5),(0.1,0.5),(0.2,0.5)]
    mm = mapwindow(extrema, A, 7)
    @test mm == [(0.1,0.5),(0.1,0.5),(0.1,0.5),(0.1,0.5),(0.2,0.5)]
    # 2d case
    A = rand(5,5)/10
    A[2,2] = 0.8
    A[4,4] = 0.6
    for w in ((2,2), (2,3), (3,2), (3,3), (2,5))
        mm = mapwindow(extrema, A, w)
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
        mm = mapwindow(extrema, A, w)
        maxval = last.(mm)
        Amax = groundtruth(max, A, w)
        @test maxval == Amax
        minval = first.(mm)
        Amin = groundtruth(min, A, w)
        @test minval == Amin
    end

    # median
    for f in (median, median!)
        a = [1,1,1,2,2,2]
        @test mapwindow(f, a, -1:1) == a
        @test mapwindow(f, a, (-1:1,)) == a
        @test mapwindow(f, a, -2:2) == a
        @test mapwindow(f, a, -3:3) == a
        b = [1,100,1,2,-1000,2]
        @test mapwindow(f, b, -1:1) == [1,1,2,1,2,2]
        @test mapwindow(f, b, -2:2) == a

        A = [1 5 -2 3 7;
             2 0 3  4 4;
             3 3 6  2 5;
             1 -3 5 3 0]
        @test mapwindow(f, A, (3,3)) == [1 1 3 3 4;
                                         2 3 3 4 4;
                                         2 3 3 4 4;
                                         1 3 3 3 2]
    end

    # resolve_f
    replace_me(x) = 1
    ImageFiltering.MapWindow.replace_function(::typeof(replace_me)) = x -> 2
    @test mapwindow!(replace_me, randn(3), randn(3), (1,)) == [2,2,2]

    function groundtruth(f, A, window::Tuple, border, imginds)
        mapwindow(f,A,window,border)[imginds...]
    end
    for (f,img, window, imginds) ∈ [
            (mean, randn(10), (1,), (1:2:10,)),
            (median!, randn(10), (-1:1,), (1:2:8,)),
            (mean, randn(10), (-1:1,), (1:2:8,)),
            (mean, randn(10,5), (-1:1,0:0), (1:2:8,1:3)),
            (mean, randn(10,5), (-1:1,0:0), (Base.OneTo(2),1:3)),
        ]

        border = "replicate"
        expected = groundtruth(f,img,window,border,imginds)
        @test expected == @inferred mapwindow(f,img,window,border,imginds)
        out = similar(expected)
        @test expected == @inferred mapwindow!(f,out,img,window,border,imginds)
    end
    for (inds, args) ∈ [((Base.OneTo(3),), ("replicate", 2:2:7)),
                        ((2:7,), ("replicate", 2:7)),
                        ((Base.OneTo(10),), ())
                       ]
        @test inds == axes(mapwindow(mean, randn(10), (3,), args...))
    end

    img_48 = 10*collect(1:10)
    @test mapwindow(first, img_48, (1,), Inner()) == img_48
    res_48 = mapwindow(first, img_48, (0:1,), Inner())
    @test axes(res_48) === (Base.Slice(1:9),)
    @test res_48 == img_48[axes(res_48)...]
    inds_48 = 2:2:8
    @test mapwindow(first, img_48, (0:2,), Inner(), inds_48) == img_48[inds_48]

    @testset "desugaring window argument #58" begin
        img58 = rand(10)
        canonical_window = (-1:1,)
        truth = mapwindow(median, img58, canonical_window)
        for window in [3, (3,), [3], [-1:1], -1:1]
            @test truth == mapwindow(median, img58, window)
            out = similar(img58)
            @test truth == mapwindow!(median,out, img58, window)
        end
    end
end
