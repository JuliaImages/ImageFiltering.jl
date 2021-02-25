using ImageFiltering, Statistics, Test
using ImageFiltering: IdentityUnitRange

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

    @test_throws ArgumentError mapwindow(sum, ones(5,5), ()) # resolve_window

    # offsets
    @testset "offsets" begin
        n = 5
        arrays = [rand(n), rand(n,n), rand(n,n,n)]
        @testset "offsets ($f, $offset, $window, $dim)" for f in (extrema,maximum,minimum),
                                                            offset in -5:5,
                                                            window in [1:2:9; [0:2,-2:0]],
                                                            # broken: "reflect", NA(), NoPad()
                                                            border in ["replicate", "symmetric", Fill(randn()), Inner()],
                                                            (dim,a) in enumerate(arrays)
            offsets = ntuple(_->offset,dim)
            windows = ntuple(_->window,dim)
            winlen = window isa Number ? window : length(window)
            wrapped_f = x->f(x)
            for g in (f, wrapped_f)
                mw(a) = mapwindow(g, a, windows, border=border)

                if border == Inner() && winlen > n
                    @test_throws DimensionMismatch mw(a)
                    @test_throws DimensionMismatch mw(OffsetArray(a,offsets))
                else
                    @test OffsetArray(mw(a),offsets) == mw(OffsetArray(a,offsets))
                end
            end
        end
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

    function groundtruth(f, A, window::Tuple; indices=nothing)
        mapwindow(f,A,window)[indices...]
    end
    for (f,img, window, imginds) ∈ [
            (mean   , randn(10)  , (1,)      , (1:2:10,)),
            (median!, randn(10)  , (-1:1,)   , (1:2:8,)),
            (mean   , randn(10)  , (-1:1,)   , (1:2:8,)),
            (mean   , randn(10,5), (-1:1,0:0), (1:2:8,1:3)),
            (mean   , randn(10,5), (-1:1,0:0), (Base.OneTo(2),1:3)),
        ]

        expected = groundtruth(f,img,window,indices=imginds)
        @test expected == @inferred mapwindow(f,img,window,indices=imginds)
        out = similar(expected)
        @test expected == @inferred mapwindow!(f,out,img,window,indices=imginds)
    end
    for (inds, kw) ∈ [((Base.OneTo(3),), (border="replicate", indices=2:2:7)),
                        (axes(2:7), (indices=2:7,)),
                        ((Base.OneTo(10),), ())
                       ]
        @test inds == axes(mapwindow(mean, randn(10), (3,); kw...))
    end

    img_48 = 10*collect(1:10)
    @test mapwindow(first, img_48, (1,), border=Inner()) == img_48
    res_48 = mapwindow(first, img_48, (0:1,), border=Inner())
    @test axes(res_48) == (IdentityUnitRange(1:9),)
    @test res_48 == img_48[axes(res_48)...]
    inds_48 = 2:2:8
    @test mapwindow(first, img_48, (0:2,),
                    border=Inner(),
                    indices=inds_48) == img_48[inds_48]
    res_shifted = mapwindow(minimum, img_48, -2:0, border=Inner())
    @test res_shifted == OffsetArray(img_48[1:8], 3:10)

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

    @testset ">3D mapwindow #105" begin
        img105 = ones(5,5,5,5,5)
        out105 = mapwindow(sum, img105, [1,1,1,1,3]; border = Fill(0))
        foo = centered([1])
        bar = centered([1,1,1])
        @test out105 == imfilter(img105, kernelfactors((foo, foo, foo, foo, bar)), Fill(0))
        out1052 = mapwindow(sum, img105, [1,1,1,1,3])
        @test all(isequal(3.0), out1052)
    end
end
