@benchgroup "mapwindow" begin
    img1d = randn(1000)
    img2d = randn(30,30)
    img3d = randn(10,11,12)
    @bench "cheap f, tiny window" mapwindow(first, img1d, (1,))
    @bench "extrema" mapwindow(extrema, img2d, (5,5))
    @bench "median!" mapwindow(median!, img2d, (5,5))
    @bench "mean, small window" mapwindow(mean, img1d, (3,))
    @bench "mean, large window" mapwindow(mean, img3d, (5,5,5))
    @bench "expensive f" mapwindow(x -> quantile(vec(x), 0.7), img3d, (3,3,3))
end
