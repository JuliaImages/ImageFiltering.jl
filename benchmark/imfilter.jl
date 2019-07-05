SUITE["imfilter"] = BenchmarkGroup(["imfilter"])
SUITE["imfilter_naive"] = BenchmarkGroup(["imfilter", "naive implementation"])
SUITE["imfilter_naive"]["CPU1"] = BenchmarkGroup(["CPU", "single thread"])
SUITE["imfilter"]["CPU1"] = BenchmarkGroup(["CPU", "single thread"])
# SUITE["imfilter"]["CPUThread"] = BenchmarkGroup(["CPU", "multi threads"])
# SUITE["imfilter"]["CUDA"] = BenchmarkGroup(["GPU", "CUDA"])
# SUITE["imfilter"]["ArrayFire"] = BenchmarkGroup(["GPU", "ArrayFire"])
# SUITE["imfilter"]["OpenCL"] = BenchmarkGroup(["GPU", "OpenCL"])

img_gray_256 = testimage("lena_gray_256")
img_rgb_256 = testimage("lena_color_256")
img_list_gray = [imresize(img_gray_256, ratio=ratio) for ratio in (0.5, 1.0, 2.0, 4.0)]
img_list_rgb = [imresize(img_rgb_256, ratio=ratio) for ratio in (0.5, 1.0, 2.0, 4.0)]

default_kernel = centered(ones((31, 31))./prod((31, 31)))

for (imfilter_impl, imfilter_name) in ((imfilter, "imfilter"),
                                     (imfilter_naive, "imfilter_naive"))
    SUITE[imfilter_name]["CPU1"]["image"] = BenchmarkGroup(["image"])
    for img in img_list_gray
        SUITE[imfilter_name]["CPU1"]["image"]["size", size(img)] =
            @benchmarkable ($imfilter_impl)($img, default_kernel)
    end

    SUITE[imfilter_name]["CPU1"]["kernel"] = BenchmarkGroup(["kernel"])
    for n in 1:2:19
        w = 2n+1
        kern = centered(ones((w, w))./prod((w, w)))
        SUITE[imfilter_name]["CPU1"]["kernel"]["size", (w, w)] =
            @benchmarkable ($imfilter_impl)(img_gray_256, $(kern))
    end

    SUITE[imfilter_name]["CPU1"]["padding"] = BenchmarkGroup(["padding style"])
    for padding_style in ("replicate", "circular", "symmetric", "reflect")
        # TODO: Inner(), NA(), NoPad()
        SUITE[imfilter_name]["CPU1"]["padding"][padding_style] =
            @benchmarkable ($imfilter_impl)(img_gray_256, default_kernel, $padding_style)
    end
end
