@testset "GaborKernels" begin

@testset "Gabor" begin
    # Gray
    img = float32.(TestImages.shepp_logan(127))
    kern = Kernel.Gabor(size(img), 3.0f0, 0f0)
    img_freq = fft(channelview(img))
    kern_freq = ifftshift(fft(kern))
    out = abs.(ifft(img_freq .* kern_freq))

    # TODO(johnnychen94): currently Gabor can't be converted to CuArray directly and thus
    # FFTW is applied in the CPU side.
    img_cu = CuArray(img)
    img_freq = fft(channelview(img_cu))
    kern_freq = CuArray(ifftshift(fft(kern)))
    out_cu = abs.(ifft(img_freq .* kern_freq))

    @test out ≈ Array(out_cu)

    # RGB
    img = float32.(testimage("monarch"))
    kern = Kernel.Gabor(size(img), 3.0f0, 0f0)
    kern_freq = reshape(ifftshift(fft(kern)), 1, size(kern)...)
    img_freq = fft(channelview(img), 2:3)
    out = colorview(RGB, abs.(ifft(img_freq .* kern_freq)))

    img_cu = CuArray(img)
    kern_freq = CuArray(reshape(ifftshift(fft(kern)), 1, size(kern)...))
    img_freq = fft(channelview(img_cu), 2:3)
    out_cu = colorview(RGB, abs.(ifft(img_freq .* kern_freq)))

    @test out ≈ Array(out_cu)
end

@testset "LogGabor" begin
    # Gray
    img = float32.(TestImages.shepp_logan(127))
    kern = Kernel.LogGabor(size(img), 1.0f0/6, 0f0)
    kern_freq = OffsetArrays.no_offset_view(ifftshift(kern))
    img_freq = fft(channelview(img))
    out = abs.(ifft(kern_freq .* img_freq))

    # TODO(johnnychen94): remove this no_offset_view wrapper
    img_cu = CuArray(img)
    kern_freq = CuArray(OffsetArrays.no_offset_view(ifftshift(kern)))
    img_freq = fft(channelview(img_cu))
    out_cu = abs.(ifft(kern_freq .* img_freq))

    @test out ≈ Array(out_cu)

    # RGB
    img = float32.(testimage("monarch"))
    kern = Kernel.LogGabor(size(img), 1.0f0/6, 0f0)
    kern_freq = reshape(ifftshift(kern), 1, size(kern)...)
    img_freq = fft(channelview(img), 2:3)
    out = colorview(RGB, abs.(ifft(img_freq .* kern_freq)))

    img_cu = CuArray(img)
    kern_freq = CuArray(reshape(ifftshift(kern), 1, size(kern)...))
    img_freq = fft(channelview(img_cu), 2:3)
    out_cu = colorview(RGB, abs.(ifft(img_freq .* kern_freq)))

    @test out ≈ Array(out_cu)
end

end
