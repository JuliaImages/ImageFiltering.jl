using ImagesFiltering, ImagesCore, OffsetArrays, Colors
using Base.Test

imgf = zeros(5, 7); imgf[3,4] = 1
imgi = zeros(Int, 5, 7); imgi[3,4] = 1
imgg = fill(Gray(0), 5, 7); imgg[3,4] = 1
imgc = fill(RGB(0,0,0), 5, 7); imgc[3,4] = RGB(1,0,0)
kern = [0.1 0.2; 0.4 0.5]
kernel = OffsetArray(kern, -1:0, 1:2)
f32type(img) = f32type(eltype(img))
f32type{C<:Colorant}(::Type{C}) = base_colorant_type(C){Float32}
f32type{T<:Number}(::Type{T}) = Float32
for img in (copy(imgf), copy(imgi), copy(imgg), copy(imgc))
    targetimg = zeros(typeof(img[1]*kern[1]), size(img))
    targetimg[3:4,2:3] = rot180(kern)*img[3,4]
    @test imfilter(img, kernel) ≈ targetimg
    @test imfilter(f32type(img), img, kernel) ≈ float32(targetimg)
    for border in (Pad{:replicate}, Pad{:circular}, Pad{:symmetric}, Pad{:reflect}, Fill(zero(eltype(img))))
        @test imfilter(img, kernel, border) ≈ targetimg
        @test imfilter(f32type(img), img, kernel, border) ≈ float32(targetimg)
        for alg in (ImagesFiltering.FIR(), ImagesFiltering.FFT())
            @test imfilter(img, kernel, alg) ≈ targetimg
            @test imfilter(img, kernel, border, alg) ≈ targetimg
            @test imfilter(f32type(img), img, kernel, alg) ≈ float32(targetimg)
            @test imfilter(f32type(img), img, kernel, border, alg) ≈ float32(targetimg)
        end
    end
end
