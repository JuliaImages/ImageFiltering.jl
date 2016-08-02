using ImagesFiltering, OffsetArrays
using Base.Test

imgf = zeros(10, 11); imgf[5,6] = 1
imgi = zeros(Int, 10, 11); imgi[5,6] = 1
imgg = fill(Gray(0), 10, 11); imgg[5,6] = 1
imgc = fill(RGB(0,0,0), 10, 11); imgc[5,6] = RGB(1,0,0)
kern = [0.1 0.2 0.3; 0.4 0.5 0.6]
kernel = OffsetArray(kern, (-2,0))
f16type(img) = f16type(eltype(img))
f16type{C<:Colorant}(::Type{C}) = base_colorant_type(C){Float16}
f16type{T<:Number}(::Type{T}) = Float16
for img in (imgf, imgi, imgg, imgc)
    for B in (Borders.Replicate, Borders.Circular, Borders.Symmetric, Borders.Reflect, Borders.Fill(0))
        imfilter(img, kernel)
        imfilter(img, kernel, ImagesFiltering.FIR())
        imfilter(img, kernel, ImagesFiltering.FFT())
        border = B(kernel)
        imfilter(img, kernel, border)
        imfilter(img, kernel, border, ImagesFiltering.FIR())
        imfilter(img, kernel, border, ImagesFiltering.FFT())
        imfilter(img, kernel, ImagesFiltering.FIR(), border)
        imfilter(img, kernel, ImagesFiltering.FFT(), border)
        imfilter(f16type(img), img, kernel)
        imfilter(f16type(img), img, kernel, ImagesFiltering.FIR())
        imfilter(f16type(img), img, kernel, border)
        imfilter(f16type(img), img, kernel, border, ImagesFiltering.FIR())
        imfilter(f16type(img), img, kernel, ImagesFiltering.FIR(), border)
    end
end
