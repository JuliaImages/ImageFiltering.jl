function imfilter_naive(img, kernel, padding_style=:replicate)
    padding_size = (size(kernel).-1).÷2
    out = similar(img)
    R = CartesianIndices(img)
    img = padarray(img, Pad(padding_style, padding_size, padding_size))
    offset = last(CartesianIndices(kernel))
    for p in R
        patch = img[p-offset:p+offset]
        patch = centered(patch)
        out[p] = sum(patch .* kernel)
    end
    out
end
