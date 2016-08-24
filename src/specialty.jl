## Laplacian

function _imfilter_inbounds!(r, out, A::AbstractArray, L::Laplacian, border::NoPad, inds)
    TT = eltype(out) # accumtype(eltype(out), eltype(A))
    n = 2*length(L.offsets)
    R = CartesianRange(inds)
    @unsafe for I in R
        tmp = convert(TT, - n * A[I])
        for J in L.offsets
            tmp += A[I+J]
            tmp += A[I-J]
        end
        out[I] = tmp
    end
    out
end
