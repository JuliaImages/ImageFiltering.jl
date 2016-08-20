## Laplacian

function _imfilter_inbounds!{N}(out, A::OffsetArray, kern::Laplacian{N}, R)
    ΔI = -CartesianIndex(A.offsets)
    _imfilter_inbounds!(out, (parent(A), ΔI), kern, R)
end
function _imfilter_inbounds!(out, A::AbstractArray, L::Laplacian, R)
    TT = eltype(out) # accumtype(eltype(out), eltype(A))
    n = 2*length(L.offsets)
    for I in R
        @inbounds tmp = convert(TT, - n * A[I])
        @inbounds for J in L.offsets
            tmp += A[I+J]
            tmp += A[I-J]
        end
        @inbounds out[I] = tmp
    end
    out
end
function _imfilter_inbounds!(out, Ashift::Tuple{AbstractArray,CartesianIndex}, L::Laplacian, R)
    A, ΔI = Ashift
    TT = eltype(out) # accumtype(eltype(out), eltype(A))
    n = 2*length(L.offsets)
    for I in R
        Ishift = I + ΔI
        @inbounds tmp = convert(TT, - n * A[Ishift])
        @inbounds for J in L.offsets
            tmp += A[Ishift+J]
            tmp += A[Ishift-J]
        end
        @inbounds out[I] = tmp
    end
    out
end
