## Laplacian

# function !{S,T,N}(r::AbstractResource,
#                                   out::AbstractArray{S,N},
#                                   A::AbstractArray{T,N},
#                                   L::Tuple{Laplacian{N}})
#     _imfilter_padded!(r, out, A, L[1])
# end

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
    TT = accumtype(eltype(out), eltype(A))
    n = 2*length(L.offsets)
    for I in R
        Ishift = I + ΔI
        @inbounds tmp = convert(TT, - n * A[I])
        @inbounds for J in L.offsets
            tmp += A[Ishift+J]
            tmp += A[Ishift-J]
        end
        @inbounds out[I] = tmp
    end
    out
end
