module ImagesFiltering

using ComputationalResources

# include("kernel.jl")
# using .Kernel
include("border.jl")
using .Border
using .Border: AbstractBorder, Replicate, Circular, Symmetric, Reflect, Inner, Fill

export Kernel, Border, imfilter, imfilter!, padarray

# deliberately don't export these, but it's expected that they will be used
abstract Alg
immutable FFT <: Alg end
immutable FIR <: Alg end

## imfilter with a finite impulse response kernel

function imfilter(img::AbstractArray, kernel, args...)
    imfilter(filter_type(img, kernel), img, kernel, args...)
end

function imfilter{T}(::Type{T}, img::AbstractArray, kernel, args...)
    imfilter!(similar(img, T), img, kernel, args...)
end

function imfilter!(out::AbstractArray, img::AbstractArray,
                   kernel::AbstractVector, dim::Integer, args...)
    1 <= dim <= ndims(img) || throw(ArgumentError("dim must be between 1 and $(ndims(img)), got $dim"))
    imfilter!(out, img, ntuple(d->d==dim ? kernel : 1), args...)
end

function imfilter!(out::AbstractArray, img::AbstractArray, kernel::AbstractArray, args...)
    imfilter!(out, img, factorkernel(kernel), args...)
end

function imfilter!(out::AbstractArray, img::AbstractArray, kernel::Tuple)
    imfilter!(out, img, kernel, Replicate(kernel))
end

function imfilter!(out::AbstractArray, img::AbstractArray, kernel::Tuple, border::AbstractBorder)
    imfilter!(out, img, kernel, border, filter_algorithm(out, img, kernel))
end

function imfilter!(out::AbstractArray, img::AbstractArray, kernel::Tuple, alg::Alg)
    imfilter!(out, img, kernel, Replicate(kernel), alg)
end

function imfilter!(out::AbstractArray, img::AbstractArray, kernel::Tuple, border::AbstractBorder, alg::Alg)
    imfilter!(CPU1(alg), out, img, kernel, border)
end

function imfilter!{S,T,N}(r::AbstractResource,
                          out::AbstractArray{S,N},
                          img::AbstractArray{T,N},
                          kernel::Tuple,
                          border::AbstractBorder)
    A = padarray(img, border)
    for kern in kernel
        imfilter!(r, out, A, kern)
    end
    out
end

function imfilter!{S,T,K,N}(::CPU1{FIR},
                            out::AbstractArray{S,N},
                            A::AbstractArray{T,N},
                            kern::AbstractArray{K,N})
    indso, indsA, indsk = indices(out), indices(A), indices(kern)
    for i = 1:N
        if      first(indsA[i]) != first(indso[i]) + first(indsk[i]) ||
                last(indsA[i])  != last(indso[i])  + last(indsk[i])
            throw(DimensionMismatch("output indices $indso and kernel indices $indsk do not agree with indices of padded input, $indsA"))
        end
    end
    (isempty(A) || isempty(kern)) && return out
    p = first(A) * first(kern)
    TT = typeof(p+p)
    for I in CartesianRange(indso)
        tmp = zero(TT)
        @inbounds for J in CartesianRange(indsk)
            tmp += A[I+J]*kern[J]
        end
        @inbounds out[I] = tmp
    end
    out
end



filter_type{S,T}(img::AbstractArray{S}, kernel::AbstractArray{T}) = typeof(zero(S)*zero(T) + zero(S)*zero(T))
filter_type{S,T}(img::AbstractArray{S}, kernel::Tuple{AbstractArray{T},Vararg{AbstractArray{T}}}) = typeof(zero(S)*zero(T) + zero(S)*zero(T))

factorkernel(kernel::AbstractArray) = (copy(kernel),)  # copy to ensure consistency

# Note that this isn't (and can't be) type stable
function factorkernel(kernel::StridedMatrix)
    SVD = svdfact(kernel)
    U, S, Vt = SVD[:U], SVD[:S], SVD[:Vt]
    separable = true
    EPS = sqrt(eps(eltype(S)))
    for i = 2:length(S)
        separable &= (abs(S[i]) < EPS)
    end
    separable || return (copy(kernel),)
    s = S[1]
    u, v = U[:,1:1], Vt[1:1,:]
    ss = sqrt(s)
    (ss*u, ss*v)
end

function factorkernel{T}(kernel::AbstractMatrix{T})
    m, n = length(indices(kernel,1)), length(indices(kernel,2))
    kern = Array{T}(m, n)
    copy!(kern, 1:m, 1:n, kernel, indices(kernel,1), indices(kernel,2))
    _factorkernel(factorkernel(kern), kernel)
end
_factorkernel(fk::Tuple{Matrix}, kernel::AbstractMatrix) = (copy(kernel),)
function _factorkernel(fk::Tuple{Matrix,Matrix}, kernel::AbstractMatrix)
    kern1 = fk[1]
    k1 = similar(kernel, eltype(kern1), (indices(kernel,1), 0:0))
    copy!(k1, indices(k1)..., kern1, indices(kern1)...)
    kern2 = fk[2]
    k2 = similar(kernel, eltype(kern1), (0:0, indices(kernel,2)))
    copy!(k2, indices(k2)..., kern2, indices(kern2)...)
    (k1, k2)
end

filter_algorithm(out, img, kernel) = FIR()

function __init__()
    # See ComputationalResources README for explanation
    push!(LOAD_PATH, dirname(@__FILE__))
    # if haveresource(ArrayFireLibs)
    #     @eval using DummyAF
    # end
    pop!(LOAD_PATH)
end

end # module
