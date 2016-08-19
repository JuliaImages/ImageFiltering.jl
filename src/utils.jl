"""
    centered(kernel) -> shiftedkernel

Shift the origin-of-coordinates to the center of `kernel`. The
center-element of `kernel` will be accessed by `shiftedkernel[0, 0,
...]`.

This function makes it easy to supply kernels using regular Arrays,
and provides compatibility with other languages that do not support
arbitrary indices.

See also: imfilter.
"""
centered(A::AbstractArray) = OffsetArray(A, map(n->-((n+1)>>1), size(A)))

dummyind(::Base.OneTo) = Base.OneTo(1)
dummyind(::AbstractUnitRange) = 0:0

dummykernel{N}(inds::Indices{N}) = similar(dims->ones(ntuple(d->1,Val{N})), map(dummyind, inds))

_reshape{_,N}(A::OffsetArray{_,N}, ::Type{Val{N}}) = A
_reshape{N}(A::OffsetArray, ::Type{Val{N}}) = OffsetArray(reshape(parent(A), Val{N}), fill_to_length(A.offsets, -1, Val{N}))
_reshape{N}(A::AbstractArray, ::Type{Val{N}}) = reshape(A, Val{N})

samedims{N}(::Type{Val{N}}, kernel) = _reshape(kernel, Val{N})
samedims{N}(::Type{Val{N}}, kernel::Tuple) = map(k->_reshape(k, Val{N}), kernel)
samedims{T,N}(::AbstractArray{T,N}, kernel) = samedims(Val{N}, kernel)

_tail(R::CartesianRange{CartesianIndex{0}}) = R
_tail(R::CartesianRange) = CartesianRange(CartesianIndex(tail(R.start.I)),
                                          CartesianIndex(tail(R.stop.I)))

to_ranges(R::CartesianRange) = map((b,e)->b:e, R.start.I, R.stop.I)
