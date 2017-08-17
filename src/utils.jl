"""
    centered(kernel) -> shiftedkernel

Shift the origin-of-coordinates to the center of `kernel`. The
center-element of `kernel` will be accessed by `shiftedkernel[0, 0,
...]`.

This function makes it easy to supply kernels using regular Arrays,
and provides compatibility with other languages that do not support
arbitrary indices.

See also: [`imfilter`](@ref).
"""
centered(A::AbstractArray) = OffsetArray(A, map(n->-((n+1)>>1), size(A)))

dummyind(::Base.OneTo) = Base.OneTo(1)
dummyind(::AbstractUnitRange) = 0:0

dummykernel(inds::Indices{N}) where {N} = similar(dims->ones(ntuple(d->1,Val{N})), map(dummyind, inds))

nextendeddims(inds::Indices) = sum(ind->length(ind)>1, inds)
nextendeddims(a::AbstractArray) = nextendeddims(indices(a))

function checkextended(inds::Indices, n)
    dimstr = n == 1 ? "dimension" : "dimensions"
    nextendeddims(inds) != n && throw(ArgumentError("need $n extended $dimstr, got indices $inds"))
    nothing
end
checkextended(a::AbstractArray, n) = checkextended(indices(a), n)

ranges(R::CartesianRange) = map(colon, R.start.I, R.stop.I)

_reshape(A::OffsetArray{_,N}, ::Type{Val{N}}) where {_,N} = A
_reshape(A::OffsetArray, ::Type{Val{N}}) where {N} = OffsetArray(reshape(parent(A), Val{N}), fill_to_length(A.offsets, -1, Val{N}))
_reshape(A::AbstractArray, ::Type{Val{N}}) where {N} = reshape(A, Val{N})

_vec(a::AbstractVector) = a
_vec(a::AbstractArray) = (checkextended(a, 1); a)
_vec(a::OffsetArray{_,1}) where {_} = a
function _vec(a::OffsetArray)
    inds = indices(a)
    checkextended(inds, 1)
    i = find(ind->length(ind)>1, inds)
    OffsetArray(vec(parent(a)), inds[i])
end

samedims(::Type{Val{N}}, kernel) where {N} = _reshape(kernel, Val{N})
samedims(::Type{Val{N}}, kernel::Tuple) where {N} = map(k->_reshape(k, Val{N}), kernel)
samedims(::AbstractArray{T,N}, kernel) where {T,N} = samedims(Val{N}, kernel)

_tail(R::CartesianRange{CartesianIndex{0}}) = R
_tail(R::CartesianRange) = CartesianRange(CartesianIndex(tail(R.start.I)),
                                          CartesianIndex(tail(R.stop.I)))

to_ranges(R::CartesianRange) = map((b,e)->b:e, R.start.I, R.stop.I)

# ensure that overflow is detected, by ensuring that it doesn't happen
# at intermediate stages of the computation
accumfilter(pixelval, filterval) = pixelval * filterval
const SmallInts = Union{UInt8,Int8,UInt16,Int16}
accumfilter(pixelval::SmallInts, filterval::SmallInts) = Int(pixelval)*Int(filterval)
# advice: don't use FixedPoint for the kernel
accumfilter(pixelval::N0f8, filterval::N0f8) = Float32(pixelval)*Float32(filterval)
accumfilter(pixelval::Colorant{N0f8}, filterval::N0f8) = float32(c)*Float32(filterval)

# In theory, the following might need to be specialized. For safety, make it a
# standalone function call.
safe_for_prod(x, ref) = oftype(ref, x)
