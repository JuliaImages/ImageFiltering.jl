# patch for ColorVectorSpace 0.9
# for CVS < 0.9, we can just use the fallback solution in Distances
if isdefined(ImageCore.ColorVectorSpace, :⊙)
    # Because how abs2 calculated in color vector space is ambiguious, abs2(::RGB) is un-defined
    # since ColorVectorSpace 0.9
    # https://github.com/JuliaGraphics/ColorVectorSpace.jl/pull/131
    _abs2(c::Colorant) = mapreducec(abs2, +, 0, c)
    _abs2(x::Number) = abs2(x)

    _abs(c::Colorant) = mapreducec(abs, +, 0, c)
    _abs(x::Number) = abs(x)
else
    _abs2(c::Union{Colorant, Number}) = abs2(c)
    _abs(x::Number) = abs(x)
end
