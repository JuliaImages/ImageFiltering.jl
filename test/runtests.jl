using ImageFiltering, Test
import StaticArrays
using Random

# Ambiguity test
if Base.VERSION >= v"1.6.0-DEV.1005"   # julia #37616
    @test isempty(detect_ambiguities(ImageFiltering, Kernel, KernelFactors))
else
    # Because StaticArrays may have ambiguities with Base, we have to "subtract" these
    aif = detect_ambiguities(ImageFiltering, Kernel, KernelFactors, Base)
    asa = detect_ambiguities(StaticArrays, Base)
    @test isempty(setdiff(aif, asa))
end

function typestring(::Type{T}) where T   # from https://github.com/JuliaImages/ImageCore.jl/pull/133
    buf = IOBuffer()
    show(buf, T)
    String(take!(buf))
end

include("border.jl")
include("nd.jl")
include("2d.jl")
include("triggs.jl")
include("cascade.jl")
include("specialty.jl")
include("gradient.jl")
include("mapwindow.jl")
include("basic.jl")
include("gabor.jl")

nothing
