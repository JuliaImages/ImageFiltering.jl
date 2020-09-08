using ImageFiltering, Test
import StaticArrays
using Random

aif = detect_ambiguities(ImageFiltering, Kernel, KernelFactors, Base)
# Because StaticArrays has ambiguities with Base, we have to "subtract" these
asa = detect_ambiguities(StaticArrays, Base)
@test isempty(setdiff(aif, asa))

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
