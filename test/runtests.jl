using ImagesFiltering, Base.Test

aif = detect_ambiguities(ImagesFiltering, Kernel, KernelFactors, Base)
# Because StaticArrays has ambiguities with Base, we have to "subtract" these
asa = detect_ambiguities(StaticArrays, Base)
@test isempty(setdiff(aif, asa))

include("border.jl")
include("2d.jl")
include("triggs.jl")
include("cascade.jl")
include("specialty.jl")
include("basic.jl")
include("deprecated.jl")

nothing
