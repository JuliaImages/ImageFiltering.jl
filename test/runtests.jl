using ImageFiltering, ImageCore, ImageBase
using OffsetArrays
using Test
using TestImages
using ImageQualityIndexes
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

include("compat.jl")
include("border.jl")
include("nd.jl")
include("2d.jl")
include("triggs.jl")
include("cascade.jl")
include("specialty.jl")
include("gradient.jl")
include("mapwindow.jl")
include("extrema.jl")
include("basic.jl")
include("gabor.jl")
include("models.jl")


CUDA_INSTALLED = false
try
    global CUDA_INSTALLED
    # This errors with `IOError` when nvidia driver is not available,
    # in which case we don't even need to try `using CUDA`
    run(pipeline(`nvidia-smi`, stdout=devnull, stderr=devnull))
    push!(LOAD_PATH, "@v#.#") # force using global CUDA installation

    @eval using CUDA
    CUDA.allowscalar(false)
    CUDA_INSTALLED = true
catch e
    e isa Base.IOError || @warn e LOAD_PATH
end
CUDA_FUNCTIONAL = CUDA_INSTALLED && CUDA.functional()
if CUDA_FUNCTIONAL
    @info "CUDA test: enabled"
    @testset "CUDA" begin
        include("cuda/runtests.jl")
    end
else
    @warn "CUDA test: disabled"
end
nothing
