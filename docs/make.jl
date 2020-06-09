using Documenter, ImageFiltering

makedocs(
    modules  = [ImageFiltering, Kernel, KernelFactors, ImageFiltering.MapWindow],
    format   = Documenter.HTML(),
    sitename = "ImageFiltering",
    pages    = [
        "index.md", 
        "Function reference" => "function_reference.md"
    ]
)

deploydocs(
    repo   = "github.com/JuliaImages/ImageFiltering.jl.git",
    push_preview = true
)
