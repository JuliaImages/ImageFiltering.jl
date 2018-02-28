using Documenter, ImageFiltering

makedocs(modules  = [ImageFiltering, Kernel, KernelFactors, ImageFiltering.MapWindow],
         format   = :html,
         sitename = "ImageFiltering",
         pages    = ["index.md", "Function reference" => "function_reference.md"])

deploydocs(repo   = "github.com/JuliaImages/ImageFiltering.jl.git",
           target = "build",
           deps   = nothing,
           make   = nothing,
           julia  = "0.5")
