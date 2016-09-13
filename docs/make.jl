using Documenter, ImageFiltering

makedocs(modules  = [ImageFiltering, Kernel, KernelFactors],
         format   = Documenter.Formats.HTML,
         sitename = "ImageFiltering",
         pages    = ["index.md", "Function reference" => "function_reference.md"])

deploydocs(repo   = "github.com/JuliaImages/ImageFiltering.jl.git")
         # deps   = Deps.pip("mkdocs", "python-markdown-math"),
