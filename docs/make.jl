using Documenter, Gomah

makedocs(;
    modules=[Gomah],
    format=Documenter.HTML(assets=String[]),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/terasakisatoshi/Gomah.jl/blob/{commit}{path}#L{line}",
    sitename="Gomah.jl",
    authors="SatoshiTerasaki <terasakisatoshi.math@gmail.com>",
)

deploydocs(;
    repo="github.com/terasakisatoshi/Gomah.jl",
)
