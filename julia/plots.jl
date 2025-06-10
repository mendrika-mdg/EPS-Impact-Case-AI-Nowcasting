import Pkg; Pkg.add("PlotlyJS")
using Plots
plotlyjs()

# Use the same synthetic data
scatter(lon, lat, marker_z=prob,
    xlabel="Longitude", ylabel="Latitude",
    color=cgrad(:viridis, rev=true),
    clims=(0, 1),
    markersize=5,
    title="Interactive Core Probabilities",
    aspect_ratio=:equal)
