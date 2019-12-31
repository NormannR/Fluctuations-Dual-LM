# %%
using SharedArrays
using Distributed
addprocs()
# %%
using CSV
using NLsolve
using DataFrames
using Statistics
using Plots
# %%
@everywhere begin
using JLD2, FileIO
using Distributions
using LinearAlgebra
using QuantEcon
end
pyplot()
# %%
include("./functions/FilterData.jl")
# %%
path = "./code/v1.2/parallel/input/rawdata.csv"
vars = [:y,:pi,:R,:n]
detrend = [:y,:n]
demean = [:pi,:R]
t1 = "1997Q1"
t2 = "2007Q4"
# %%
Y = linear_detrend(path,vars,detrend,demean,t1,t2)
@eval @everywhere Y = $Y
plot(Y,label = ["y" "π" "R" "n"])
# %%
@everywhere begin
    @load "./code/v1.2/parallel/input/parameters.jld2"
    Ez = exp(σᶻ^2/2)
    Φ(x) = cdf(Normal(0,1),x)
    CDF(x) = Φ(log(max(x,0))/σᶻ)
    PDF(x) = pdf(LogNormal(0,σᶻ),x)
    I(x) = exp(σᶻ^2/2)*Φ((σᶻ^2-log(max(x,0)))/σᶻ)
    int(x) = I(x) - x*(1-CDF(x))
end
include("./functions/Model.jl")
# %%
ss = steady_state()
@everywhere @eval ss = $ss
# %%
@everywhere bound = [
0. 1.;
0. 1.;
0. 1.;
0. 1.;
0. Inf;
0. Inf;
0. 1.;
0. Inf;
0. Inf;
0. Inf;
0. Inf ]

param_names = [
"ρ_A",
"ρ_μ",
"ρ_g",
"ρ_R",
"ρ_π",
"ρ_y",
"ψ",
"σ_A",
"σ_μ",
"σ_g",
"σ_m" ]
# %%
@everywhere param = [
0.84,
0.04,
0.90,
0.74,
1.82,
0.66,
0.79,
0.13,
0.26,
4.75,
0.08 ]
# %%
@everywhere begin const
    n_obs = 4
    n_y = 21
    n_x = 4
    n_η = 6
    Nparam = 11
end
# %%
include("./functions/SMCFunctions.jl")
SMC(Nphi = 100, Npart = 1000)
