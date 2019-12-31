using CSV
using NLsolve
using DataFrames
using Statistics
using Plots
using JLD2, FileIO
using Distributions
pyplot()
# %%
include("./classic/functions/FilterData.jl")
# %%
path = "./code/v1.2/input/rawdata.csv"
vars = [:y,:pi,:R,:n]
detrend = [:y,:n]
demean = [:pi,:R]
t1 = "1997Q1"
t2 = "2007Q4"
# %%
Y = linear_detrend(path,vars,detrend,demean,t1,t2)
plot(Y,label = ["y" "π" "R" "n"])
# %%
@load "./code/v1.2/input/parameters.jld2"
Ez = exp(σᶻ^2/2)
Φ(x) = cdf(Normal(0,1),x)
CDF(x) = Φ(log(max(x,0))/σᶻ)
PDF(x) = pdf(LogNormal(0,σᶻ),x)
I(x) = exp(σᶻ^2/2)*Φ((σᶻ^2-log(max(x,0)))/σᶻ)
int(x) = I(x) - x*(1-CDF(x))
include("./functions/Model.jl")
# %%
ss = steady_state()
# %%
bound = [
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
# %%
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
param = [
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
begin const
    n_obs = 4
    n_y = 21
    n_x = 4
    n_η = 6
    Nparam = 11
end
# %%
include("./functions/SMCFunctions.jl")
SMC()
# %%
M = [655.2962270911684 197.7887806758644 -107.93248958388439 647.7173022393374; 197.78878587711287 59.706558771213366 -32.57725165467117 195.5022442070402; -107.93248923935175 -32.577250687624684 17.777340411999962 -106.6841338376952; 647.7173002137473 195.5022385298836 -106.68413383714264 640.2276985330057]

eigvals(M)

[1.5721140987659025e-8, 0.0008153378517477329, 0.007217371085632351, 1372.999792082729]

any(eigvals(M) .< 0)
# %%
Nblock = 1
Npart = 1000
Nphi = 100
lambda = 2
Γ₀ = zeros(n_y,n_y)
Γ₁ = zeros(n_y,n_y)
Ψ = zeros(n_y,n_x)
Π = zeros(n_y,n_η)
ZZ = zeros(n_obs,n_y)
SDX = zeros(n_x,n_x)

RR = zeros(n_y,n_x)
GG = zeros(n_y,n_y)
eu = [0, 0]

set_obs_eq!(ZZ)

#MH parameters

c = 0.5
acpt = 0.25
trgt = 0.25

#Tempering schedule

phi = ((0:(Nphi-1))/(Nphi-1)).^lambda

#Storing the results

drawsMat = zeros(Nphi, Npart, Nparam)
weightsMat   = zeros(Npart, Nphi)
constMat    = zeros(Nphi)
id = zeros(Int64,Npart)
loglh   = zeros(Npart)
logpost = zeros(Npart)
nresamp = 0

cMat    = zeros(Nphi)
ESSMat  = zeros(Nphi)
acptMat = zeros(Nphi)
rsmpMat = zeros(Nphi)
acpt_part = zeros(Npart)
tune = Array{Any,1}(undef,4)

f(x1,x2) = Likelihoods!(x1,x2,Y,bound,param,Γ₀,Γ₁,Ψ,Π,SDX,GG,RR,ZZ,eu)
i = 2
SMCInit!(weightsMat,constMat,drawsMat,logpost,loglh,Npart,phi,f)
# %%
using BenchmarkTools
# %%
@time weightsMat[:, i] = weightsMat[:, i-1].*exp.((phi[i]-phi[i-1]).*loglh)
@time weightsMat[:, i] = @. weightsMat[:, i-1]*exp((phi[i]-phi[i-1])*loglh)



@time weightsMat[:, i] .= weightsMat[:, i-1].*exp.((phi[i]-phi[i-1]).*loglh)
@time @views weightsMat[:, i] .= weightsMat[:, i-1].*exp.((phi[i]-phi[i-1]).*loglh)
