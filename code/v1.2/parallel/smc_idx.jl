#Model with indexed F on ϕ and A
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
    @load "./code/v1.2/parallel/input/parameters_idx.jld2"
    Ez = exp(σᶻ^2/2)
    Φ(x) = cdf(Normal(0,1),x)
    CDF(x) = Φ(log(max(x,0))/σᶻ)
    PDF(x) = pdf(LogNormal(0,σᶻ),x)
    I(x) = exp(σᶻ^2/2)*Φ((σᶻ^2-log(max(x,0)))/σᶻ)
    int(x) = I(x) - x*(1-CDF(x))
end
include("./functions/Model_idx.jl")
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
# %%
meta_param = [(80,10000),(160,5000),(640,1250),(1280,625)]
Nrun = 20
Nmeta = size(meta_param,1)
mu_meta = zeros(Nmeta,Nrun,Nparam)
sig_meta = zeros(Nmeta,Nrun,Nparam)
# %%
for i = Nmeta:Nmeta
    for j = 1:Nrun
        println("Parameters $(meta_param[i]), Run $j")
        _, _, mu_meta[i,j,:], sig_meta[i,j,:] = SMC(Nphi = meta_param[i][1], Npart = meta_param[i][2])
    end
end
# %%
@save "./code/v1.2/parallel/input/meta.jld2" mu_meta sig_meta
# %%
@load "./code/v1.2/parallel/input/meta.jld2" mu_meta sig_meta
# %%
pgfplots()
# %%
using LaTeXStrings
# %%
pyplot()
# %%
# 800,000 likelihood calls
meta_param = [(80,10000),(160,5000),(640,1250),(1280,625)]
Nrun = 20
Nmeta = size(meta_param,1)
param_labels = [
L"$\rho_{A}$",
L"$\rho_\mu$",
L"$\rho_g$",
L"$\rho_R$",
L"$\rho_\pi$",
L"$\rho_y$",
L"$\psi$",
L"$\sigma_A$",
L"$\sigma_\mu$",
L"$\sigma_g$",
L"$\sigma_m$" ]
# %%
markers = [:circle, :rect, :diamond, :dtriangle, :cross]
x = reshape(mean(sig_meta,dims=2).^2 ./ var(mu_meta,dims=2),Nmeta,Nparam)
scatter(x'[:,1], yaxis=:log, color = :black, marker = markers[1], label = "$(meta_param[1])")
for i = 2:Nmeta
    scatter!(x'[:,i], yaxis=:log, color = :black, marker = markers[i], label = "$(meta_param[i])")
end
xticks!(1:11,param_labels)
xlabel!("Estimated parameters", )
ylabel!(L"$\frac{ \mathbb{V} \left( \bar{\theta} \right)}{\mathbb{V}_{\pi} \left( \theta \right) }$")
# %%
savefig("./latex/figures_new/tuning_meta_8e5.pdf")
# %%
display(mean(mu_meta[2,:,:],dims=1))
display(mean(sig_meta[2,:,:],dims=1))
# %%
# 100,000 likelihood evaluations
meta_param = [(400,250),(200,500),(100,1000),(50,2000)]
Nrun = 20
Nmeta = size(meta_param,1)
mu_meta = zeros(Nmeta,Nrun,Nparam)
sig_meta = zeros(Nmeta,Nrun,Nparam)
# %%
for i = 1:Nmeta
    for j = 1:Nrun
        println("Parameters $(meta_param[i]), Run $j")
        _, _, mu_meta[i,j,:], sig_meta[i,j,:] = SMC(Nphi = meta_param[i][1], Npart = meta_param[i][2])
    end
end
# %%
@save "./code/v1.2/parallel/input/meta_1e5.jld2" mu_meta sig_meta
# %%
@load "./code/v1.2/parallel/input/meta_1e5.jld2" mu_meta sig_meta
# %%
param_labels = [
L"$\rho_{A}$",
L"$\rho_\mu$",
L"$\rho_g$",
L"$\rho_R$",
L"$\rho_\pi$",
L"$\rho_y$",
L"$\psi$",
L"$\sigma_A$",
L"$\sigma_\mu$",
L"$\sigma_g$",
L"$\sigma_m$" ]
# %%
markers = [:circle, :rect, :diamond, :dtriangle, :cross]
x = reshape(mean(sig_meta,dims=2).^2 ./ var(mu_meta,dims=2),Nmeta,Nparam)
scatter(x'[:,1], yaxis=:log, color = :black, marker = markers[1], label = "$(meta_param[1])")
for i = 2:Nmeta
    scatter!(x'[:,i], yaxis=:log, color = :black, marker = markers[i], label = "$(meta_param[i])")
end
xticks!(1:11,param_labels)
xlabel!("Estimated parameters", )
ylabel!(L"$\frac{ \mathbb{V} \left( \bar{\theta} \right)}{\mathbb{V}_{\pi} \left( \theta \right) }$")
# %%
savefig("./latex/figures_new/tuning_meta_1e5.pdf")
# %%
display(mean(mu_meta[4,:,:],dims=1))
display(mean(sig_meta[4,:,:],dims=1))
