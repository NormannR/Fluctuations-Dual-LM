import Pkg; Pkg.activate(".")
path = pwd() * "/functions"
push!(LOAD_PATH,path)
# %%
using Revise
# %%
using SharedArrays
using Distributed
using Dual
using Classic
using FilterData
using Distributions
using SMC_aux
using SMC_main
using JLD2, FileIO
# %%
using Plots
using LaTeXStrings
pyplot()
# %%
include("./functions/OutputFunctions.jl")
# %%
prior = [
"Uniform" 0. 1. ;
"Uniform" 0. 1. ;
"Uniform" 0. 1. ;
"Normal" 1.5 0.75 ;
"Normal" 0.12 0.15 ;
"Beta" 0.7 0.05;
"IGamma" 0.5 4.
"IGamma" 0.5 4.
"IGamma" 0.5 4.
"IGamma" 0.5 4.
]
# %%
param_labels = [
"\$\\rho_{A}\$",
"\$\\rho_g\$",
"\$\\rho_R\$",
"\$\\rho_\\pi\$",
"\$\\rho_y\$",
"\$\\psi\$",
"\$\\sigma_A\$",
"\$\\sigma_\\mu\$",
"\$\\sigma_g\$",
"\$\\sigma_m\$" ]
# %%
JLD2.@load "output/classic.jld2" SMC_classic
# %%
table_estimates(prior, SMC_classic.μ, SMC_classic.σ, SMC_classic.draws, SMC_classic.weights[:,end], param_labels)
# %%
#Plotting IRFs
irf_classic(SMC_classic.μ, 5, SMC_classic.EstM)
# %%
savefig("./figures/irf_A_classic.pdf")
# %%
irf_classic(SMC_classic.μ, 7, SMC_classic.EstM)
# %%
savefig("./figures/irf_g_classic.pdf")
# %%
irf_classic(SMC_classic.μ, 8, SMC_classic.EstM)
# %%
savefig("./figures/irf_m_classic.pdf")
# %%
JLD2.@load "output/dual.jld2" SMC_dual
# %%
μ = SMC_dual.μ
# %%
JLD2.@save "output/dual_mean_est.jld2" μ
# %%
table_estimates(prior, SMC_dual.μ, SMC_dual.σ, SMC_dual.draws, SMC_dual.weights[:,end], param_labels)
# %%
#Plotting IRFs
irf(SMC_dual.μ, 6, SMC_dual.EstM)
# %%
savefig("./figures/irf_A.pdf")
# %%
irf(SMC_dual.μ, 8, SMC_dual.EstM)
# %%
savefig("./figures/irf_g.pdf")
# %%
irf(SMC_dual.μ, 9, SMC_dual.EstM)
# %%
savefig("./figures/irf_m.pdf")
# %%
#Plotting Posterior and Prior distributions
param_latex = [
L"$\rho_{A}$",
L"$\rho_g$",
L"$\rho_R$",
L"$\rho_\pi$",
L"$\rho_y$",
L"$\psi$",
L"$\sigma_A$",
L"$\sigma_\mu$",
L"$\sigma_g$",
L"$\sigma_m$" ]
prior_post_plot(SMC_dual, param_latex)
# %%
savefig("./figures/prior_post.pdf")
# %%
#Displaying the estimation results with different filters for the input data.
JLD2.@load "output/filters.jld2" SMC_h SMC_lt SMC_fd SMC_hp
# %%
param_labels = [
"\$\\rho_{A}\$",
"\$\\rho_g\$",
"\$\\rho_R\$",
"\$\\rho_\\pi\$",
"\$\\rho_y\$",
"\$\\psi\$",
"\$\\sigma_A\$",
"\$\\sigma_\\mu\$",
"\$\\sigma_g\$",
"\$\\sigma_m\$" ]
table_filter_robust(SMC_h.μ, SMC_lt.μ, SMC_fd.μ, SMC_hp.μ, SMC_h.σ, SMC_lt.σ, SMC_fd.σ, SMC_hp.σ, param_labels)
# %%
# Plots the series of labor market moments with different filters.
path = "input/lm_moments.csv"
demean = [:pi,:R]
detrend = [:y,:n,:jcp,:jcf,:mufsmu,:nf,:jdp,:jdf,:v]
# %%
data_lt,b_lt,e_lt = linear_df(path,detrend,demean)
data_fd,b_fd,e_fd = firstdifference_df(path,detrend,demean)
data_hp,b_hp,e_hp = hodrickprescott_df(path,detrend,demean)
data_h,b_h,e_h = hamilton_detrend_df(path,detrend,demean)
# %%
plot_lm_series(data_lt,b_lt,e_lt)
plot_lm_series(data_fd,b_fd,e_fd)
plot_lm_series(data_hp,b_hp,e_hp)
plot_lm_series(data_h,b_h,e_h)
savefig("./figures/lm_moments.pdf")
# %%
# Plots the data used for the estimation.
path = "input/lm_moments_short.csv"
demean = [:pi,:R]
detrend = [:y, :n, :nf, :jcp, :jcf, :jc, :mufsmu, :jdp, :jdf, :jd, :v]
# %%
data_lt,b_lt,e_lt = linear_df(path,detrend,demean)
data_fd,b_fd,e_fd = firstdifference_df(path,detrend,demean)
data_hp,b_hp,e_hp = hodrickprescott_df(path,detrend,demean)
data_h,b_h,e_h = hamilton_detrend_df(path,detrend,demean)
# %%
plot_estimation_data(data_lt,b_lt,e_lt)
plot_estimation_data(data_fd,b_fd,e_fd)
plot_estimation_data(data_hp,b_hp,e_hp)
plot_estimation_data(data_h,b_h,e_h)
# %%
# Computes the likelihood of data for both the classic and dual-labor-market model.
JLD2.@load "output/logY.jld2" logY_run
# %%
mean(logY_run[2:end,:], dims=1)
std(logY_run[2:end,:], dims=1)
# %%
# Plots the data used for the estimation and plots cross covariances and correlations.
path = "input/lm_moments_short.csv"
demean = [:pi,:R]
detrend = [:y,:n,:nf,:jcp,:jcf,:jc,:mufsmu,:jdp,:jdf,:jd,:v]
data,b,e = linear_df(path,detrend,demean)
std_obs, cov_obs, cor_obs = cov_cor_obs(data,b,e)
# %%
JLD2.@load "output/cov_cor.jld2" std_sim_mean std_sim_q cor_sim_mean cor_sim_q cov_sim_mean cov_sim_q
# %%
lm_latex = [
"\$Y\$",
"\$\\pi\$",
"\$R\$",
"\$n\$",
"\$n^f\$",
"\$jc^p\$",
"\$jc^f\$",
"\$jc\$",
"\$\\mu^f / \\left( \\mu^f + \\mu^p \\right)\$",
"\$jd^p\$",
"\$jd^f\$",
"\$jd\$",
"\$v\$"
]
# %%
table_lm_moments(std_obs,std_sim_mean,cor_obs,cor_sim_mean,lm_latex)
table_lm_moments_withq(std_obs, std_sim_mean, std_sim_q, cor_obs, cor_sim_mean, cor_sim_q, lm_latex)
# %%
lm_names = [
"Y",
"\\pi",
"R",
"n",
"n^f",
"jc^p",
"jc^f",
"jc",
"\\left( jc^f / jc \\right)",
"jd^p",
"jd^f",
"jd",
"v"
]
# %%
mat_graphe = plot_cor(cor_obs, cor_sim_mean, cor_sim_q, lm_names)
n_data = size(cor_sim_mean,1)
plot(mat_graphe..., layout=(n_data,n_data))
plot!(size=(1680,1050))
savefig("./figures/autocor.pdf")
# %%
mat_graphe = plot_cor(cov_obs, cov_sim_mean, cov_sim_q, lm_names)
n_data = size(cov_sim_mean,1)
plot(mat_graphe..., layout=(n_data,n_data))
plot!(size=(1680,1050))
savefig("./figures/autocov.pdf")
# %%
# %%
# Classic: plots the data used for the estimation and plots cross covariances and correlations.
path = "input/lm_moments_short.csv"
vars = [:q,:y,:pi,:R,:n,:jcp,:jdp,:v]
demean = [:pi,:R]
detrend = [:y,:n,:jcp,:jdp,:v]
data,b,e = linear_df(path,detrend,demean)
std_obs, cov_obs, cor_obs = cov_cor_obs(data[:,vars],b,e)
# %%
JLD2.@load "output/cov_cor_classic.jld2" std_sim_mean std_sim_q cor_sim_mean cor_sim_q cov_sim_mean cov_sim_q
# %%
lm_latex = [
"\$Y\$",
"\$\\pi\$",
"\$R\$",
"\$n\$",
"\$jc^p\$",
"\$jd^p\$",
"\$v\$"
]
# %%
table_lm_moments(std_obs,std_sim_mean,cor_obs,cor_sim_mean,lm_latex)
table_lm_moments_withq(std_obs, std_sim_mean, std_sim_q, cor_obs, cor_sim_mean, cor_sim_q, lm_latex)
# %%
lm_names = [
"Y",
"\\pi",
"R",
"n",
"jc^p",
"jd^p",
"v"
]
# %%
mat_graphe = plot_cor(cor_obs, cor_sim_mean, cor_sim_q, lm_names)
n_data = size(cor_sim_mean,1)
plot(mat_graphe..., layout=(n_data,n_data))
plot!(size=(1680,1050))
savefig("./figures/autocor_classic.pdf")
# %%
mat_graphe = plot_cor(cov_obs, cov_sim_mean, cov_sim_q, lm_names)
n_data = size(cov_sim_mean,1)
plot(mat_graphe..., layout=(n_data,n_data))
plot!(size=(1680,1050))
savefig("./figures/autocov_classic.pdf")
# %%
# Steady-state impact of firing costs on volatility.
path = "input/lm_moments_short.csv"
demean = [:pi,:R]
detrend = [:y,:n,:jcp,:jcf,:mufsmu,:nf,:jdp,:jdf,:v]
data,b,e = linear_df(path,detrend,demean)
std_obs, cov_obs, cor_obs = cov_cor_obs(data,b,e)
# %%
lm_latex = [
"\$Y\$",
"\$\\pi\$",
"\$R\$",
"\$n\$",
"\$n^f\$",
"\$jc^p\$",
"\$jc^f\$",
"\$jc\$",
"\$\\mu^f / \\left( \\mu^f + \\mu^p \\right)\$",
"\$jd^p\$",
"\$jd^f\$",
"\$jd\$",
"\$v\$"
]
# %%
JLD2.@load "output/cov_cor.jld2" std_sim_mean std_sim_q cor_sim_mean cor_sim_q cov_sim_mean cov_sim_q
std_sim_mean0 = deepcopy(std_sim_mean)
std_sim_q0 = deepcopy(std_sim_q)
cor_sim_mean0 = deepcopy(cor_sim_mean)
cor_sim_q0 = deepcopy(cor_sim_q)
cov_sim_mean0 = deepcopy(cov_sim_mean)
cov_sim_q0 = deepcopy(cov_sim_q)
JLD2.@load "output/cov_cor_rhoF.jld2" std_sim_mean std_sim_q cor_sim_mean cor_sim_q cov_sim_mean cov_sim_q
# %%
table_lm_moments_comp(std_obs, cor_obs, std_sim_mean0, cor_sim_mean0, std_sim_mean, cor_sim_mean, lm_latex)
# %%
# Plots the counterfactual historic series of observables with each shock hitting separately
JLD2.@load "output/dual.jld2" SMC_dual
# %%
var_names = ["Y", "\\pi", "R", "n"]
shock_names = ["A", "\\mu", "g", "m"]
# %%
mat_graphe = counterfactual(SMC_dual, var_names, shock_names)
# %%
n = size(var_names,1)
plot(mat_graphe..., layout=(n,n))
plot!(size=(1680,1050))
savefig("./figures/counterfactuals.pdf")
