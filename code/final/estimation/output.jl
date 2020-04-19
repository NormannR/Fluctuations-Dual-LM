import Pkg; Pkg.activate("../../../../")
push!(LOAD_PATH,pwd())
# %%
using Revise
# %%
using SharedArrays
using Distributed
using Dual
using FilterData
using Distributions
using SMC_aux
using SMC_main
using JLD2, FileIO
# %%
JLD2.@load "dual.jld2" SMC_dual
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
table_estimates(prior, SMC_dual.μ, SMC_dual.σ, SMC_dual.draws, SMC_dual.weights[:,end], param_labels)
# %%
#Plotting IRFs
irf(SMC_dual.μ, 6, SMC_dual.EstM)
# %%
savefig("../../../../latex/figures_03_2020/irf_A.pdf")
# %%
irf(SMC_dual.μ, 8, SMC_dual.EstM)
# %%
savefig("../../../../latex/figures_03_2020/irf_g.pdf")
# %%
irf(SMC_dual.μ, 9, SMC_dual.EstM)
# %%
savefig("../../../../latex/figures_03_2020/irf_m.pdf")
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
savefig("../../../../latex/figures_03_2020/prior_post.pdf")
# %%
#Displaying the estimation results with different filters for the input data.
@load "filters.jld2" SMC_h SMC_lt SMC_fd SMC_hp
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
path = "../input/lm_moments.csv"
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
savefig("./latex/figures_new/lm_moments.pdf")
# %%
# Plots the data used for the estimation.
path = "../input/lm_moments_short.csv"
demean = [:pi,:R]
detrend = [:y, :n, :jcp, :jcf, :mufsmu, :nf, :jdp, :jdf, :v]
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
@load "logY.jld2" logY_run
# %%
mean(logY_run[2:end,:], dims=1)
std(logY_run[2:end,:], dims=1)
# %%
# Plots the data used for the estimation and plots cross covariances and correlations.
path = "../input/lm_moments_short.csv"
demean = [:pi,:R]
detrend = [:y,:n,:jcp,:jcf,:mufsmu,:nf,:jdp,:jdf,:v]
data,b,e = linear_df(path,detrend,demean)
std_obs, cov_obs, cor_obs = cov_cor_obs(data,b,e)
# %%
JLD2.@load "cov_cor.jld2" std_sim_mean std_sim_q cor_sim_mean cor_sim_q cov_sim_mean cov_sim_q
# %%
lm_latex = [
"\$Y\$",
"\$\\pi\$",
"\$R\$",
"\$n\$",
"\$jc^p\$",
"\$jc^f\$",
"\$\\mu^f / \\left( \\mu^f + \\mu^p \\right)\$",
"\$n^f\$",
"\$jd^p\$",
"\$jd^f\$",
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
"jc^f",
"\\left( jc^f / jc \\right)",
"n^f",
"jd^p",
"jd^f",
"v"
]
# %%
mat_graphe = plot_cor(cor_obs, cor_sim_mean, cor_sim_q, lm_names)
n_data = size(cor_sim_mean,1)
plot(mat_graphe..., layout=(n_data,n_data))
plot!(size=(1680,1050))
savefig("../../../../latex/figures_03_2020/autocor.pdf")
# %%
mat_graphe = plot_cor(cov_obs, cov_sim_mean, cov_sim_q, lm_names)
n_data = size(cov_sim_mean,1)
plot(mat_graphe..., layout=(n_data,n_data))
plot!(size=(1680,1050))
savefig("../../../../latex/figures_03_2020/autocov.pdf")
# %%
# Steady-state impact of firing costs on volatility.
JLD2.@load "cov_cor.jld2" std_sim_mean std_sim_q cor_sim_mean cor_sim_q cov_sim_mean cov_sim_q
std_sim_mean0 = deepcopy(std_sim_mean)
std_sim_q0 = deepcopy(std_sim_q)
cor_sim_mean0 = deepcopy(cor_sim_mean)
cor_sim_q0 = deepcopy(cor_sim_q)
cov_sim_mean0 = deepcopy(cov_sim_mean)
cov_sim_q0 = deepcopy(cov_sim_q)
# %%
JLD2.@load "cov_cor_rhoF.jld2" std_sim_mean std_sim_q cor_sim_mean cor_sim_q cov_sim_mean cov_sim_q
# %%
std_sim_mean0
std_sim_q0
std_sim_mean
std_sim_q
