import Pkg; Pkg.activate(".")
path = pwd() * "/functions/"
push!(LOAD_PATH, path)
# %%
using Revise
# %%
using SparseSym
using LinearAlgebra
using SparseArrays, SymEngine, Espresso
using Kamenik
using SymDual
using EconModel
using JLD2, FileIO
# %%
M = sym_dual()
# %%
sym_f_v,sym_f_vv,sym_f_vvv = chainrule(M)
# %%
Eϵ = Dict()
n_ϵ = size(M.η, 2)
Eϵ[2] = I(n_ϵ)[:]
Eϵ[3] = spzeros(n_ϵ^3)
# %%
@load "output/mu_dual.jld2" μ
# %%
params = zeros(18)
params[1:10] .= μ
params[11:end] .= [0.7,0.7,0.7,0.7,1.,1.,1.,1.]
# params[8:9,:] .= 0.
# %%
order1 = Order1(M, sym_f_v, params)
# %%
solve_order1!(order1)
# %%
order2 = Order2(order1, sym_f_vv)
# %%
solve_order2!(order2, Eϵ)
# %%
order3 = Order3(order2, sym_f_vvv)
# %%
solve_order3!(order3, Eϵ)
# %%
using StatsBase
using Plots
using LaTeXStrings
using Macros
using SpecialFunctions
using FilterData
using VAR
pyplot()
include("functions/IRF.jl")
# %%
m = sto_ss_pruning(order3)
# %%
x,y = irf(order3, 5, m)
plot_irf(x, y, M.calibrated, M.ss_x, M.ss_y)
savefig("./figures/irf_vol_A.pdf")
# %%
x,y = irf(order3, 6, m)
plot_irf(x, y, M.calibrated, M.ss_x, M.ss_y)
savefig("./figures/irf_vol_mu.pdf")
# %%
x,y = irf(order3, 7, m)
plot_irf(x, y, M.calibrated, M.ss_x, M.ss_y)
savefig("./figures/irf_vol_g.pdf")
# %%
x,y = irf(order3, 8, m)
plot_irf(x, y, M.calibrated, M.ss_x, M.ss_y)
savefig("./figures/irf_vol_m.pdf")
# %%
path = "input/var.csv"
# vars = [:CAC40,:VOL,:N]
vars = [:CAC40,:VOL,:MUFSMU,:N]
detrend = vars
demean = []
t1 = "1960Q1"
t2 = "2020Q1"
Y = hamilton_detrend(path,vars,detrend,demean,t1,t2)
IRF = sim_irf_var(Y, 1)
# %%
p1 = plot(IRF[3,2,:,1],ribbon=(IRF[3,2,:,2], IRF[3,2,:,3]), linestyle = :solid, linewidth=2, color = "black", label = "")
plot!([0.], seriestype = :hline, linestyle = :dash, linewidth=2, color = "black", label = "")
ylabel!(L"jc^f / jc")
xlabel!("Quarters")
xgrid!(:off)
ygrid!(:off)
# %%
p2 = plot(IRF[4,2,:,1],ribbon=(IRF[4,2,:,2], IRF[4,2,:,3]), linestyle = :solid, linewidth=2, color = "black", label = "95% confidence interval", legend = :bottomright)
plot!([0.], seriestype = :hline, linestyle = :dash, linewidth=2, color = "black", label = "")
ylabel!(L"n")
xlabel!("Quarters")
xgrid!(:off)
ygrid!(:off)
# %%
plot(p1,p2, layout = (1,2))
# %%
savefig("./figures/var.pdf")
# %%
