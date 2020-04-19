using Distributed, SharedArrays
addprocs()
# %%
@everywhere begin
    import Pkg; Pkg.activate("../../../")
    path = pwd() * "/functions/"
    push!(LOAD_PATH, path)
end
# %%
using Revise
# %%
using SparseSym
using Revise
using LinearAlgebra
using SparseArrays, SymEngine, Espresso
@everywhere using Kamenik
using SymDual
using EconModel
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
params = [
0.71,
0.88,
0.77,
2.2,
0.29,
0.82,
0.31,
5.31,
0.08,
0.7,
0.7,
0.7,
1.,
1.,
1. ]
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
include("functions/IRF.jl")
# %%
m = sto_ss_pruning(order3)
# %%
x,y = irf(order3, 4, m)
# %%
plot(x[1,:])
# %%
x_f, x_s, x_rd, y_rd = irf(order3, 4)
# %%
mean((x_f + x_s + x_rd)[:,:,100,1], dims = 2)
# %%
using StatsBase
using Plots
pyplot()
# %%
mean(x, dims=2)
# %%
plot(mean(y, dims=2)[11,1,:])
