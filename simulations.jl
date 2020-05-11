using SharedArrays
using Distributed
addprocs()
# %%
@everywhere begin
	import Pkg; Pkg.activate(".")
	path = pwd() * "/functions"
	push!(LOAD_PATH,path)
end
# %%
using Revise
# %%
@everywhere begin
	using Classic
	using Dual
	using SMC_aux
end
# %%
using SMC_main
using FilterData
using Distributions
# %%
using JLD2, FileIO
# %%
include("./functions/OutputFunctions.jl")
# %%
# Simulating the model to compare labor market moments and the data.
JLD2.@load "output/dual.jld2" SMC_dual
μ = SMC_dual.μ
JLD2.@save "output/mu_dual.jld2" μ
# %%
#JLD2 works badly with SharedArrays ! I need to make a clean copy.
draws = SharedArray{Float64}(size(SMC_dual.draws))
draws .= Array(SMC_dual.draws)
sim_x, sim_y = simul_draws(SMC_dual.EstM, draws, SMC_dual.n_part ; T=200)
# %%
JLD2.@save "output/sim_states.jld2" sim_x sim_y
# %%
JLD2.@load "output/sim_states.jld2" sim_x sim_y
# %%
std_sim_mean, std_sim_q, cor_sim_mean, cor_sim_q, cov_sim_mean, cov_sim_q = cov_cor_draws(SMC_dual.EstM, sim_x, sim_y, Array(SMC_dual.weights[:,end]))
JLD2.@save "output/cov_cor.jld2" std_sim_mean std_sim_q cor_sim_mean cor_sim_q cov_sim_mean cov_sim_q
# %%
# Simulating the model with different labor market institutions.
cal = calibrate()
cal[:F] *= 0.95
M = lin_dual(cal)
# %%
JLD2.@load "output/dual.jld2" SMC_dual
SMC_dual.EstM.model = M
draws = SharedArray{Float64}(size(SMC_dual.draws))
draws .= Array(SMC_dual.draws)
sim_x, sim_y = simul_draws(SMC_dual.EstM, draws, SMC_dual.n_part ; T=200)
JLD2.@save "output/sim_rhoF.jld2" sim_x sim_y
# %%
JLD2.@load "output/sim_rhoF.jld2" sim_x sim_y
std_sim_mean, std_sim_q, cor_sim_mean, cor_sim_q, cov_sim_mean, cov_sim_q = cov_cor_draws(SMC_dual.EstM, sim_x, sim_y, Array(SMC_dual.weights[:,end]))
JLD2.@save "output/cov_cor_rhoF.jld2" std_sim_mean std_sim_q cor_sim_mean cor_sim_q cov_sim_mean cov_sim_q
# %%
# Simulating the model to compare labor market moments and the data.
JLD2.@load "output/classic.jld2" SMC_classic
#JLD2 works badly with SharedArrays ! I need to make a clean copy.
draws = SharedArray{Float64}(size(SMC_classic.draws))
draws .= Array(SMC_classic.draws)
sim_x, sim_y = simul_draws(SMC_classic.EstM, draws, SMC_classic.n_part ; T=200)
# %%
JLD2.@save "output/sim_states_classic.jld2" sim_x sim_y
# %%
JLD2.@load "output/sim_states_classic.jld2" sim_x sim_y
std_sim_mean, std_sim_q, cor_sim_mean, cor_sim_q, cov_sim_mean, cov_sim_q = cov_cor_classic_draws(SMC_classic.EstM, sim_x, sim_y, Array(SMC_classic.weights[:,end]))
JLD2.@save "output/cov_cor_classic.jld2" std_sim_mean std_sim_q cor_sim_mean cor_sim_q cov_sim_mean cov_sim_q
