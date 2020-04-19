using SharedArrays
using Distributed
addprocs()
# %%
@everywhere begin
	import Pkg; Pkg.activate("../../../../")
	push!(LOAD_PATH,pwd())
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
JLD2.@load "dual.jld2" SMC_dual
#JLD2 works badly with SharedArrays ! I need to make a clean copy.
draws = SharedArray{Float64}(size(SMC_dual.draws))
draws .= Array(SMC_dual.draws)
sim_x, sim_y = simul_draws(SMC_dual.EstM, draws, SMC_dual.n_part ; T=200)
# %%
JLD2.@save "sim_states.jld2" sim_x sim_y
# %%
JLD2.@load "sim_states.jld2" sim_x sim_y
std_sim_mean, std_sim_q, cor_sim_mean, cor_sim_q, cov_sim_mean, cov_sim_q = cov_cor_draws(SMC_dual.EstM, sim_x, sim_y, Array(SMC_dual.weights[:,end]))
JLD2.@save "cov_cor.jld2" std_sim_mean std_sim_q cor_sim_mean cor_sim_q cov_sim_mean cov_sim_q
# %%
# Simulating the model with different labor market institutions.
cal = calibrate()
cal[:ρᶠ] -= 0.05
M = lin_dual([0.66], cal)
# %%
JLD2.@load "dual.jld2" SMC_dual
SMC_dual.EstM.model = M
draws = SharedArray{Float64}(size(SMC_dual.draws))
draws .= Array(SMC_dual.draws)
sim_x, sim_y = simul_draws(SMC_dual.EstM, draws, SMC_dual.n_part ; T=200)
JLD2.@save "sim_rhoF.jld2" sim_x sim_y
# %%
JLD2.@load "sim_rhoF.jld2" sim_x sim_y
std_sim_mean, std_sim_q, cor_sim_mean, cor_sim_q, cov_sim_mean, cov_sim_q = cov_cor_draws(SMC_dual.EstM, sim_x, sim_y, Array(SMC_dual.weights[:,end]))
JLD2.@save "cov_cor_rhoF.jld2" std_sim_mean std_sim_q cor_sim_mean cor_sim_q cov_sim_mean cov_sim_q
