using SharedArrays
using Distributed
addprocs()
# %%
# Needs to be changed to point towards folder which contains the Project.toml and Manifest.toml files.
@everywhere begin
	import Pkg; Pkg.activate(".")
	path = pwd() * "/functions"
	push!(LOAD_PATH,path)
end
# %%
using Revise
# %%
@everywhere begin
	using Dual
	using SMC_aux
	using Classic
end
# %%
using SMC_main
using FilterData
using Distributions
# %%
using JLD2, FileIO
# %%
M = lin_dual()
include("./functions/EstDual.jl")
EstM = init_EstLinMod(prior(), Y, bound, M)
SMC_dual = init_EstSMC(EstM)
# SMC_dual = init_EstSMC(EstM, n_ϕ = 500, n_part = 30000)
# %%
@everywhere @eval SMC_dual = $SMC_dual
# %%
estimate!(SMC_dual)
# %%
JLD2.@save "output/dual.jld2" SMC_dual
# %%
M = lin_classic()
include("./functions/EstDual.jl")
EstM = init_EstLinMod(prior(), Y, bound, M)
# SMC_classic = init_EstSMC(EstM, n_ϕ = 500, n_part = 30000)
SMC_classic = init_EstSMC(EstM)
# %%
@everywhere @eval SMC_classic = $SMC_classic
# %%
estimate!(SMC_classic)
# %%
@save "output/classic.jld2" SMC_classic
# %%
M = lin_dual()
include("./functions/EstDual.jl")
# %%
#Hamilton
Y = hamilton_detrend(path, vars, detrend, demean, t1, t2)
EstM = init_EstLinMod(prior(), Y, bound, M)
SMC_h = init_EstSMC(EstM)
estimate!(SMC_h)
# %%
#Linear trend
Y = linear_detrend(path, vars, detrend, demean, t1, t2)
EstM = init_EstLinMod(prior(), Y, bound, M)
SMC_lt = init_EstSMC(EstM)
estimate!(SMC_lt)
# %%
#First difference
Y = first_difference(path,vars,detrend,t1,t2)
EstM = init_EstLinMod(prior(), Y, bound, M)
SMC_fd = init_EstSMC(EstM)
estimate!(SMC_fd)
# %%
#Hodrick-Prescott
Y = hp_detrend(path, vars, detrend, demean, t1, t2)
EstM = init_EstLinMod(prior(), Y, bound, M)
SMC_hp = init_EstSMC(EstM)
estimate!(SMC_hp)
# %%
@save "output/filters.jld2" SMC_h SMC_lt SMC_fd SMC_hp
# %%
M = lin_dual()
include("./functions/EstDual.jl")
EstM = init_EstLinMod(prior(), Y, bound, M)
SMC_dual = init_EstSMC(EstM)
M = lin_classic()
include("./functions/EstDual.jl")
EstM = init_EstLinMod(prior(), Y, bound, M)
SMC_classic = init_EstSMC(EstM)
# %%
n_param = length(EstM.param)
n_run = 50
logY_run = zeros(n_run, 2)
for n in 1:n_run
    display("$n / $n_run")
    estimate!(SMC_dual; display=SMC_dual.n_ϕ)
	SMC_dual.EstM.param = SMC_dual.μ
	logprior, _ = SMC_aux.logprior(EstM)
	logY_run[n,1] = sum(log.(SMC_dual.const_vec)) + logprior
    estimate!(SMC_classic; display=SMC_classic.n_ϕ)
	SMC_classic.EstM.param = SMC_classic.μ
	logprior, _ = SMC_aux.logprior(EstM)
	logY_run[n,2] = sum(log.(SMC_classic.const_vec)) + logprior
end
# %%
@save "output/logY.jld2" logY_run
# %%
