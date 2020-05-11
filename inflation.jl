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
	using Dual_pi
	using SMC_aux
end
# %%
using SMC_main
using FilterData
using Distributions
# %%
using JLD2, FileIO
# %%
JLD2.@load "output/mu_dual.jld2" μ
# %%
include("./functions/InflationFunctions.jl")
# %%
M = lin_dual()
include("./functions/EstDual.jl")
EstM = init_EstLinMod(prior(), Y, bound, M)
EstM.param .= μ
solve!(EstM)
state_space!(EstM)
# %%
sim_x, sim_y = draw(EstM, N=5000)
# %%
μᵖzs, μzᶠ, cov_sim, sim = π_var(EstM, sim_x, sim_y)
# %%
cal = calibrate()
cal[:F] *= 0.95
M = lin_dual(cal)
include("./functions/EstDual.jl")
EstM = init_EstLinMod(prior(), Y, bound, M)
EstM.param .= μ
solve!(EstM)
state_space!(EstM)
# %%
sim_x_0, sim_y_0 = draw(EstM, N=5000)
# %%
μᵖzs_0, μzᶠ_0, cov_sim_0, sim_0 = π_var(EstM, sim_x_0, sim_y_0)
# %%
cov_sim .= 16*cov_sim
cov_sim_0 .= 16*cov_sim_0
# %%
# println("\\begin{table}[H]
# \\begin{center}
# \\begin{tabular}{ccc}
# \\toprule
# & Baseline & Reduced firing costs\\\\ \\midrule
# Covariances & & \\\\
# \$var \\left( \\pi \\right)\$ & $(round(sum(cov_sim), digits=2)) & $(round(sum(cov_sim_0), digits=2)) \\\\
# \$var \\left( hc \\right)\$ & $(round(cov_sim[1,1], digits=2)) & $(round(cov_sim_0[1,1], digits=2)) \\\\
# \$2 cov \\left( hc , \\Pi^f \\right)\$ & $(round(2*cov_sim[1,3], digits=2)) & $(round(2*cov_sim_0[1,3], digits=2)) \\\\
# \$var \\left( \\Pi^f \\right)\$ & $(round(cov_sim[3,3], digits=2)) & $(round(cov_sim_0[3,3], digits=2)) \\\\
# \$var \\left( subs^p \\right)\$ & \$1.6 \\cdot 10^{-4}\$ & \$4.3 \\cdot 10^{-4}\$ \\\\
# \\bottomrule
# \\end{tabular}
# \\end{center}
# \\end{table}")
# %%
