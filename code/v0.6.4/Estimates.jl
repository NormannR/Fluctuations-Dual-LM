working_dir = "C:\\Users\\Normann\\Dropbox\\Ch2\\code\\23_02_2019_Euro"
cd(working_dir)
using StatsBase
using JLD

Nparam = 11
NY = 15
NX = 4

var = JLD.load("LT_SHORT.jld")
theta = var["theta"]
wght = repmat(var["weights"], 1, Nparam)
mu  = reshape(sum(theta.*wght,1)',Nparam)

sig = sum((theta - repmat(mu, 1, size(theta,1))').^2 .*wght,1)
sig = (sqrt.(sig))

q = zeros(Nparam,2)
for p in 1:Nparam
        q[p,1] = quantile(theta[:,p],Weights(var["weights"]),0.05)
        q[p,2] = quantile(theta[:,p],Weights(var["weights"]),0.95)
end

Estimates = Array{Any,2}(Nparam, 8)

Estimates[:,1:4] = [    "\$\\rho_A\$"	"Uniform"	0	1 ;
                        "\$\\rho_\\mu\$"	"Uniform"	0	1 ;
                        "\$\\rho_g\$"	"Uniform"	0	1 ;
                        "\$\\rho_R\$"	"Uniform"	0	1
                        "\$\\rho_\\pi\$"	"Normal"	1.5	0.75 ;
                        "\$\\rho_y\$"	"Normal"	0.5	0.75 ;
                        "\$\\psi\$"	"Uniform"	0	1 ;
                        "\$\\sigma_A\$"	"IGamma"	0.1	4 ;
                        "\$\\sigma_\\mu\$"	"IGamma"	0.1	4 ;
                        "\$\\sigma_g\$"	"IGamma"	0.1	4 ;
                        "\$\\sigma_m\$"	"IGamma"	0.1	4 ]

Estimates[:,5] = mu
Estimates[:,6] = sig
Estimates[:,7:8] = q

writedlm("EstimationResults.txt", Estimates)

#PRIOR POSTERIOR HISTOGRAMS
