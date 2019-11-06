graph_dir = "E:\\Dropbox\\Ch2\\latex\\figures"
working_dir = "E:\\Dropbox\\Ch2\\code\\23_02_2019_Euro"
# graph_dir = "C:\\Users\\Normann\\Dropbox\\Ch2\\latex\\figures"
# working_dir = "C:\\Users\\Normann\\Dropbox\\Ch2\\code\\23_02_2019_Euro"
cd(working_dir)
using FileIO, ExcelFiles, DataFrames, ExcelReaders
using DataFrames
using Distributions.MvNormal
using Roots
using ForwardDiff
using JLD
using Optim
using Calculus
using PyPlot
using Distributions
using QuantEcon.solve_discrete_lyapunov
using NLsolve.nlsolve
using StatsFuns
using Gensys.gensysdt
using StatsBase

include("SMCfunctions.jl")
# include("IskrevFunctions.jl")
include("input\\InitialGuess.jl")
ss = steady_state_eval(param)
linearized_model(param) = linearized_model_ss(param,ss)
funcmod = linearized_model

# data = DataFrame(load("input\\LMmoments.xlsx", "04032019!A1:J95"))
# T = size(data,1)
# m = 4
# h = 8
# p = 4
#
# Filtered,Ind = HamiltonFilter(data[:,[:Y,:n,:jcp,:jcf,:mufsmu,:nfsn,:endo_jdp,:exo_jdf,:v]],h,p)


data = DataFrame(load("input\\LMmoments.xlsx", "LT!L1:T51"))
Y = data[:,[:Y,:n,:jcp,:jcf,:mufsmu,:nfsn,:endo_jdp,:exo_jdf,:v]]
T = size(Y,1)
k = size(Y,2)

Ind = Array{Int,2}(k,2)
for i in 1:k

     Begin = findfirst(x->(x > -999.),Y[:,i])
     Ind[i,:] = [Begin,T]
 end

stddev = ObsStd(Y,Ind)
correlation = ObsCor(Y,Ind)

#==============================================================================#
#================================Stylized Facts================================#
#==============================================================================#

# Y = 1
# n = 2
# jcp = 3
# jcf = 4
# mufsmu = 5
# nfsn = 6
# jdp = 7
# jdf = 8
# v = 9
#
# figure()
#
# subplot(2,2,1)
# plot(Ind[Y,1]:Ind[Y,2],Filtered[Ind[Y,1]:Ind[Y,2],Y],"k-")
# plot(Ind[n,1]:Ind[n,2],Filtered[Ind[n,1]:Ind[n,2],n],"k--")
# legend(["\$Y\$","\$n\$","\$v\$"],loc="lower right")
# locs=Ind[Y,1]:8:Ind[Y,2]
# labels=[]
# for l in Int64.(locs)
#     if l >= Ind[Y,1] && l <= Ind[Y,2]
#         labels=vcat(labels,[data[p+h+l,:quarter]])
#     else
#         labels=vcat(labels,[""])
#     end
# end
# xticks(locs, labels, rotation=45)
#
#
# subplot(2,2,2)
# plot(Ind[jcp,1]:Ind[jcp,2],Filtered[Ind[jcp,1]:Ind[jcp,2],Y],"k:")
# plot(Ind[jcp,1]:Ind[jcp,2],Filtered[Ind[jcp,1]:Ind[jcp,2],jcp],"k-")
# plot(Ind[jcf,1]:Ind[jcf,2],Filtered[Ind[jcf,1]:Ind[jcf,2],jcf],"k--")
# legend(["\$Y\$","\$jc^p\$","\$jc^f\$"],loc="center left", bbox_to_anchor=(1, 0.5))
# locs=Ind[jcp,1]:8:Ind[jcp,2]
# labels=[]
# for l in Int64.(locs)
#     if l >= Ind[jcp,1] && l <= Ind[jcp,2]
#         labels=vcat(labels,[data[p+h+l,:quarter]])
#     else
#         labels=vcat(labels,[""])
#     end
# end
# xticks(locs, labels, rotation=45)
#
#
# subplot(2,2,3)
# plot(Ind[jdp,1]:Ind[jdp,2],Filtered[Ind[jdp,1]:Ind[jdp,2],Y],"k:")
# plot(Ind[jdp,1]:Ind[jdp,2],Filtered[Ind[jdp,1]:Ind[jdp,2],jdp],"k-")
# plot(Ind[jdf,1]:Ind[jdf,2],Filtered[Ind[jdf,1]:Ind[jdf,2],jdf],"k--")
# legend(["\$Y\$","\$jd^p\$","\$jd^f\$"],loc="upper left")
# locs=Ind[jdp,1]:8:Ind[jdp,2]
# labels=[]
# for l in Int64.(locs)
#     if l >= Ind[jdp,1] && l <= Ind[jdp,2]
#         labels=vcat(labels,[data[p+h+l,:quarter]])
#     else
#         labels=vcat(labels,[""])
#     end
# end
# xticks(locs, labels, rotation=45)
#
# subplot(2,2,4)
# plot(Ind[mufsmu,1]:Ind[mufsmu,2],Filtered[Ind[mufsmu,1]:Ind[mufsmu,2],Y],"k:")
# plot(Ind[nfsn,1]:Ind[nfsn,2],Filtered[Ind[nfsn,1]:Ind[nfsn,2],nfsn],"k-")
# plot(Ind[mufsmu,1]:Ind[mufsmu,2],Filtered[Ind[mufsmu,1]:Ind[mufsmu,2],mufsmu],"k--")
# legend(["\$Y\$","\$n^f / n\$","\$\\mu^f / \\mu\$"],loc="center left", bbox_to_anchor=(1, 0.5))
# locs=Ind[mufsmu,1]:8:Ind[mufsmu,2]
# labels=[]
# for l in Int64.(locs)
#     if l >= Ind[mufsmu,1] && l <= Ind[mufsmu,2]
#         labels=vcat(labels,[data[p+h+l,:quarter]])
#     else
#         labels=vcat(labels,[""])
#     end
# end
# xticks(locs, labels, rotation=45)
# tight_layout()
# cd(graph_dir)
# savefig("StylizedFacts.pdf", bbox_inches="tight")

#==============================================================================#
#================================Simulation====================================#
#==============================================================================#

cd(working_dir)

var = JLD.load("LT_SHORT.jld")

theta = var["theta"]
weights = var["weights"]
wght = repmat(var["weights"], 1, Nparam)
param[indposest]  = reshape(sum(theta.*wght,1)',Nparam)

Nparam = 11
NY = 21
NX = 4

i = 1
c = 2
np = 3
nf = 4
zp = 5
zf = 6
zs = 7
theta = 8
phi = 9
pi = 10
Y = 11
v = 12

A = 13
mu = 14
g = 15

E_c = 16
E_zp = 17
E_zf = 18
E_theta = 19
E_phi = 20
E_pi = 21

e_A = 1
e_mu = 2
e_g = 3
e_m = 4

rho = param[1]
sigma = param[2]
eta = param[3]
m = param[4]
sigz = param[5]
epsilon = param[6]
beta = param[7]
gamma = param[8]
F = param[9]
b = param[10]
xif = param[11]
s = param[12]
gsY = param[13]

rho_A = param[14]
rho_mu = param[15]
rho_g = param[16]
r_i = param[17]
r_pi = param[18]
r_y = param[19]
psi = param[20]

sig_A = param[21]
sig_mu = param[22]
sig_g = param[23]
sig_m = param[24]

#Distribution

Ez = exp(sigz^2/2)
CDF(x) = normcdf(0,sigz,log(max(x,0)))
PDF(x) = normpdf(0,sigz,log(max(x,0)))/max(x,0)
II(x) = exp(sigz^2/2)*normcdf(0,1,(sigz^2-log(max(x,0)))/sigz)
int(x) = II(x) - x*(1-CDF(x))

GG,RR,SDX,_,_ = funcmod(param)

N = 1000
T = 500
Tburn = 100
e = randn(N,T,NX)
sim = zeros(N,T,21)

for n in 1:N
    for t in 1:T-1
        sim[n,t+1,:] = GG*sim[n,t,:] + RR*SDX*e[n,t,:]
    end
end

SimY = zeros(N,9,T)

SimY[:,1,:] = sim[:,:,Y]
SimY[:,2,:] = 100*log.( (ss[np]*exp.(sim[:,:,np]/100) + ss[nf]*exp.(sim[:,:,nf]/100))/(ss[np]+ss[nf]) )
SimY[:,3,:] = sim[:,:,v]-sigma*sim[:,:,theta]+100*log.( (1-CDF.(ss[zs]*exp.(sim[:,:,zs]/100))) / (1-CDF(ss[zs])) )
SimY[:,4,:] = sim[:,:,v]-sigma*sim[:,:,theta]+100*log.( (CDF.(ss[zs]*exp.(sim[:,:,zs]/100)) - CDF.(ss[zf]*exp.(sim[:,:,zf]/100))) / (CDF(ss[zs])-CDF(ss[zf])) )
SimY[:,5,:] = 100*log.( (CDF.(ss[zs]*exp.(sim[:,:,zs]/100)) - CDF.(ss[zf]*exp.(sim[:,:,zf]/100))) ./ ( 1 - CDF.(exp.(sim[:,:,zf]/100)*ss[zf]) ) ) - 100*log((CDF(ss[zs])-CDF(ss[zf]))/(1-CDF(ss[zf])))
# SimY[:,4,:] = sim[:,:,v]-sigma*sim[:,:,theta]+(ss[zs]*PDF(ss[zs])*sim[:,:,zs] - ss[zf]*PDF(ss[zf])*sim[:,:,zf])/(CDF(ss[zs])-CDF(ss[zf]))
# c_zs = PDF(ss[zs])*ss[zs]/(CDF(ss[zs])-CDF(ss[zf]))
# c_zf = -(1-CDF(ss[zs]))*PDF(ss[zf])*ss[zf]/(1-CDF(ss[zf]))/(CDF(ss[zs])-CDF(ss[zf]))
# SimY[:,5,:] = c_zs*sim[:,:,zs] + c_zf*sim[:,:,zf]
SimY[:,6,:] = sim[:,:,nf] - SimY[:,2,:]
SimY[:,7,:] = [zeros(N) sim[:,1:T-1,np]] -  SimY[:,2,:] + 100*log.(CDF.(ss[zp]*exp.(sim[:,:,zp]/100))/CDF(ss[zp]))
SimY[:,8,:] = [zeros(N) sim[:,1:T-1,nf]] - SimY[:,2,:]
SimY[:,9,:] = sim[:,:,v]

sim_stddev = zeros(9)
for i in 1:9
    sim_stddev[i] = mean(std(SimY[:,i,Tburn:end],2))
end

sim_cor = zeros(9,9)
for i in 1:9
    for j in 1:9
        for n in 1:N
            sim_cor[i,j] += cor(SimY[n,i,Tburn:end],SimY[n,j,Tburn:end])
        end
    end
end

sim_cor /= N

#==============================================================================#
#================================Table=========================================#
#==============================================================================#

Y = 1
n = 2
jcp = 3
jcf = 4
mufsmu = 5
nfsn = 6
jdp = 7
jdf = 8
v = 9

TableLatex = Array{Any,2}(10,5)
TableLatex[1,:] = ["","\\sigma","\$Cov(Y,.)\$","\\sigma","\$Cov(Y,.)\$"]
TableLatex[2:end,1] = ["\$Y\$","\$n\$","\$jc^p\$","\$jc^f\$","\$\\frac{\\mu^f}{\\mu}\$",
                        "\$\\frac{n^f}{n}\$","Endogenous \$jd^p\$","\$jd^f\$","\$v\$"]
TableLatex[2:end,2] = stddev[[Y,n,jcp,jcf,mufsmu,nfsn,jdp,jdf,v]]
TableLatex[2:end,3] = correlation[Y,[Y,n,jcp,jcf,mufsmu,nfsn,jdp,jdf,v]]
TableLatex[2:end,4] = sim_stddev[[Y,n,jcp,jcf,mufsmu,nfsn,jdp,jdf,v]]
TableLatex[2:end,5] = sim_cor[Y,[Y,n,jcp,jcf,mufsmu,nfsn,jdp,jdf,v]]


TableLatex = Array{Any,2}(10,5)
TableLatex[1,:] = ["","Std. Dev.","\$Cor(Y,.)\$","Std. Dev.","\$Cor(Y,.)\$"]
TableLatex[2:end,1] = ["\$Y\$","\$n\$","\$jc^p\$","\$jc^f\$","\$\\frac{\\mu^f}{\\mu}\$",
                        "\$\\frac{n^f}{n}\$","Endogenous \$jd^p\$","\$jd^f\$","\$v\$"]
TableLatex[2:end,2] = stddev[[Y,n,jcp,jcf,mufsmu,nfsn,jdp,jdf,v]]
TableLatex[2:end,3] = correlation[Y,[Y,n,jcp,jcf,mufsmu,nfsn,jdp,jdf,v]]
TableLatex[2:end,4] = sim_stddev[[Y,n,jcp,jcf,mufsmu,nfsn,jdp,jdf,v]]
TableLatex[2:end,5] = sim_cor[Y,[Y,n,jcp,jcf,mufsmu,nfsn,jdp,jdf,v]]
writedlm("Moments.txt", TableLatex)
