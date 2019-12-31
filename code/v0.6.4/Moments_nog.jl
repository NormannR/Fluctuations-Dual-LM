# graph_dir = "E:\\Dropbox\\Ch2\\latex\\figures"
# working_dir = "E:\\Dropbox\\Ch2\\code\\23_02_2019_Euro"
graph_dir = "C:\\Users\\Normann\\Dropbox\\Ch2\\latex\\figures_nog"
working_dir = "C:\\Users\\Normann\\Dropbox\\Ch2\\code\\23_02_2019_Euro"
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

include("SMCfunctions_nog.jl")
include("input\\InitialGuess_nog.jl")
ss = steady_state_eval(param)
funcmod(param) = linearized_model_ss(param,ss)

data = DataFrame(load("input\\LMmoments.xlsx", "04032019!A1:J95"))
T = size(data,1)
m = 4
h = 8
p = 4

Filtered,Ind = HamiltonFilter(data[:,[:Y,:n,:jcp,:jcf,:mufsmu,:nfsn,:endo_jdp,:exo_jdf,:v]],h,p)
Filtered[Ind[3,1]:Ind[3,2],:mufsmu] = mean(Filtered[Ind[3,1]:Ind[3,2],:jcp])*(- Filtered[Ind[3,1]:Ind[3,2],:jcp] + Filtered[Ind[3,1]:Ind[3,2],:jcf])/(mean(Filtered[Ind[3,1]:Ind[3,2],:jcp]) + mean(Filtered[Ind[3,1]:Ind[3,2],:jcp]))

stddev = ObsStd(Filtered,Ind)
correlation = ObsCor(Filtered,Ind)

#==============================================================================#
#================================Stylized Facts================================#
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

figure()

subplot(2,2,1)
plot(Ind[Y,1]:Ind[Y,2],Filtered[Ind[Y,1]:Ind[Y,2],Y],"k:")
plot(Ind[n,1]:Ind[n,2],Filtered[Ind[n,1]:Ind[n,2],n],"k-")
plot(Ind[mufsmu,1]:Ind[mufsmu,2],Filtered[Ind[mufsmu,1]:Ind[mufsmu,2],mufsmu],"k--")
legend(["\$Y\$","\$n\$","\$\\frac{\\mu^f}{\\mu}\$"], loc="lower right")
locs=Ind[Y,1]:8:Ind[Y,2]
labels=[]
for l in Int64.(locs)
    if l >= Ind[Y,1] && l <= Ind[Y,2]
        labels=vcat(labels,[data[p+h+l,:quarter]])
    else
        labels=vcat(labels,[""])
    end
end
xticks(locs, labels, rotation=45)


subplot(2,2,2)
plot(Ind[jcp,1]:Ind[jcp,2],Filtered[Ind[jcp,1]:Ind[jcp,2],Y],"k:")
plot(Ind[jcp,1]:Ind[jcp,2],Filtered[Ind[jcp,1]:Ind[jcp,2],jcp],"k-")
plot(Ind[jcf,1]:Ind[jcf,2],Filtered[Ind[jcf,1]:Ind[jcf,2],jcf],"k--")
legend(["\$Y\$","\$jc^p\$","\$jc^f\$"], loc="lower right")
locs=Ind[jcp,1]:8:Ind[jcp,2]
labels=[]
for l in Int64.(locs)
    if l >= Ind[jcp,1] && l <= Ind[jcp,2]
        labels=vcat(labels,[data[p+h+l,:quarter]])
    else
        labels=vcat(labels,[""])
    end
end
xticks(locs, labels, rotation=45)


subplot(2,2,3)
plot(Ind[jdp,1]:Ind[jdp,2],Filtered[Ind[jdp,1]:Ind[jdp,2],Y],"k:")
plot(Ind[jdp,1]:Ind[jdp,2],Filtered[Ind[jdp,1]:Ind[jdp,2],jdp],"k-")
plot(Ind[jdf,1]:Ind[jdf,2],Filtered[Ind[jdf,1]:Ind[jdf,2],jdf],"k--")
legend(["\$Y\$","\$jd^p\$","\$jd^f\$"], loc="upper right")
locs=Ind[jdp,1]:8:Ind[jdp,2]
labels=[]
for l in Int64.(locs)
    if l >= Ind[jdp,1] && l <= Ind[jdp,2]
        labels=vcat(labels,[data[p+h+l,:quarter]])
    else
        labels=vcat(labels,[""])
    end
end
xticks(locs, labels, rotation=45)

subplot(2,2,4)
plot(Ind[v,1]:Ind[v,2],Filtered[Ind[v,1]:Ind[v,2],Y],"k:")
plot(Ind[nfsn,1]:Ind[nfsn,2],Filtered[Ind[nfsn,1]:Ind[nfsn,2],nfsn],"k-")
plot(Ind[v,1]:Ind[v,2],Filtered[Ind[v,1]:Ind[v,2],v],"k--")
legend(["\$Y\$","\$n^f / n\$","\$v\$"], loc="lower right")
locs=Ind[np,1]:8:Ind[np,2]
labels=[]
for l in Int64.(locs)
    if l >= Ind[np,1] && l <= Ind[np,2]
        labels=vcat(labels,[data[p+h+l,:quarter]])
    else
        labels=vcat(labels,[""])
    end
end
xticks(locs, labels, rotation=45)
tight_layout()
cd(graph_dir)
savefig("StylizedFacts.pdf")

#==============================================================================#
#================================Simulation====================================#
#==============================================================================#

cd(working_dir)

Nparam = 9
NY = 14
NX = 3

# var = JLD.load("LT_SHORT.jld")
# var = JLD.load("LONG.jld")
var = JLD.load("LONG_nog.jld")

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

E_c = 15
E_zp = 16
E_zf = 17
E_theta = 18
E_phi = 19
E_pi = 20

e_A = 1
e_mu = 2
e_m = 3

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
r_i = param[16]
r_pi = param[17]
r_y = param[18]
psi = param[19]

sig_A = param[20]
sig_mu = param[21]
sig_m = param[22]

#Distribution

Ez = exp(sigz^2/2)
CDF(x) = normcdf(0,sigz,log(max(x,0)))
PDF(x) = normpdf(0,sigz,log(max(x,0)))/max(x,0)
II(x) = exp(sigz^2/2)*normcdf(0,1,(sigz^2-log(max(x,0)))/sigz)
int(x) = II(x) - x*(1-CDF(x))

draws = var["theta"]
weights = var["weights"]

N = size(draws,1)
T = 400
Tburn = 100
e = rand(N,T,NX)
sim = zeros(N,T,20)


for n in 1:N
    param[indposest] = draws[n,:]
    GG,RR,SDX,_,_ = funcmod(param)
    for t in 1:T-1
        sim[n,t+1,:] = GG*sim[n,t,:] + RR*SDX*e[n,t,:]
    end
end

SimY = zeros(N,9,T)

SimY[:,1,:] = sim[:,:,Y]
SimY[:,2,:] = ss[np]*sim[:,:,np]/(ss[np]+ss[nf]) + ss[nf]*sim[:,:,nf]/(ss[np]+ss[nf])
SimY[:,3,:] = sim[:,:,v]-sigma*sim[:,:,theta]-ss[zs]*PDF(ss[zs])*sim[:,:,zs]/(1-CDF(ss[zs]))
SimY[:,4,:] = sim[:,:,v]-sigma*sim[:,:,theta]+(ss[zs]*PDF(ss[zs])*sim[:,:,zs] - ss[zf]*PDF(ss[zf])*sim[:,:,zf])/(CDF(ss[zs])-CDF(ss[zf]))

c_zs = PDF(ss[zs])*ss[zs]/(CDF(ss[zs])-CDF(ss[zf]))
c_zf = -PDF(ss[zf])*ss[zf]/(1-CDF(ss[zf]))/(CDF(ss[zs])-CDF(ss[zf]))
SimY[:,5,:] = c_zs*sim[:,:,zs] + c_zf*sim[:,:,zf]
SimY[:,6,:] =  - ss[np]*sim[:,:,np]/(ss[np]+ss[nf]) + ss[nf]*sim[:,:,nf]/(ss[np]+ss[nf])
SimY[:,7,:] = [zeros(N) sim[:,1:T-1,np]] + PDF(ss[zp])*ss[zp]*sim[:,:,zp]/CDF(ss[zp]) - SimY[:,2,:]
SimY[:,8,:] = [zeros(N) sim[:,1:T-1,nf]] - SimY[:,2,:]
SimY[:,9,:] = sim[:,:,v]


sim_stddev = zeros(9)
for i in 1:9
    sim_stddev[i] = sum(std(SimY[:,i,Tburn:end],2).*weights)
end

sim_cor = zeros(9,9)
for i in 1:9
    for j in 1:9
        for n in 1:N
            sim_cor[i,j] += cor(SimY[n,i,Tburn:end],SimY[n,j,Tburn:end])*weights[n]
        end
    end
end

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
                        "\$\\frac{n^f}{n}\$","\$jd^p\$","\$jd^f\$","\$v\$"]
TableLatex[2:end,2] = stddev[[Y,n,jcp,jcf,mufsmu,nfsn,jdp,jdf,v]]
TableLatex[2:end,3] = correlation[Y,[Y,n,jcp,jcf,mufsmu,nfsn,jdp,jdf,v]]
TableLatex[2:end,4] = sim_stddev[[Y,n,jcp,jcf,mufsmu,nfsn,jdp,jdf,v]]
TableLatex[2:end,5] = sim_cor[Y,[Y,n,jcp,jcf,mufsmu,nfsn,jdp,jdf,v]]
