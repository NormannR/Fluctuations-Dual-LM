# graph_dir = "E:\\Dropbox\\Ch2\\latex\\figures_habits_capital"
# working_dir = "E:\\Dropbox\\Ch2\\code\\23_02_2019_Euro"
graph_dir = "C:\\Users\\Normann\\Dropbox\\Ch2\\latex\\figures_habits_capital"
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

include("SMCfunctions_habits_capital.jl")
include("input\\InitialGuess_habits_capital.jl")
ss = steady_state_eval(param)
linearized_model(param) = linearized_model_ss(param,ss)
funcmod = linearized_model
# data = DataFrame(load("input\\LMmoments.xlsx", "Transform!A1:J95"))
# T = size(data,1)
# m = 4
# h = 8
# p = 4
#
# Filtered,Ind = HamiltonFilter(data[:,[:jcp,:jcf,:jdp,:jdf,:np,:nf,:Y,:n,:mufsmu]],h,p)
# Filtered[Ind[2,1]:Ind[2,2],:mufsmu] = mean(Filtered[Ind[2,1]:Ind[2,2],:jcp])*(- Filtered[Ind[2,1]:Ind[2,2],:jcp] + Filtered[Ind[2,1]:Ind[2,2],:jcf])/(mean(Filtered[Ind[2,1]:Ind[2,2],:jcp]) + mean(Filtered[Ind[2,1]:Ind[2,2],:jcp]))
# Filtered[Ind[2,1]:Ind[2,2],:mufsmu] = mean(Filtered[Ind[2,1]:Ind[2,2],:jcp])*(- Filtered[Ind[2,1]:Ind[2,2],:jcp] + Filtered[Ind[2,1]:Ind[2,2],:jcf])/(mean(Filtered[Ind[2,1]:Ind[2,2],:jcp]) + mean(Filtered[Ind[2,1]:Ind[2,2],:jcp]))
# stddev = ObsStd(Filtered,Ind)
# correlation = ObsCor(Filtered,Ind)

data = DataFrame(load("input\\LMmoments.xlsx", "04032019!A1:J95"))
T = size(data,1)
m = 4
h = 8
p = 4

Filtered,Ind = HamiltonFilter(data[:,[:Y,:n,:jcp,:jcf,:mufsmu,:nfsn,:endo_jdp,:exo_jdf,:v]],h,p)
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
locs=Ind[v,1]:8:Ind[v,2]
labels=[]
for l in Int64.(locs)
    if l >= Ind[v,1] && l <= Ind[v,2]
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

Nparam = 19
NY = 36
NX = 6

var = JLD.load("LT_SHORT_habits_capital.jld")

r = 1
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
k = 13
kp = 14
nu = 15
qk = 16
rk = 17
mc = 18
X = 19
i = 20

A = 21
mu = 22

E_c = 23
E_zp = 24
E_zf = 25
E_theta = 26
E_phi = 27
E_pi = 28

lambda = 29
E_lambda = 30
E_qk = 31
E_i = 32
g = 33
E_rk = 34

s_i = 35
s_b = 36

#Perturbations

e_A = 1
e_mu = 2
e_m = 3
e_g = 4
e_b = 5
e_i = 6

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
Tburn = 50
e = rand(N,T,NX)
sim = zeros(N,T,NY)
e[:,:,5] = 0.

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
c_zf = -(1-CDF(ss[zs]))*PDF(ss[zf])*ss[zf]/(1-CDF(ss[zf]))/(CDF(ss[zs])-CDF(ss[zf]))
SimY[:,5,:] = c_zs*sim[:,:,zs] + c_zf*sim[:,:,zf]
SimY[:,6,:] =  - ss[np]*sim[:,:,np]/(ss[np]+ss[nf]) + ss[np]*sim[:,:,nf]/(ss[np]+ss[nf])
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
TableLatex[2:end,2] = stddev[[Y,n,jcp,jcf,mufsmu,nfsn,jdp,jdf,v]]/stddev[Y]
TableLatex[2:end,3] = correlation[Y,[Y,n,jcp,jcf,mufsmu,nfsn,jdp,jdf,v]]
TableLatex[2:end,4] = sim_stddev[[Y,n,jcp,jcf,mufsmu,nfsn,jdp,jdf,v]]/sim_stddev[Y]
TableLatex[2:end,5] = sim_cor[Y,[Y,n,jcp,jcf,mufsmu,nfsn,jdp,jdf,v]]
