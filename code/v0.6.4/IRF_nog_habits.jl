# graph_dir = "E:\\Dropbox\\Ch2\\latex\\figures_nog"
# working_dir = "E:\\Dropbox\\Ch2\\code\\23_02_2019_Euro"
graph_dir = "C:\\Users\\Normann\\Dropbox\\Ch2\\latex\\figures_nog_habits"
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
using StatsFuns.normcdf
using StatsFuns.normpdf
using Gensys.gensysdt

include("SMCfunctions_nog_habits.jl")
include("input\\InitialGuess_nog_habits.jl")
ss = steady_state_eval(param)
# funcmod(param) = linearized_model_ss(param,ss)

Nparam = 11
NX = 3

var = JLD.load("LONG_nog_habits.jld")
theta = var["theta"]
wght = repmat(var["weights"], 1, Nparam)
mu  = reshape(sum(theta.*wght,1)',Nparam)

param[indposest] = mu

GG,RR,SDX,ZZ,eu=funcmod(param)

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

Nperiods=48
IRF=zeros(size(GG,1),size(RR,2),Nperiods)
Impulse=RR*SDX
IRF[:,:,1]=Impulse
for jj=2:Nperiods
    Impulse=GG*Impulse
    IRF[:,:,jj]=Impulse
end


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

#==============================================================================#
#================================Shock in A====================================#
#==============================================================================#
figure()
# i,pi,Y

subplot(2,3,1)
plot(0:Nperiods-1, IRF[Y,e_A,:],"k-",linewidth=1)
plot(0:Nperiods-1, IRF[i,e_A,:],"k--",linewidth=1)
plot(0:Nperiods-1, IRF[pi,e_A,:],"k:",linewidth=1)
legend(["\$Y\$","\$i\$","\$\\pi\$"])

# theta, mufsmu

subplot(2,3,2)
c_zs = PDF(ss[zs])*ss[zs]/(CDF(ss[zs])-CDF(ss[zf]))
c_zf = -PDF(ss[zf])*ss[zf]/(1-CDF(ss[zf]))/(CDF(ss[zs])-CDF(ss[zf]))
plot(0:Nperiods-1, c_zs*IRF[zs,e_A,:]+c_zf*IRF[zf,e_A,:],"k--",linewidth=1)
plot(0:Nperiods-1, IRF[theta,e_A,:],"k-",linewidth=1)
legend(["\$\\frac{\\mu^f}{\\mu}\$","\$\\theta\$"])

# theta, mufsmu

subplot(2,3,3)
plot(0:Nperiods-1, IRF[np,e_A,:],"k-",linewidth=1)
plot(0:Nperiods-1, IRF[nf,e_A,:],"k--",linewidth=1)
legend(["\$n^p\$","\$n^f\$"])

# Thresholds

subplot(2,3,4)
plot(0:Nperiods-1, IRF[zp,e_A,:],"k-",linewidth=1)
plot(0:Nperiods-1, IRF[zf,e_A,:],"k--",linewidth=1)
plot(0:Nperiods-1, IRF[zs,e_A,:],"k:",linewidth=1)
legend(["\$z^p\$","\$z^f\$","\$z^*\$"])

# Job creation

subplot(2,3,5)
plot(0:Nperiods-1, IRF[v,e_A,:]-sigma*IRF[theta,e_A,:]-ss[zs]*PDF(ss[zs])*IRF[zs,e_A,:]/(1-CDF(ss[zs])),"k-",linewidth=1)
plot(0:Nperiods-1, IRF[v,e_A,:]-sigma*IRF[theta,e_A,:]+(ss[zs]*PDF(ss[zs])*IRF[zs,e_A,:] - ss[zf]*PDF(ss[zf])*IRF[zf,e_A,:])/(CDF(ss[zs])-CDF(ss[zf])),"k--",linewidth=1)
legend(["\$jc^p\$","\$jc^f\$"])

# Job destruction

subplot(2,3,6)
plot(0:Nperiods-1, vcat(0,IRF[np,e_A,1:Nperiods-1]) + (1-s)*PDF(ss[zp])*ss[zp]*IRF[zp,e_A,:]/(s + (1-s)*CDF(ss[zp])),"k-",linewidth=1)
plot(0:Nperiods-1, vcat(0,IRF[nf,e_A,1:Nperiods-1]),"k--",linewidth=1)
legend(["\$jd^p\$","\$jd^f\$"])

tight_layout()

cd(graph_dir)
savefig("IRF_A.pdf")


#==============================================================================#
#================================Shock in m====================================#
#==============================================================================#
figure()

# i,pi,Y
subplot(2,3,1)
plot(0:Nperiods-1, IRF[i,e_m,:],"k-",linewidth=1)
plot(0:Nperiods-1, IRF[pi,e_m,:],"k--",linewidth=1)
legend(["\$i\$","\$\\pi\$"])

# theta, mufsmu

subplot(2,3,2)
c_zs = PDF(ss[zs])*ss[zs]/(CDF(ss[zs])-CDF(ss[zf]))
c_zf = -PDF(ss[zf])*ss[zf]/(1-CDF(ss[zf]))/(CDF(ss[zs])-CDF(ss[zf]))
plot(0:Nperiods-1, c_zs*IRF[zs,e_m,:]+c_zf*IRF[zf,e_m,:],"k--",linewidth=1)
plot(0:Nperiods-1, IRF[theta,e_m,:],"k-",linewidth=1)
legend(["\$\\frac{\\mu^f}{\\mu}\$","\$\\theta\$"])

# theta, mufsmu

subplot(2,3,3)
plot(0:Nperiods-1, IRF[np,e_m,:],"k-",linewidth=1)
plot(0:Nperiods-1, IRF[nf,e_m,:],"k--",linewidth=1)
plot(0:Nperiods-1, IRF[Y,e_m,:],"k:",linewidth=1)
legend(["\$n^p\$","\$n^f\$","\$Y\$"])

# Thresholds

subplot(2,3,4)
plot(0:Nperiods-1, IRF[zp,e_m,:],"k-",linewidth=1)
plot(0:Nperiods-1, IRF[zf,e_m,:],"k--",linewidth=1)
plot(0:Nperiods-1, IRF[zs,e_m,:],"k:",linewidth=1)
legend(["\$z^p\$","\$z^f\$","\$z^*\$"])

# Job creation

subplot(2,3,5)
plot(0:Nperiods-1, IRF[v,e_m,:]-sigma*IRF[theta,e_m,:]-ss[zs]*PDF(ss[zs])*IRF[zs,e_m,:]/(1-CDF(ss[zs])),"k-",linewidth=1)
plot(0:Nperiods-1, IRF[v,e_m,:]-sigma*IRF[theta,e_m,:]+(ss[zs]*PDF(ss[zs])*IRF[zs,e_m,:] - ss[zf]*PDF(ss[zf])*IRF[zf,e_m,:])/(CDF(ss[zs])-CDF(ss[zf])),"k--",linewidth=1)
legend(["\$jc^p\$","\$jc^f\$"])

# Job destruction

subplot(2,3,6)
plot(0:Nperiods-1, vcat(0,IRF[np,e_m,1:Nperiods-1]) + (1-s)*PDF(ss[zp])*ss[zp]*IRF[zp,e_m,:]/(s + (1-s)*CDF(ss[zp])),"k-",linewidth=1)
plot(0:Nperiods-1, vcat(0,IRF[nf,e_m,1:Nperiods-1]),"k--",linewidth=1)
legend(["\$jd^p\$","\$jd^f\$"])

tight_layout()

cd(graph_dir)
savefig("IRF_m.pdf")


#==============================================================================#
#================================Shock in mu====================================#
#==============================================================================#
figure()
# i,pi,Y

subplot(2,3,1)
plot(0:Nperiods-1, IRF[Y,e_mu,:],"k-",linewidth=1)
plot(0:Nperiods-1, IRF[i,e_mu,:],"k--",linewidth=1)
plot(0:Nperiods-1, IRF[pi,e_mu,:],"k:",linewidth=1)
legend(["\$Y\$","\$i\$","\$\\pi\$"])

# theta, mufsmu

subplot(2,3,2)
c_zs = PDF(ss[zs])*ss[zs]/(CDF(ss[zs])-CDF(ss[zf]))
c_zf = -PDF(ss[zf])*ss[zf]/(1-CDF(ss[zf]))/(CDF(ss[zs])-CDF(ss[zf]))
plot(0:Nperiods-1, c_zs*IRF[zs,e_mu,:]+c_zf*IRF[zf,e_mu,:],"k--",linewidth=1)
plot(0:Nperiods-1, IRF[theta,e_mu,:],"k-",linewidth=1)
legend(["\$\\frac{\\mu^f}{\\mu}\$","\$\\theta\$"])

# theta, mufsmu

subplot(2,3,3)
plot(0:Nperiods-1, IRF[np,e_mu,:],"k-",linewidth=1)
plot(0:Nperiods-1, IRF[nf,e_mu,:],"k--",linewidth=1)
legend(["\$n^p\$","\$n^f\$"])

# Thresholds

subplot(2,3,4)
plot(0:Nperiods-1, IRF[zp,e_mu,:],"k-",linewidth=1)
plot(0:Nperiods-1, IRF[zf,e_mu,:],"k--",linewidth=1)
plot(0:Nperiods-1, IRF[zs,e_mu,:],"k:",linewidth=1)
legend(["\$z^p\$","\$z^f\$","\$z^*\$"])

# Job creation

subplot(2,3,5)
plot(0:Nperiods-1, IRF[v,e_mu,:]-sigma*IRF[theta,e_mu,:]-ss[zs]*PDF(ss[zs])*IRF[zs,e_mu,:]/(1-CDF(ss[zs])),"k-",linewidth=1)
plot(0:Nperiods-1, IRF[v,e_mu,:]-sigma*IRF[theta,e_mu,:]+(ss[zs]*PDF(ss[zs])*IRF[zs,e_mu,:] - ss[zf]*PDF(ss[zf])*IRF[zf,e_mu,:])/(CDF(ss[zs])-CDF(ss[zf])),"k--",linewidth=1)
legend(["\$jc^p\$","\$jc^f\$"])

# Job destruction

subplot(2,3,6)
plot(0:Nperiods-1, vcat(0,IRF[np,e_mu,1:Nperiods-1]) + (1-s)*PDF(ss[zp])*ss[zp]*IRF[zp,e_mu,:]/(s + (1-s)*CDF(ss[zp])),"k-",linewidth=1)
plot(0:Nperiods-1, vcat(0,IRF[nf,e_mu,1:Nperiods-1]),"k--",linewidth=1)
legend(["\$jd^p\$","\$jd^f\$"])

tight_layout()

cd(graph_dir)
savefig("IRF_mu.pdf")
