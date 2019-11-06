graph_dir = "E:\\Dropbox\\Ch2\\latex\\figures_habits_capital"
working_dir = "E:\\Dropbox\\Ch2\\code\\23_02_2019_Euro"
# graph_dir = "C:\\Users\\Normann\\Dropbox\\Ch2\\latex\\figures_habits_capital"
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
using StatsFuns.normcdf
using StatsFuns.normpdf
using Gensys.gensysdt

include("SMCfunctions_habits_capital.jl")
include("input\\InitialGuess_habits_capital.jl")
ss = steady_state_eval(param)
# funcmod(param) = linearized_model_ss(param,ss)

Nparam = 19
NX = 6

var = JLD.load("LONG_habits_capital.jld")
theta = var["theta"]
wght = repmat(var["weights"], 1, Nparam)
mu  = reshape(sum(theta.*wght,1)',Nparam)

param[indposest] = mu

GG,RR,SDX,ZZ,eu=funcmod(param)

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
alpha = param[14]
delta = param[15]

rho_A = param[16]
rho_mu = param[17]
r_i = param[18]
r_pi = param[19]
r_y = param[20]
psi = param[21]

sig_A = param[22]
sig_mu = param[23]
sig_m = param[24]

h = param[25]
sig_c = param[26]

psi_nu = param[27]
eta_k = param[28]

rho_g = param[29]
sig_g = param[30]
rho_b = param[31]
sig_b = param[32]
rho_i = param[33]
sig_i = param[34]

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
plot(0:Nperiods-1, IRF[r,e_A,:],"k--",linewidth=1)
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
#================================Shock in g====================================#
#==============================================================================#
figure()

# i,pi,Y
subplot(2,3,1)
plot(0:Nperiods-1, IRF[r,e_g,:],"k-",linewidth=1)
plot(0:Nperiods-1, IRF[pi,e_g,:],"k--",linewidth=1)
legend(["\$i\$","\$\\pi\$"])

# theta, mufsmu

subplot(2,3,2)
c_zs = PDF(ss[zs])*ss[zs]/(CDF(ss[zs])-CDF(ss[zf]))
c_zf = -PDF(ss[zf])*ss[zf]/(1-CDF(ss[zf]))/(CDF(ss[zs])-CDF(ss[zf]))
plot(0:Nperiods-1, c_zs*IRF[zs,e_g,:]+c_zf*IRF[zf,e_g,:],"k--",linewidth=1)
plot(0:Nperiods-1, IRF[theta,e_g,:],"k-",linewidth=1)
legend(["\$\\frac{\\mu^f}{\\mu}\$","\$\\theta\$"])

# theta, mufsmu

subplot(2,3,3)
plot(0:Nperiods-1, IRF[np,e_g,:],"k-",linewidth=1)
plot(0:Nperiods-1, IRF[nf,e_g,:],"k--",linewidth=1)
plot(0:Nperiods-1, IRF[Y,e_g,:],"k:",linewidth=1)
legend(["\$n^p\$","\$n^f\$","\$Y\$"])

# Thresholds

subplot(2,3,4)
plot(0:Nperiods-1, IRF[zp,e_g,:],"k-",linewidth=1)
plot(0:Nperiods-1, IRF[zf,e_g,:],"k--",linewidth=1)
plot(0:Nperiods-1, IRF[zs,e_g,:],"k:",linewidth=1)
legend(["\$z^p\$","\$z^f\$","\$z^*\$"])

# Job creation

subplot(2,3,5)
plot(0:Nperiods-1, IRF[v,e_g,:]-sigma*IRF[theta,e_g,:]-ss[zs]*PDF(ss[zs])*IRF[zs,e_g,:]/(1-CDF(ss[zs])),"k-",linewidth=1)
plot(0:Nperiods-1, IRF[v,e_g,:]-sigma*IRF[theta,e_g,:]+(ss[zs]*PDF(ss[zs])*IRF[zs,e_g,:] - ss[zf]*PDF(ss[zf])*IRF[zf,e_g,:])/(CDF(ss[zs])-CDF(ss[zf])),"k--",linewidth=1)
legend(["\$jc^p\$","\$jc^f\$"])

# Job destruction

subplot(2,3,6)
plot(0:Nperiods-1, vcat(0,IRF[np,e_g,1:Nperiods-1]) + (1-s)*PDF(ss[zp])*ss[zp]*IRF[zp,e_g,:]/(s + (1-s)*CDF(ss[zp])),"k-",linewidth=1)
plot(0:Nperiods-1, vcat(0,IRF[nf,e_g,1:Nperiods-1]),"k--",linewidth=1)
legend(["\$jd^p\$","\$jd^f\$"])

tight_layout()

cd(graph_dir)
savefig("IRF_g.pdf")


#==============================================================================#
#================================Shock in m====================================#
#==============================================================================#
figure()

# i,pi,Y
subplot(2,3,1)
plot(0:Nperiods-1, IRF[r,e_m,:],"k-",linewidth=1)
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
plot(0:Nperiods-1, IRF[r,e_mu,:],"k--",linewidth=1)
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


#==============================================================================#
#================================Shock in b====================================#
#==============================================================================#
figure()

# i,pi,Y

subplot(2,3,1)
plot(0:Nperiods-1, IRF[Y,e_b,:],"k-",linewidth=1)
plot(0:Nperiods-1, IRF[r,e_b,:],"k--",linewidth=1)
plot(0:Nperiods-1, IRF[pi,e_b,:],"k:",linewidth=1)
legend(["\$Y\$","\$i\$","\$\\pi\$"])

# theta, mufsmu

subplot(2,3,2)
c_zs = PDF(ss[zs])*ss[zs]/(CDF(ss[zs])-CDF(ss[zf]))
c_zf = -PDF(ss[zf])*ss[zf]/(1-CDF(ss[zf]))/(CDF(ss[zs])-CDF(ss[zf]))
plot(0:Nperiods-1, c_zs*IRF[zs,e_b,:]+c_zf*IRF[zf,e_b,:],"k--",linewidth=1)
plot(0:Nperiods-1, IRF[theta,e_b,:],"k-",linewidth=1)
legend(["\$\\frac{\\mu^f}{\\mu}\$","\$\\theta\$"])

# theta, mufsmu

subplot(2,3,3)
plot(0:Nperiods-1, IRF[np,e_b,:],"k-",linewidth=1)
plot(0:Nperiods-1, IRF[nf,e_b,:],"k--",linewidth=1)
legend(["\$n^p\$","\$n^f\$"])

# Thresholds

subplot(2,3,4)
plot(0:Nperiods-1, IRF[zp,e_b,:],"k-",linewidth=1)
plot(0:Nperiods-1, IRF[zf,e_b,:],"k--",linewidth=1)
plot(0:Nperiods-1, IRF[zs,e_b,:],"k:",linewidth=1)
legend(["\$z^p\$","\$z^f\$","\$z^*\$"])

# Job creation

subplot(2,3,5)
plot(0:Nperiods-1, IRF[v,e_b,:]-sigma*IRF[theta,e_b,:]-ss[zs]*PDF(ss[zs])*IRF[zs,e_b,:]/(1-CDF(ss[zs])),"k-",linewidth=1)
plot(0:Nperiods-1, IRF[v,e_b,:]-sigma*IRF[theta,e_b,:]+(ss[zs]*PDF(ss[zs])*IRF[zs,e_b,:] - ss[zf]*PDF(ss[zf])*IRF[zf,e_b,:])/(CDF(ss[zs])-CDF(ss[zf])),"k--",linewidth=1)
legend(["\$jc^p\$","\$jc^f\$"])

# Job destruction

subplot(2,3,6)
plot(0:Nperiods-1, vcat(0,IRF[np,e_b,1:Nperiods-1]) + (1-s)*PDF(ss[zp])*ss[zp]*IRF[zp,e_b,:]/(s + (1-s)*CDF(ss[zp])),"k-",linewidth=1)
plot(0:Nperiods-1, vcat(0,IRF[nf,e_b,1:Nperiods-1]),"k--",linewidth=1)
legend(["\$jd^p\$","\$jd^f\$"])

tight_layout()

cd(graph_dir)
savefig("IRF_b.pdf")


#==============================================================================#
#================================Shock in i====================================#
#==============================================================================#
figure()

# i,pi,Y

subplot(2,3,1)
plot(0:Nperiods-1, IRF[Y,e_i,:],"k-",linewidth=1)
plot(0:Nperiods-1, IRF[r,e_i,:],"k--",linewidth=1)
plot(0:Nperiods-1, IRF[pi,e_i,:],"k:",linewidth=1)
legend(["\$Y\$","\$i\$","\$\\pi\$"])

# theta, mufsmu

subplot(2,3,2)
c_zs = PDF(ss[zs])*ss[zs]/(CDF(ss[zs])-CDF(ss[zf]))
c_zf = -PDF(ss[zf])*ss[zf]/(1-CDF(ss[zf]))/(CDF(ss[zs])-CDF(ss[zf]))
plot(0:Nperiods-1, c_zs*IRF[zs,e_i,:]+c_zf*IRF[zf,e_i,:],"k--",linewidth=1)
plot(0:Nperiods-1, IRF[theta,e_i,:],"k-",linewidth=1)
legend(["\$\\frac{\\mu^f}{\\mu}\$","\$\\theta\$"])

# theta, mufsmu

subplot(2,3,3)
plot(0:Nperiods-1, IRF[np,e_i,:],"k-",linewidth=1)
plot(0:Nperiods-1, IRF[nf,e_i,:],"k--",linewidth=1)
legend(["\$n^p\$","\$n^f\$"])

# Thresholds

subplot(2,3,4)
plot(0:Nperiods-1, IRF[zp,e_i,:],"k-",linewidth=1)
plot(0:Nperiods-1, IRF[zf,e_i,:],"k--",linewidth=1)
plot(0:Nperiods-1, IRF[zs,e_i,:],"k:",linewidth=1)
legend(["\$z^p\$","\$z^f\$","\$z^*\$"])

# Job creation

subplot(2,3,5)
plot(0:Nperiods-1, IRF[v,e_i,:]-sigma*IRF[theta,e_i,:]-ss[zs]*PDF(ss[zs])*IRF[zs,e_i,:]/(1-CDF(ss[zs])),"k-",linewidth=1)
plot(0:Nperiods-1, IRF[v,e_i,:]-sigma*IRF[theta,e_i,:]+(ss[zs]*PDF(ss[zs])*IRF[zs,e_i,:] - ss[zf]*PDF(ss[zf])*IRF[zf,e_i,:])/(CDF(ss[zs])-CDF(ss[zf])),"k--",linewidth=1)
legend(["\$jc^p\$","\$jc^f\$"])

# Job destruction

subplot(2,3,6)
plot(0:Nperiods-1, vcat(0,IRF[np,e_i,1:Nperiods-1]) + (1-s)*PDF(ss[zp])*ss[zp]*IRF[zp,e_i,:]/(s + (1-s)*CDF(ss[zp])),"k-",linewidth=1)
plot(0:Nperiods-1, vcat(0,IRF[nf,e_i,1:Nperiods-1]),"k--",linewidth=1)
legend(["\$jd^p\$","\$jd^f\$"])

tight_layout()

cd(graph_dir)
savefig("IRF_i.pdf")
