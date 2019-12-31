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
using StatsFuns.normcdf
using StatsFuns.normpdf
using Gensys.gensysdt

include("Pol_IRF_Functions.jl")
include("input\\InitialGuess.jl")
ss = steady_state_eval(param)

PolParam = zeros(size(param,1)+2)
PolParam[1:size(param,1)] = param
PolParam[size(param,1)+1:end] = [exp(-log(100)/(5*4)), 1.]
linearized_model(param) = POL_linearized_model_ss(PolParam,ss)
funcmod = linearized_model


Nparam = 11
NY = 16
NX = 5

var = JLD.load("LT_SHORT.jld")
theta = var["theta"]
wght = repmat(var["weights"], 1, Nparam)
mu  = reshape(sum(theta.*wght,1)',Nparam)

PolParam[indposest] = mu

GG,RR,SDX,eu=funcmod(param)

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

eta_F = 22

e_A = 1
e_mu = 2
e_g = 3
e_m = 4
e_F = 5

Nperiods=24
IRF=zeros(size(GG,1),size(RR,2),Nperiods)
Impulse=RR*SDX
IRF[:,:,1]=Impulse
for jj=2:Nperiods
    Impulse=GG*Impulse
    IRF[:,:,jj]=Impulse
end


rho = PolParam[1]
sigma = PolParam[2]
eta = PolParam[3]
m = PolParam[4]
sigz = PolParam[5]
epsilon = PolParam[6]
beta = PolParam[7]
gamma = PolParam[8]
F = PolParam[9]
b = PolParam[10]
xif = PolParam[11]
s = PolParam[12]
gsY = PolParam[13]

rho_A = PolParam[14]
rho_mu = PolParam[15]
rho_g = PolParam[16]
r_i = PolParam[17]
r_pi = PolParam[18]
r_y = PolParam[19]
psi = PolParam[20]

sig_A = PolParam[21]
sig_mu = PolParam[22]
sig_g = PolParam[23]
sig_m = PolParam[24]

rho_F = PolParam[25]
sig_F = PolParam[26]

#Distribution

Ez = exp(sigz^2/2)
CDF(x) = normcdf(0,sigz,log(max(x,0)))
PDF(x) = normpdf(0,sigz,log(max(x,0)))/max(x,0)
II(x) = exp(sigz^2/2)*normcdf(0,1,(sigz^2-log(max(x,0)))/sigz)
int(x) = II(x) - x*(1-CDF(x))

alpha_p = m*ss[theta]^(-sigma)*(1-CDF(ss[zs]))*(II(ss[zs])/(1-CDF(ss[zs])) - ss[zp] - F)
alpha_f = m*ss[theta]^(-sigma)*(CDF(ss[zs])-CDF(ss[zf]))*((II(ss[zf]) - II(ss[zs]))/(CDF(ss[zs])-CDF(ss[zf])) - ss[zf] )

alpha_p /= (alpha_p+alpha_f)
alpha_f = 1 - alpha_p

#==============================================================================#
#================================Shock in F====================================#
#==============================================================================#

figure()

# i,pi,Y

subplot(2,3,1)
plot(0:Nperiods-1, IRF[Y,e_F,:],"k-",linewidth=1)
plot(0:Nperiods-1, IRF[i,e_F,:],"k--",linewidth=1)
plot(0:Nperiods-1, IRF[pi,e_F,:],"k:",linewidth=1)
legend(["\$Y\$","\$i\$","\$\\pi\$"])

# theta, mufsmu

subplot(2,3,2)
c_zs = PDF(ss[zs])*ss[zs]/(CDF(ss[zs])-CDF(ss[zf]))
c_zf = -(1-CDF(ss[zs]))*PDF(ss[zf])*ss[zf]/(1-CDF(ss[zf]))/(CDF(ss[zs])-CDF(ss[zf]))
plot(0:Nperiods-1, c_zs*IRF[zs,e_F,:]+c_zf*IRF[zf,e_F,:],"k--",linewidth=1)
plot(0:Nperiods-1, IRF[theta,e_F,:],"k-",linewidth=1)
legend(["\$\\frac{\\mu^f}{\\mu}\$","\$\\theta\$"])

# theta, mufsmu

subplot(2,3,3)
plot(0:Nperiods-1, IRF[np,e_F,:],"k-",linewidth=1)
plot(0:Nperiods-1, IRF[nf,e_F,:],"k--",linewidth=1)
legend(["\$n^p\$","\$n^f\$"])

# Thresholds

subplot(2,3,4)
plot(0:Nperiods-1, IRF[zp,e_F,:],"k-",linewidth=1)
plot(0:Nperiods-1, IRF[zf,e_F,:],"k--",linewidth=1)
plot(0:Nperiods-1, IRF[zs,e_F,:],"k:",linewidth=1)
legend(["\$z^p\$","\$z^f\$","\$z^*\$"])

# Job creation

subplot(2,3,5)
plot(0:Nperiods-1, IRF[v,e_F,:]-sigma*IRF[theta,e_F,:]-ss[zs]*PDF(ss[zs])*IRF[zs,e_F,:]/(1-CDF(ss[zs])),"k-",linewidth=1)
plot(0:Nperiods-1, IRF[v,e_F,:]-sigma*IRF[theta,e_F,:]+(ss[zs]*PDF(ss[zs])*IRF[zs,e_F,:] - ss[zf]*PDF(ss[zf])*IRF[zf,e_F,:])/(CDF(ss[zs])-CDF(ss[zf])),"k--",linewidth=1)
legend(["\$jc^p\$","\$jc^f\$"])

# Job destruction

subplot(2,3,6)
plot(0:Nperiods-1, vcat(0,IRF[np,e_F,1:Nperiods-1]) + (1-s)*PDF(ss[zp])*ss[zp]*IRF[zp,e_F,:]/(s + (1-s)*CDF(ss[zp])),"k-",linewidth=1)
plot(0:Nperiods-1, vcat(0,IRF[nf,e_F,1:Nperiods-1]),"k--",linewidth=1)
legend(["\$jd^p\$","\$jd^f\$"])

tight_layout()

cd(graph_dir)
savefig("IRF_F.pdf")

cd(working_dir)

#==============================================================================#
#================================Permanent Shocks in F=========================#
#==============================================================================#

PolParam = zeros(size(param,1)+2)
PolParam[1:size(param,1)] = param
PolParam[size(param,1)+1:end] = [0.8, 10.]
linearized_model(param) = POL_linearized_model_ss(PolParam,ss)
funcmod = linearized_model

Nparam = 11
NY = 16
NX = 5

var = JLD.load("LT_SHORT.jld")
theta = var["theta"]
wght = repmat(var["weights"], 1, Nparam)
mu  = reshape(sum(theta.*wght,1)',Nparam)

PolParam[indposest] = mu

GG,RR,SDX,eu=funcmod(param)

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

eta_F = 22

e_A = 1
e_mu = 2
e_g = 3
e_m = 4
e_F = 5

Nperiods=12
IRF=zeros(size(GG,1),size(RR,2),Nperiods)
Impulse=-RR*SDX
IRF[:,:,1]=Impulse
for jj=2:Nperiods
    Impulse=GG*Impulse
    IRF[:,:,jj]=Impulse
end


rho = PolParam[1]
sigma = PolParam[2]
eta = PolParam[3]
m = PolParam[4]
sigz = PolParam[5]
epsilon = PolParam[6]
beta = PolParam[7]
gamma = PolParam[8]
F = PolParam[9]
b = PolParam[10]
xif = PolParam[11]
s = PolParam[12]
gsY = PolParam[13]

rho_A = PolParam[14]
rho_mu = PolParam[15]
rho_g = PolParam[16]
r_i = PolParam[17]
r_pi = PolParam[18]
r_y = PolParam[19]
psi = PolParam[20]

sig_A = PolParam[21]
sig_mu = PolParam[22]
sig_g = PolParam[23]
sig_m = PolParam[24]

rho_F = PolParam[25]
sig_F = PolParam[26]

#Distribution

Ez = exp(sigz^2/2)
CDF(x) = normcdf(0,sigz,log(max(x,0)))
PDF(x) = normpdf(0,sigz,log(max(x,0)))/max(x,0)
II(x) = exp(sigz^2/2)*normcdf(0,1,(sigz^2-log(max(x,0)))/sigz)
int(x) = II(x) - x*(1-CDF(x))

figure()

# i,pi,Y

subplot(2,3,1)
plot(0:Nperiods-1, IRF[Y,e_F,:],"k-",linewidth=1)
plot(0:Nperiods-1, IRF[i,e_F,:],"k--",linewidth=1)
plot(0:Nperiods-1, IRF[pi,e_F,:],"k:",linewidth=1)
legend(["\$Y\$","\$i\$","\$\\pi\$"])

# theta, mufsmu

subplot(2,3,2)
c_zs = PDF(ss[zs])*ss[zs]/(CDF(ss[zs])-CDF(ss[zf]))
c_zf = -(1-CDF(ss[zs]))*PDF(ss[zf])*ss[zf]/(1-CDF(ss[zf]))/(CDF(ss[zs])-CDF(ss[zf]))
plot(0:Nperiods-1, c_zs*IRF[zs,e_F,:]+c_zf*IRF[zf,e_F,:],"k--",linewidth=1)
plot(0:Nperiods-1, IRF[theta,e_F,:],"k-",linewidth=1)
legend(["\$\\frac{\\mu^f}{\\mu}\$","\$\\theta\$"])

# theta, mufsmu

subplot(2,3,3)
plot(0:Nperiods-1, IRF[np,e_F,:],"k-",linewidth=1)
plot(0:Nperiods-1, IRF[nf,e_F,:],"k--",linewidth=1)
legend(["\$n^p\$","\$n^f\$"])

# Thresholds

subplot(2,3,4)
plot(0:Nperiods-1, IRF[zp,e_F,:],"k-",linewidth=1)
plot(0:Nperiods-1, IRF[zf,e_F,:],"k--",linewidth=1)
plot(0:Nperiods-1, IRF[zs,e_F,:],"k:",linewidth=1)
legend(["\$z^p\$","\$z^f\$","\$z^*\$"])

# Job creation

subplot(2,3,5)
plot(0:Nperiods-1, IRF[v,e_F,:]-sigma*IRF[theta,e_F,:]-ss[zs]*PDF(ss[zs])*IRF[zs,e_F,:]/(1-CDF(ss[zs])),"k-",linewidth=1)
plot(0:Nperiods-1, IRF[v,e_F,:]-sigma*IRF[theta,e_F,:]+(ss[zs]*PDF(ss[zs])*IRF[zs,e_F,:] - ss[zf]*PDF(ss[zf])*IRF[zf,e_F,:])/(CDF(ss[zs])-CDF(ss[zf])),"k--",linewidth=1)
legend(["\$jc^p\$","\$jc^f\$"])

# Job destruction

subplot(2,3,6)
plot(0:Nperiods-1, vcat(0,IRF[np,e_F,1:Nperiods-1]) + (1-s)*PDF(ss[zp])*ss[zp]*IRF[zp,e_F,:]/(s + (1-s)*CDF(ss[zp])),"k-",linewidth=1)
plot(0:Nperiods-1, vcat(0,IRF[nf,e_F,1:Nperiods-1]),"k--",linewidth=1)
legend(["\$jd^p\$","\$jd^f\$"])

tight_layout()

cd(graph_dir)
savefig("IRF_F_perm.pdf")

# ============================================================================#
# ================================WEIGHTS=====================================#
# ============================================================================#

N = 100
grid_F = linspace(0.95*F,1.05*F,N)
grid_alphap = zeros(N)
for i in 1:N
    param[9] = grid_F[i]
    ss = steady_state_eval(param)
    alpha_p = m*ss[theta]^(-sigma)*(1-CDF(ss[zs]))*(II(ss[zs])/(1-CDF(ss[zs])) - ss[zp] - F)
    alpha_f = m*ss[theta]^(-sigma)*(CDF(ss[zs])-CDF(ss[zf]))*((II(ss[zf]) - II(ss[zs]))/(CDF(ss[zs])-CDF(ss[zf])) - ss[zf] )
    alpha_p /= (alpha_p+alpha_f)
    grid_alphap[i] = alpha_p
end

figure()
plot(linspace(-5,5,N),grid_alphap,"k-")
xlabel("\$F\$")
ylabel("\$\\psi_p \\mu^p / \\left( \\psi_p \\mu^p + \\psi_f \\mu^f \\right)\$")
tight_layout()
cd(graph_dir)
savefig("NKPC.pdf")
