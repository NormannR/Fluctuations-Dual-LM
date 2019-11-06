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
include("input\\InitialGuess.jl")
ss = steady_state_eval(param)
F0 = param[9]
param[9] = 1.01*F0
ss_pol = steady_state_eval(param)

cd(working_dir)
var = JLD.load("LT_SHORT.jld")
theta = var["theta"]
weights = var["weights"]
wght = repmat(var["weights"], 1, Nparam)

param[9] = F0

param[indposest]  = reshape(sum(theta.*wght,1)',Nparam)

PolParam = copy(param)
PolParam[9] = 1.01*F0

linearized_model(param) = linearized_model_ss(param,ss)
funcmod = linearized_model
linearized_model_pol(param) = linearized_model_ss(PolParam,ss_pol)
funcmod_pol = linearized_model_pol

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

GG,RR,SDX,_,_ = funcmod_pol(PolParam)

N = 1000
T = 500
Tburn = 100
sim_pol = zeros(N,T,21)

for n in 1:N
    for t in 1:T-1
        sim_pol[n,t+1,:] = GG*sim_pol[n,t,:] + RR*SDX*e[n,t,:]
    end
end

#==============================================================================#
#================================Moments=======================================#
#==============================================================================#

std_pol = 4*mean(std(sim_pol[:,Tburn:end,10],2))
std0 = 4*mean(std(sim[:,Tburn:end,10],2))
