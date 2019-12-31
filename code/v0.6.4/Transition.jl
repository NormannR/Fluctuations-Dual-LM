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

psi = param[20]

T = 10
states = zeros(5,T)
states[1,:] = ss_pol[3]
states[2,:] = ss_pol[4]
states[3,:] = ss_pol[1]
states[4,:] = 1.
states[5,:] = 1.

i = 1
c = 2
np = 3
nf = 4
zp = 5
zf = 6
zs = 7
theta = 8
phi = 9
P = 10
Y = 11
v = 12
Delta = 13

pol = zeros(13,T)

tol = 1e-7
dist = 1e10
iter = 1
while dist > tol
    println("$iter : $dist")

    for t in 1:T



    end


end

function IRF_Solve(R,x,param,states,next)

    #Parameters

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
    rho_R = param[17]
    rho_pi = param[18]
    rho_y = param[19]
    psi = param[20]

    sig_A = param[21]
    sig_mu = param[22]
    sig_g = param[23]
    sig_m = param[24]

    Ez = exp(sigz^2/2)
    G(x) = normcdf(0,sigz,log(max(x,0)))
    II(x) = exp(sigz^2/2)*normcdf(0,1,(sigz^2-log(max(x,0)))/sigz)
    int(x) = II(x) - x*(1-G(x))

    # Variables

    c = x[1]
    zp = x[2]
    zf = x[3]
    theta = x[4]
    P = x[6]

    E_c = next[1]
    E_zp = next[2]
    E_zf = next[3]
    E_theta = next[4]
    E_P = next[6]

    # States

    np_m = states[1]
    nf_m = states[2]
    r_m = states[3]
    P_m = states[4]
    Delta_m = states[5]

    # Intermediate calculations

    xip = s+(1-s)*G(zp)

    u_m = 1-np_m-nf_m
    e = u_m + xip*np_m + xif*nf_m
    v = theta*e
    q = m*theta^(-sigma)
    p = theta*q
    Y = (c+gamma*v)/(1-gsY)
    Delta = np_m*(1-s)*II(zp) + v*q*II(zs) + nf_m*rho*(1-xif)*Ez + rho*v*q*(II(zf)-II(zs))
    Delta /= Y

    zc = zp+F
    zs = (zc - rho*zf)/(1-rho)

    phi = gamma/(1-eta)/q/(int(zs) + rho*(int(zf)-int(zs)))
    E_phi = gamma/(1-eta)/q/(int(E_zs) + rho*(int(E_zf)-int(E_zs)))

    pi = P/P_m
    E_pi = E_P/P

    r = ss[1]*exp(rho_R*log(r_m/ss[1]) + (1-rho_R)*(rho_Y*log(Y/ss[11])+rho_pi*log(E_pi)) )
    P_star = ((Delta - psi*pi^epsilon*Delta_m)*P^(-epsilon)/(1-psi))^(-1/epsilon)

    # Equations

    kappa = (1-beta*psi)*(1-psi)/psi
    bbeta = beta*c/E_c

    R[1] = phi*zp + phi*F - bbeta*(1-s)*E_phi*F + bbeta*(1-s)*E_phi*int(E_zp) - b - bbeta*(1-s)*(1-G(E_zp))*eta*gamma*E_theta/(1-eta)
    R[2] = rho*phi*zf + (1-xif)*rho*bbeta*E_phi*(Ez - E_zf) - b - bbeta*(1-xif)*eta*gamma*E_theta/(1-eta)
    R[3] = 1 - beta*r*c/E_c/E_pi
    R[4] = P^(1-epsilon) - (psi*P_m^(1-epsilon) + (1-psi)*P_star^(1-epsilon))
    R[5] = log(pi) - beta*log(E_pi) - kappa*log(phi/ss[9])
    
end
