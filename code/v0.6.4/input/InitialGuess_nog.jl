#I) Reading initial guess for parameters

var = load("input\\parameters.jld")
b = var["b"]
sigma = var["sigma"]
xif = var["xif"]
s = var["s"]
gsY = var["gsY"]
eta = var["eta"]
sigz = var["sigma_z"]
rho = var["rho"]
beta = var["beta"]
m = var["m"]
epsilon = var["epsilon"]
gamma = var["gamma"]
F = var["F"]

param = zeros(22)

param[1] = rho
param[2] = sigma
param[3] = eta
param[4] = m
param[5] = sigz
param[6] = epsilon
param[7] = beta
param[8] = gamma
param[9] = F
param[10] = b
param[11] = xif
param[12] = s
param[13] = gsY

param[14] = 0.5
param[15] = 0.5
param[16] = 0.5
param[17] = 1.7
param[18] = 0.5
param[19] = 0.7

param[20] = 1.
param[21] = 1.
param[22] = 1.

posest = ["rho_A","rho_mu","r_i","r_pi","r_y","psi","sig_A","sig_mu","sig_m"]

#Estimated Parameters

rho = 1
sigma = 2
eta = 3
m = 4
sigz = 5
epsilon = 6
beta = 7
gamma = 8
F = 9
b = 10
xif = 11
s = 12
gsY = 13

rho_A = 14
rho_mu = 15
r_i = 16
r_pi = 17
r_y = 18
psi = 19

sig_A = 20
sig_mu = 21
sig_m = 22

indposest = rho_A:sig_m

bound =[0. 1. ;
        0. 1. ;
        0. 1. ;
        1. 1e10;
        0. 1e10;
        0. 1. ;
        0. 1e10;
        0. 1e10;
        0. 1e10]

println("Loaded initial parameter guess")
