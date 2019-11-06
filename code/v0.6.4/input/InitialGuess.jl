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

param = zeros(24)

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

param[14] = 0.9
param[15] = 0.9
param[16] = 0.9
param[17] = 0.7
param[18] = 3.15
param[19] = 0.02
param[20] = 0.88

param[21] = 1.
param[22] = 1.
param[23] = 1.
param[24] = 1.

posest = ["rho_A","rho_mu","rho_g","r_i","r_pi","r_y","psi","sig_A","sig_mu","sig_g","sig_m"]

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
rho_g = 16
r_i = 17
r_pi = 18
r_y = 19
psi = 20

sig_A = 21
sig_mu = 22
sig_g = 23
sig_m = 24

indposest = rho_A:sig_m

bound =[0. 1. ;
        0. 1. ;
        0. 1. ;
        0. 1. ;
        1. 1e10;
        0. 1e10;
        0. 1. ;
        0. 1e10;
        0. 1e10;
        0. 1e10;
        0. 1e10]

println("Loaded initial parameter guess")
