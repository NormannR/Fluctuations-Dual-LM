#I) Reading initial guess for PolParameters

var = load("input\\PolParameters.jld")
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

PolParam = zeros(24)

PolParam[1] = rho
PolParam[2] = sigma
PolParam[3] = eta
PolParam[4] = m
PolParam[5] = sigz
PolParam[6] = epsilon
PolParam[7] = beta
PolParam[8] = gamma
PolParam[9] = F
PolParam[10] = b
PolParam[11] = xif
PolParam[12] = s
PolParam[13] = gsY

PolParam[14] = 0.9
PolParam[15] = 0.9
PolParam[16] = 0.9
PolParam[17] = 0.7
PolParam[18] = 3.15
PolParam[19] = 0.02
PolParam[20] = 0.88

PolParam[21] = 1.
PolParam[22] = 1.
PolParam[23] = 1.
PolParam[24] = 1.

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

println("Loaded initial PolParameter guess")
