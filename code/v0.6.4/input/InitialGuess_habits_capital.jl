#I) Reading initial guess for parameters

var = load("input\\parameters_capital.jld")
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
alpha = var["alpha"]
delta = var["delta"]

param = zeros(34)

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
param[14] = alpha
param[15] = delta

param[16] = 0.5
param[17] = 0.5
param[18] = 0.5
param[19] = 1.7
param[20] = 0.5
param[21] = 0.7

param[22] = 1.
param[23] = 1.
param[24] = 1.

param[25] = 0.9
param[26] = 1.5

param[27] = 0.5 #psi_nu
param[28] = 2.5 #eta_k

param[29] = 0.5
param[30] = 1.
param[31] = 0.5
param[32] = 1.
param[33] = 0.5
param[34] = 1.

posest = ["rho_A","rho_mu","r_i","r_pi","r_y","psi","sig_A","sig_mu","sig_m",
        "h","sig_c","psi_nu","eta_k","rho_g","sig_g","rho_b","sig_b","rho_i","sig_i"]

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
alpha = 14
delta = 15

rho_A = 16
rho_mu = 17
r_i = 18
r_pi = 19
r_y = 20
psi = 21

sig_A = 22
sig_mu = 23
sig_m = 24

h = 25
sig_c = 26

psi_nu = 27
eta_k = 28

rho_g = 29
sig_g = 30
rho_b = 31
sig_b = 32
rho_i = 33
sig_i = 34

indposest = rho_A:sig_i

bound =[0. 1. ;
        0. 1. ;
        0. 1. ;
        1. 1e10;
        0. 1e10;
        0. 1. ;
        0. 1e10;
        0. 1e10;
        0. 1e10;
        0. 1.;
        0. 1e10;
        0. 1.;
        0. 1e10;
        0. 1.;
        0. 1e10;
        0. 1.;
        0. 1e10;
        0. 1.;
        0. 1e10]

println("Loaded initial parameter guess")
