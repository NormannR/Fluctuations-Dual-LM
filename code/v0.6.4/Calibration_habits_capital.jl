cd("E:\\Dropbox\\Ch2\\code\\23_02_2019_Euro\\input")

using NLsolve.nlsolve
using StatsFuns.normcdf
using JLD

#INITIAL PARAMETERS AND TARGETS

eta = 0.6
sigma = 0.6
beta = 0.99
nfsn = 0.15
ssxip = 0.7
xip = 0.03

gsY = 0.2
alpha = 0.33
delta = 0.025
rk = 1/beta - (1-delta)

epsilon = 6
mu = epsilon/(epsilon-1)

mc = 1/mu
phi = (mc*alpha^alpha*(1-alpha)^(1-alpha)/(rk^alpha))^(1/(1-alpha))
ksX = (rk/alpha/mc)^(1/(alpha-1))
Isk = delta

qk = 1
nu = 1

mufsmu = 0.7
rhoF = 4/9
q = 0.7
epsilon = 6
sigz = 0.2
u = 1-0.67

Ez = exp(sigz^2/2)
G(x) = normcdf(0,sigz,log(max(x,0)))
II(x) = exp(sigz^2/2)*normcdf(0,1,(sigz^2-log(max(x,0)))/sigz)
int(x) = II(x) - x*(1-G(x))

function f!(F,z,xip,ssxip,G)
	F[1] = xip-ssxip*xip-(1-ssxip*xip)*G(z[1])
end

g!(F,x) = f!(F,x,xip,ssxip,G)

F = zeros(1)
x0 = [ Ez ]
results = nlsolve(g!, x0, show_trace = true, method = :trust_region, ftol = eps())
zp = results.zero[1]

s = ssxip*xip
xif = mufsmu*(1-nfsn)*xip/(1-mufsmu)/nfsn

n = 1-u
nf = nfsn*n
np = n-nf

e = u+xip*np+xif*nf
mup = xip*np/e
muf = xif*nf/e

param = Dict{String,Float64}()
param["mup"] = mup
param["muf"] = muf
param["zp"] = zp
param["s"] = s
param["eta"] = eta
param["beta"] = beta
param["xip"] = xip
param["xif"] = xif
param["mufsmu"] = mufsmu
param["rhoF"] = rhoF
param["nfsn"] = nfsn
param["q"] = q
param["ssxip"] = ssxip
param["phi"] = phi
param["sigz"] = sigz

function system_cal!(R,x,param)

	mup = param["mup"]
	muf = param["muf"]
	zp = param["zp"]
	s = param["s"]
	eta = param["eta"]
	beta = param["beta"]
	xip = param["xip"]
	xif = param["xif"]
	mufsmu = param["mufsmu"]
	rhoF = param["rhoF"]
	nfsn = param["nfsn"]
	q = param["q"]
	ssxip = param["ssxip"]
	phi = param["phi"]
	sigz = param["sigz"]

	F = x[1]
    b = x[2]
    zf = x[3]
    rho = x[4]

	Ez = exp(sigz^2/2)
    G(x) = normcdf(0,sigz,log(max(x,0)))
    II(x) = exp(sigz^2/2)*normcdf(0,1,(sigz^2-log(max(x,0)))/sigz)
    int(x) = II(x) - x*(1-G(x))

    s = xip*ssxip

    zc = zp+F
    zs = (zc-rho*zf)/(1-rho)
    gamma_theta = (phi*zp + beta*(1-s)*phi*int(zp) + (1-beta*(1-s))*phi*F - b)*(1-eta)/(1-xip)/beta/eta

    R[1] =  phi*F/rhoF - eta*phi*((1-xip)*II(zp)/(1-G(zp)) + xip*II(zs)/(1-G(zs)) ) -
            eta*((1-beta*(1-s))*phi*F + beta*(1-xip)*gamma_theta) - (1-eta)*b
    R[2] = G(zs)-G(zf) - mufsmu*(1-G(zf))
    R[3] = gamma_theta - phi*(1-eta)*( mup*( II(zs)/(1-G(zs)) - zc ) + rho*muf*( (II(zf)-II(zs))/(G(zs)-G(zf)) - zf) )
    R[4] = rho*phi*( zf + beta*(1-xif)*(Ez - zf) ) - b - beta*(1-xif)*eta*gamma_theta/(1-eta)

end

f!(R,x)=system_cal!(R,x,param)
R = ones(4)
x0 = [0.46188,0.8409,1.0806,0.9741]
results = nlsolve(f!, x0, show_trace = true, method = :trust_region, ftol = eps())
x = results.zero
F = x[1]
b = x[2]
zf = x[3]
rho = x[4]

zc = zp+F
zs = (zc-rho*zf)/(1-rho)
gamma_theta = (phi*zp + beta*(1-s)*phi*int(zp) + (1-beta*(1-s))*phi*F - b)*(1-eta)/(1-xip)/beta/eta
gamma = q*(1-eta)*phi*( int(zs) + rho*(int(zf)-int(zs)) )
theta = gamma_theta/gamma

m = q*theta^sigma

v = theta*e

wp_bar =    eta*phi*((1-xip)*II(zp)/(1-G(zp)) + xip*II(zs)/(1-G(zs)) ) +
            eta*((1-beta*(1-s))*phi*F + beta*(1-xip)*gamma_theta) + (1-eta)*b

wf_bar =    eta*phi*(rho*(1-xif)*Ez + rho*xif*(II(zf) - II(zs))/(G(zs)-G(zf)) ) +
            eta*beta*(1-xif)*gamma_theta + (1-eta)*b

w_bar =     (wp_bar*np+wf_bar*nf)/n

p = theta*q

X = np*((1-xip)*II(zp)/(1-G(zp)) + xip*II(zs)/(1-G(zs))) + nf*rho*((1-xif)*Ez
 	+ xif*(II(zf)-II(zs))/(G(zs)-G(zf)) )

k = ksX*X
I = Isk*k

Y = k^alpha*X^(1-alpha)

g = gsY*Y
c = Y-g-gamma*v-I

save("parameters_capital.jld", "rho", rho, "sigma", sigma, "eta", eta, "m", m, "sigma_z",
	sigz, "epsilon", epsilon, "beta", beta, "gamma", gamma, "F", F, "b", b, "xif",
	xif, "s", s, "gsY", gsY,"alpha",alpha,"delta",delta)
