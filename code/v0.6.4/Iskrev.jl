working_dir = "E:\\Dropbox\\Ch2\\code\\Euro"
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

include("SMCfunctions.jl")
include("IskrevFunctions.jl")
include("input\\InitialGuess.jl")

Nparam = size(indposest,1)
NY = 15
NX = 4

var = JLD.load("LT_SHORT.jld")
theta = var["theta"]
wght = repmat(var["weights"], 1, Nparam)
mu  = reshape(sum(theta.*wght,1)',Nparam)
param[indposest] = mu

ss = steady_state_eval(param)
linearized_model(param) = linearized_model_ss(param,ss)
funcmod = linearized_model

data = DataFrame(load("input\\data.xlsx", "SHORT_LT!F1:I45"))
d = data[:,[:Y,:pi,:R,:n]]
Y = convert(Array{Float64,2},d)

#Derivatives

Nparam = size(indposest,1)
T = size(Y,1)

J1 = zeros(NX*(NX+1)*T/2 ,Nparam)
J2 = zeros(NY^2 + NY*(NY+1)/2,Nparam)

dGAM0 = Vector{Array{Float64,2}}(Nparam)
dGAM1 = Vector{Array{Float64,2}}(Nparam)
dGAM2 = Vector{Array{Float64,2}}(Nparam)
dGAM3 = Vector{Array{Float64,2}}(Nparam)
dA = Vector{Array{Float64,2}}(Nparam)
dB = Vector{Array{Float64,2}}(Nparam)
dO = Vector{Array{Float64,2}}(Nparam)
dSIG0 = Vector{Array{Float64,2}}(Nparam)
dSIG = Array{Array{Float64,2},2}(T,Nparam)

A,B,SDX,C,eu=funcmod(param)

id = Iskrev!(A,B,SDX,C,param,indposest,ss,Y,dGAM0,dGAM1,dGAM2,dGAM3,dA,dB,dO,dSIG0,dSIG,J1,J2)

alpha = 1
beta = 1

bbeta = Beta(alpha,beta)

mu = 1.
sig = 1.

alpha = 2. + (mu/sig)^2
beta = mu*(alpha-1)
IG = InverseGamma(alpha,beta)

Npart = 10000
c = 0
for n in 1:Npart
	# println("n : $n")
	valid = false
	while !valid
		param[indposest[[1,2,3,4,7]]] = rand(bbeta,5)
		param[indposest[5]] = rand(Truncated(Normal(1,100),0,Inf))
		param[indposest[6]] = rand(Truncated(Normal(0,100),0,Inf))
		param[indposest[8:11]] = rand(IG,4)
		try

			A,B,SDX,C,eu=funcmod(param)
			id = Iskrev!(A,B,SDX,C,param,indposest,ss,Y,dGAM0,dGAM1,dGAM2,dGAM3,dA,dB,dO,dSIG0,dSIG,J1,J2)
			valid = true
			if id != [true,true]
				c += 1
			end
		end


	end
end


# function tau(theta,param,indposest)
#
# 	param[indposest] = theta
# 	A,B,SDX,C,eu=funcmod(param)
# 	NY = 15
# 	A = A[1:NY,1:NY]
# 	B = B[1:NY,:]
# 	B = B*SDX
# 	C = C[:,1:NY]
# 	O = B*B'
#
# 	return vcat(vec(A),vech(O))
#
# end
#
# function jacobian(f,x,h)
#
# 	n = size(x,1)
# 	m = size(f(x),1)
# 	J = zeros(m,n)
#
# 	for j in 1:n
# 		xh = copy(x)
# 		xh[j] += h
# 		J[:,j] = (f(xh)-f(x))./h
# 	end
#
# 	return J
#
# end
#
# J = jacobian(theta-> tau(theta,param,indposest), param[indposest],1e-12)
