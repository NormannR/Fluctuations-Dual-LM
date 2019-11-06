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

include("SMCfunctions.jl")
include("IskrevFunctions.jl")
include("input\\InitialGuess.jl")
ss = steady_state_eval(param)
funcmod(param) = linearized_model_ss(param,ss)

Nparam = 11
NY = 15
NX = 4

var = JLD.load("LT_SHORT.jld")
theta = var["theta"]
wght = repmat(var["weights"], 1, Nparam)
mu  = reshape(sum(theta.*wght,1)',Nparam)

param[indposest] = mu

GG,RR,SDX,ZZ,eu=funcmod(param)
