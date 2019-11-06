# addprocs(4)

#=========================================================================#
#                           Environment
#=========================================================================#

@everywhere begin
    # working_dir = "E:\\Dropbox\\Ch2\\code\\23_02_2019_Euro"
    working_dir = "C:\\Users\\Normann\\Dropbox\\Ch2\\code\\23_02_2019_Euro"
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

    include("SMCfunctions_nog_habits.jl")
    # include("IskrevFunctions.jl")
    include("input\\InitialGuess_nog_habits.jl")
    ss = steady_state_eval(param)
    linearized_model(param) = linearized_model_ss(param,ss)
end

funcmod = linearized_model
@eval @everywhere funcmod = $funcmod

#Copies pour parallélisation

@eval @everywhere param=$param
@eval @everywhere indposest=$indposest
@eval @everywhere bound=$bound

#=========================================================================#
#                           Load data
#=========================================================================#

# data = DataFrame(load("input\\data.xlsx", "SHORT_LT!F1:I45"))
# Y = convert(Array{Float64,2},data)
# plot(Y)
# legend(["Y","pi","R","n"])

data = DataFrame(load("input\\data.xlsx", "LONG!A1:E95"))
T = size(data,1)
m = 4
h = 8
p = 4
Filtered,_ = HamiltonFilter(data[:,[:n]],h,p)
Filtered = convert(Array{Float64,2},Filtered)
Y = zeros(T-(p+h)+1,3)
Y[:,1] = Filtered
Y[:,2:3] = convert(Array{Float64,2},data[p+h:end,[:pi,:R]]) - mean(convert(Array{Float64,2},data[p+h:end,[:pi,:R]]))
figure()
plot(Y)
legend(["n","pi","R"])

@eval @everywhere Y = $Y

#=========================================================================#
#                       Set Parameters of Algorithm
#=========================================================================#

#SMC parameters

Nblock = 1
Nparam = 11
Npart = 1000
Nphi = 100
lambda = 2

#MH parameters

c = 0.5
acpt = 0.25
trgt = 0.25

#Tempering schedule

phi = ((0:(Nphi-1))/(Nphi-1)).^lambda

#Storing the results

drawsMat = zeros(Nphi, Npart, Nparam)
weightsMat   = zeros(Npart, Nphi)
constMat    = zeros(Nphi)
loglh   = zeros(Npart)
logpost = zeros(Npart)
nresamp = 0

cMat    = zeros(Nphi,1)
ESSMat  = zeros(Nphi,1)
acptMat = zeros(Nphi,1)
rsmpMat = zeros(Nphi,1)

@everywhere f(x1,x2) = Likelihoods!(x1,x2,param,indposest,Y,funcmod,bound,ss)
# f(x1,x2) = Likelihoods!(x1,x2,param,indposest,Y,funcmod,bound)
f(param[indposest],1)

#Conversion for parallelization

loglh = convert(SharedArray,loglh)
logpost = convert(SharedArray,logpost)
drawsMat = convert(SharedArray,drawsMat)
loglh_temp = SharedArray{Float64,1}(Npart)
logpost_temp = SharedArray{Float64,1}(Npart)
acpt_temp = SharedArray{Float64,1}(Npart)

#=========================================================================#
#             Initialize Algorithm: Draws from prior
#=========================================================================#

println("SMC starts....")

weightsMat[:,1] = 1/Npart
constMat[1] = sum(weightsMat[:,1])

drawsMat[1,:,:] = PriorDraws(drawsMat[1,:,:],f)

@sync @parallel for i in 1:Npart
# for i in 1:Npart
    logpost[i], loglh[i] = f(drawsMat[1,i,:],phi[1])
end

smctime   = tic()
totaltime = 0

println("SMC recursion starts...")

for i in 2:Nphi

    #-----------------------------------
    # (a) Correction
    #-----------------------------------

    incwt = exp.((phi[i]-phi[i-1])*loglh)
    weightsMat[:, i] = weightsMat[:, i-1].*incwt
    constMat[i]     = sum(weightsMat[:, i])
    weightsMat[:, i] /= constMat[i]

    #-----------------------------------
    # (b) Selection
    #-----------------------------------

    ESS = 1/sum(weightsMat[:, i].^2)

    if (ESS < 0.5*Npart)

        id = MultinomialResampling(weightsMat[:, i])
        drawsMat[i-1, :, :] = drawsMat[i-1, id, :]

        loglh              = loglh[id]
        logpost            = logpost[id]
        weightsMat[:, i]   = 1/Npart
        nresamp            = nresamp + 1
        rsmpMat[i]         = 1

    end


    #--------------------------------------------------------
    # (c) Mutation
    #--------------------------------------------------------

    c = c*(0.95 + 0.10*exp(16*(acpt-trgt))/(1 + exp(16*(acpt-trgt))))

    para      = drawsMat[i-1, :, :]
    wght      = repmat(weightsMat[:, i], 1, Nparam)

    mu        = sum(para.*wght,1)
    z         = (para - repmat(mu, Npart, 1))
    R       = (z.*wght)'*z

    Rdiag   = diagm(diag(R))
    Rchol   = chol(Hermitian(R))'
    Rchol2  = sqrt.(Rdiag)

    tune = Array{Any,1}(4)
    tune[1] = c
    tune[2] = R
    tune[3] = Nparam
    tune[4] = phi

    loglh = convert(SharedArray,loglh)
    logpost = convert(SharedArray,logpost)

    @sync @parallel for j in 1:Npart
    # for j in 1:Npart
        ind_para, ind_loglh, ind_post, ind_acpt = MutationRWMH(para[j,:], loglh[j], logpost[j], tune, i, f)
        drawsMat[i,j,:] = ind_para
        loglh_temp[j]       = ind_loglh
        logpost_temp[j]     = ind_post
        acpt_temp[j] = ind_acpt
    end

    loglh = loglh_temp
    logpost = logpost_temp

    acpt = mean(acpt_temp)

    cMat[i,:]    = c
    ESSMat[i,:]  = ESS
    acptMat[i,:] = acpt

    if mod(i, 1) == 0

        para = drawsMat[i, :, :]
        wght = repmat(weightsMat[:, i], 1, Nparam)

        mu  = sum(para.*wght,1)

        sig = sum((para - repmat(mu, Npart, 1)).^2 .*wght,1)
        sig = (sqrt.(sig))

        totaltime = totaltime + toc()
        avgtime   = totaltime/i
        remtime   = avgtime*(Nphi-i)

        print("-----------------------------------------------\n")
        print(" Iteration = $i / $Nphi \n")
        print("-----------------------------------------------\n")
        print(" phi  = $(phi[i]) \n")
        print("-----------------------------------------------\n")
        print("  c    = $c\n")
        print("  acpt = $acpt\n")
        print("  ESS  = $ESS  ($nresamp total resamples.)\n")
        print("-----------------------------------------------\n")
        print("  time elapsed   = $totaltime\n")
        print("  time average   = $avgtime\n")
        print("  time remained  = $remtime\n")
        print("-----------------------------------------------\n")
        print("para      mean    std\n")
        print("------    ----    ----\n")
        for k in 1:Nparam
            print("$(posest[k])     $(mu[k])    $(sig[k])\n")
        end

        tic()
    end

end

print("-----------------------------------------------\n")
println("logML = $(sum(log.(constMat)))")
print("-----------------------------------------------\n")

# JLD.save("LT_SHORT.jld","theta",drawsMat[Nphi,:,:],"weights",weightsMat[:,Nphi],"loglh",loglh,"logpost",logpost,"constMat",constMat)
JLD.save("LONG_nog_habits.jld","theta",drawsMat[Nphi,:,:],"weights",weightsMat[:,Nphi],"loglh",loglh,"logpost",logpost,"constMat",constMat)
