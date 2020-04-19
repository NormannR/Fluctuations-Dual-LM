module VAR

using LinearAlgebra
using StatsBase

export VAR_estimation, irf_var, sim_irf_var

function VAR_estimation(y,p)

    #Following notations in Hamilton (1994) "Time series analysis" Ch 11

    n,T = size(y)

    X = zeros(n*p+1,T-p)
    X[1,:] .= 1
    for i in 1:p
        X[2+(i-1)*n:1+i*n,:] .= y[:,p-(i-1):T-i]
    end
    A = zeros(n,n*p+1)
    B = zeros(n*p+1,n*p+1)

    Y = y[:,p+1:end]
    for t in 1:T-p
        A .= A + Y[:,t]*X[:,t]'
        B .= B + X[:,t]*X[:,t]'
    end
    Π = A*inv(B)
    ϵ = Y - Π*X
    Ω = zeros(n,n)
    for t in 1:T-p
        Ω .= Ω + ϵ[:,t]*ϵ[:,t]'
    end
    Ω .= Ω/(T-p)

    return (Π,Ω,ϵ)

end

function irf_var(Y, p; T=50)

    Π,Ω,_ = VAR_estimation(Y',p)
    P = cholesky(Ω).L
    n = size(Y,2)
    IRF = zeros(n*p,n*p,T)
    IRF[:,:,1] .= P*I(n)

    for t in 2:T
        IRF[:,:,t] = Π[:,2:end]*IRF[:,:,t-1]
    end

    return IRF
end

function sim_irf_var(Y, p ; N=10000, α=0.05, T_irf=24)

    T,n = size(Y)
    Π, Ω, ϵ = VAR_estimation(Y', p)
    u = ϵ[:,rand(1:T-p,T,N)]
    π = zeros(n*p,n*p+1,N)
    ω = zeros(n,n,N)
    y = zeros(n,T,N)
    IRF_sim = zeros(n,n,T_irf,N)
    y[:,1,:] .= u[:,1,:]

    for k in 1:N
        for t in 2:T
            y[:,t,k] .= Π*vcat(1,y[:,t-1,k]) + u[:,t,k]
        end
        π[:,:,k], ω[:,:,k], _ = VAR_estimation(y[:,:,k], p)
        P = cholesky(ω[:,:,k]).L
        IRF_sim[:,:,1,k] .= P*I(n)
        for t in 2:T_irf
            IRF_sim[:,:,t,k] = π[:,2:end,k]*IRF_sim[:,:,t-1,k]
        end
    end

    IRF_sim_q = zeros(n,n,T_irf,3)
    for k in 1:n
        for l in 1:n
            for t in 1:T_irf
                IRF_sim_q[k,l,t,:] .= quantile(IRF_sim[k,l,t,:],[0.5, α/2, 1-α/2])
            end
        end
    end

    IRF_sim_q[:,:,:,2] .= IRF_sim_q[:,:,:,1] - IRF_sim_q[:,:,:,2]
    IRF_sim_q[:,:,:,3] .= IRF_sim_q[:,:,:,3] - IRF_sim_q[:,:,:,1]

    return IRF_sim_q

end

end
