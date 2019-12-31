using LinearAlgebra
using QuantEcon

function solve_sims!(RR,GG,eu,Γ₀,Γ₁,Ψ,Π; tol = 1e-8)

    F = schur(Γ₀,Γ₁)

    if any((abs.(F.α) .< eps()) .& (abs.(F.β) .< eps()))
        println("Coincident zeros. Indeterminacy and/or non-existence")
        return (nothing,nothing,nothing,eu)
    end

    select = abs.(F.α./F.β) .> 1
    ordschur!(F, select)

    ns = count(select)
    n = size(Γ₀,1)
    k = n-ns

    Q1 = transpose(F.Q)[1:ns,:]
    Q2 = transpose(F.Q)[ns+1:end,:]

    F_Q2Π = svd(Q2*Π)
    F_Q2Ψ = svd(Q2*Ψ)
    ind_Q2Π = F_Q2Π.S .> tol*n
    ind_Q2Ψ = F_Q2Ψ.S .> tol*n

    if isempty(ind_Q2Π)
        eu[1] = 1
    else
        eu[1] = norm(F_Q2Ψ.U[:,ind_Q2Ψ] - F_Q2Π.U[:,ind_Q2Π]*transpose(F_Q2Π.U[:,ind_Q2Π])*F_Q2Ψ.U[:,ind_Q2Ψ]) < n*tol
    end

    if eu[1] != 1
        println("Non-existence")
        return (nothing,nothing,nothing,eu)
    end

    F_Q1Π = svd(Q1*Π)
    ind_Q1Π = F_Q1Π.S .> n*tol

    if isempty(ind_Q1Π)
        eu[2] = 1
    else
        eu[2] = norm(transpose(F_Q1Π.Vt[ind_Q1Π,:]) -transpose(F_Q2Π.Vt[ind_Q2Π,:])*F_Q2Π.Vt[ind_Q2Π,:]*transpose(F_Q1Π.Vt[ind_Q1Π,:])) < n*tol
    end

    if eu[2] != 1
        println("Indeterminacy. ", size(F_Q1Π.Vt[ind_Q1Π,:],1) - size(F_Q2Ψ.Vt[ind_Q2Ψ,:],1), " loose endogenous errors")
        return (nothing,nothing,nothing,eu)
    end

    Φ = F_Q1Π.U[:,ind_Q1Π]*Diagonal(F_Q1Π.S[ind_Q1Π])*F_Q1Π.Vt[ind_Q1Π,:]*transpose(F_Q2Π.Vt[ind_Q2Π,:])*Diagonal(1 ./ F_Q2Π.S[ind_Q2Π])*transpose(F_Q2Π.U[:,ind_Q2Π])

    invΛ11 = inv(F.S[1:ns,1:ns])
    Λ12 = F.S[1:ns,ns+1:end]
    Λ22 = F.S[ns+1:end,ns+1:end]

    Ω11 = F.T[1:ns,1:ns]
    Ω12 = F.T[1:ns,ns+1:end]
    Ω22 = F.T[ns+1:end,ns+1:end]

    H = zeros(n,n)
    H[1:ns,1:ns] .= invΛ11
    H[1:ns,ns+1:end] .= -invΛ11*(Λ12 - Φ*Λ22)
    H[ns+1:end,ns+1:end] .= I(n-ns)
    H = F.Z*H

    GG .= F.Z[:,1:ns]*invΛ11*hcat(Ω11, Ω12 - Φ*Ω22)*transpose(F.Z)
    topΘ = Q1-Φ*Q2
    RR .= H*vcat(topΘ, zeros(n-ns,n))*Ψ

end

function logprior(paramest)

    prior = zeros(length(paramest))

	α0 = 1
	α1 = 1

    prior[1] = logpdf(Beta(α0,α1),paramest[1])
    prior[2] = logpdf(Beta(α0,α1),paramest[2])
    prior[3] = logpdf(Beta(α0,α1),paramest[3])
    prior[4] = logpdf(Beta(α0,α1),paramest[4])

	prior[5] = logpdf(Truncated(Normal(1.5,0.75),1,Inf),paramest[5])
	prior[6] = logpdf(Truncated(Normal(0.5,0.75),0,Inf),paramest[6])

	mu = 0.7
	sig = 0.05
	α0 = mu^2*(1-mu)/(sig)^2 - mu
	α1 = α0*(1-mu)/mu

	prior[7] = logpdf(Beta(α0,α1),paramest[7])

	mu = 0.15
	sig = 1.

	α0 = 2. + (mu/sig)^2
	α1 = mu*(α0-1)

    prior[8] = logpdf(InverseGamma(α0,α1),paramest[8])
    prior[9] = logpdf(InverseGamma(α0,α1),paramest[9])
    prior[10] = logpdf(InverseGamma(α0,α1),paramest[10])
    prior[11] = logpdf(InverseGamma(α0,α1),paramest[11])

    if any(isnan,prior) | any(isinf,prior)
        flag_ok=0
        lprior=NaN
    else
        flag_ok=1
        lprior=sum(prior)
    end

    return (lprior,flag_ok)

end

function Likelihoods!(x,l,Y,bound,param,Γ₀,Γ₁,Ψ,Π,SDX,GG,RR,ZZ,eu)

    param .= x
    outbound = (param .< bound[:,1]) .| (param .> bound[:,2])

    if any(outbound)

        lpost = -Inf
        lY = -Inf
        lprior = -Inf
		go_on = false

    else

        set_lin_mod!(Γ₀,Γ₁,Ψ,Π,SDX,param)
		solve_sims!(RR,GG,eu,Γ₀,Γ₁,Ψ,Π)

    	if (eu[2]!=1)

			lpost = -Inf
	        lY = -Inf
	        lprior = -Inf
			go_on = false

        else

            # Initialize Kalman filter
            T,nn = size(Y)
        	ss = size(GG,1)
        	MM = RR*(SDX')
            pshat = solve_discrete_lyapunov(GG, MM*(MM'))
        	shat = zeros(ss,1)
            lht = zeros(T)
			go_on = true

			i = 1
			while (i<T+1) & (go_on)
				shat,pshat,lht[i],_,go_on = kffast(Y[i,:],ZZ,shat,pshat,GG,MM)
				i += 1
			end

			if go_on

	            lY = -((T*nn*0.5)*(log(2*pi))) + sum(lht)
	            lprior,_ = logprior(param)
	            lpost = l*lY + lprior

			else

				lpost = -Inf
				lY = -Inf
				lprior = -Inf

			end

        end

    end

    return (lpost,lY,lprior,go_on)

end

function kffast(y,Z,s,P,T,R)

	#Forecasting
	fors = T*s
	forP = T*P*T'+R*R'
	fory = Z*fors
	forV = Z*forP*Z'

	#Updating
	C = cholesky(Hermitian(forV); check = false)
	if issuccess(C)

		z = C.L\(y-fory)
		x = C.U\z
		M = forP'*Z'
		sqrtinvforV = inv(C.L)
		invforV = sqrtinvforV'*sqrtinvforV
		ups = fors + M*x
		upP = forP - M*invforV*M'

		#log-Likelihood
		lh=-.5*(y - fory)'*invforV*(y - fory) .- sum(log.(diag(C.L)))

		return (ups,upP,lh[1],fory,true)

	else

		return (fors, forP,-Inf,fory,false)

	end

end

function PriorDraws(priorMat::Array{Float64,2},f)

    Npart = size(priorMat,1)

	α0 = 1
	α1 = 1

    B = Beta(α0,α1)

	mu = 0.7
	sig = 0.05
	α0 = mu^2*(1-mu)/(sig)^2 - mu
	α1 = α0*(1-mu)/mu
	B_ψ = Beta(α0,α1)

	mu = 0.15
	sig = 1.

	α0 = 2. + (mu/sig)^2
	α1 = mu*(α0-1)
	IG = InverseGamma(α0,α1)

    for n in 1:Npart
        valid = false
        while !valid

			priorMat[n,[1,2,3,4]] = rand(B,4)
			priorMat[n,7] = rand(B_ψ)
            priorMat[n,5] = rand(Truncated(Normal(1.5,0.75),1,Inf))
			priorMat[n,6] = rand(Truncated(Normal(0.5,0.75),0,Inf))
			priorMat[n,8:11] = rand(IG,4)

			valid = !isinf(f(priorMat[n,:],0)[3])

        end
    end

    return priorMat

end

function MultinomialResampling!(id,W,Npart)

    U = rand(Npart)
    CumW = [sum(W[1:i]) for i in 1:Npart]

    for i in 1:Npart
        id[i] = findfirst(x->(U[i]<x),CumW)
    end

end

function MutationRWMH(p0, l0, post0, tune, i, f)

    c = tune[1]
    R = tune[2]
    Nparam = tune[3]
    phi = tune[4]

    valid = false
	px = zeros(Nparam)
	postx = zeros(Nparam)
	lx = -Inf

    while !valid
		C = cholesky(Hermitian(R); check = false)
		flag_chol = issuccess(C)
		if flag_chol
        	px = p0 + c*transpose(C.L)*randn(Nparam)
            postx, lx, _, valid = f(px, phi[i])
        end
    end

    post0 = post0+(phi[i]-phi[i-1])*l0

    alp = exp(postx - post0)
    u = rand()

    if u < alp
        ind_para   = px
        ind_loglh  = lx
        ind_post   = postx
        ind_acpt   = 1
    else
        ind_para   = p0
        ind_loglh  = l0
        ind_post   = post0
        ind_acpt   = 0
    end

    return (ind_para, ind_loglh, ind_post, ind_acpt)

end

function SMCInit!(weightsMat,constMat,drawsMat,logpost,loglh,Npart,phi,f)
	println("SMC starts....")
    weightsMat[:,1] .= 1/Npart
    constMat[1] = sum(weightsMat[:,1])

    drawsMat[1,:,:] .= PriorDraws(drawsMat[1,:,:],f)

    for i in 1:Npart
        logpost[i], loglh[i] = f(drawsMat[1,i,:],phi[1])
    end

end

function SMCIter!(c, acpt, weightsMat, constMat, id, drawsMat, logpost, loglh, nresamp, tune, cMat, ESSMat, acptMat, rsmpMat, acpt_part, Nphi, Npart, Nparam, phi,f, trgt)

    println("SMC recursion starts...")

	@time begin

    for i in 2:Nphi



        #-----------------------------------
        # (a) Correction
        #-----------------------------------

         @views weightsMat[:, i] .= weightsMat[:, i-1].*exp.((phi[i]-phi[i-1]).*loglh)
        constMat[i]     = sum(weightsMat[:, i])
         @views weightsMat[:, i] .=  weightsMat[:, i] / constMat[i]

        #-----------------------------------
        # (b) Selection
        #-----------------------------------

        ESS = 1/sum(weightsMat[:, i].^2)

        if (ESS < 0.5*Npart)

            MultinomialResampling!(id,weightsMat[:, i],Npart)
             @views drawsMat[i-1, :, :] .= drawsMat[i-1, id, :]

            loglh              = loglh[id]
            logpost            = logpost[id]
            weightsMat[:, i]   .= 1/Npart
            nresamp            = nresamp + 1
            rsmpMat[i]         = 1

        end


        #--------------------------------------------------------
        # (c) Mutation
        #--------------------------------------------------------

        c = c*(0.95 + 0.10*exp(16*(acpt-trgt))/(1 + exp(16*(acpt-trgt))))

        para      = drawsMat[i-1, :, :]
        wght      = repeat(weightsMat[:, i], 1, Nparam)

        mu        = sum(para.*wght,dims=1)
        z         = (para - repeat(mu, Npart, 1))
        R       = (z.*wght)'*z

        Rdiag   = diagm(diag(R))
        Rchol   = transpose(cholesky(Hermitian(R)).L)
        Rchol2  = sqrt.(Rdiag)

        tune[1] = c
        tune[2] = R
        tune[3] = Nparam
        tune[4] = phi

        for j in 1:Npart
            ind_para, ind_loglh, ind_post, ind_acpt = MutationRWMH(para[j,:], loglh[j], logpost[j], tune, i, f)
            drawsMat[i,j,:] .= ind_para
            loglh[j]       = ind_loglh
            logpost[j]     = ind_post
            acpt_part[j] = ind_acpt
        end

        acpt = mean(acpt_part)

        cMat[i]    = c
        ESSMat[i]  = ESS
        acptMat[i] = acpt

        if mod(i, 1) == 0

            para = drawsMat[i, :, :]
            wght = repeat(weightsMat[:, i], 1, Nparam)

            mu  = sum(para.*wght,dims=1)

            sig = sum((para - repeat(mu, Npart, 1)).^2 .*wght,dims=1)
            sig = (sqrt.(sig))

            # totaltime = totaltime + toc()
            # avgtime   = totaltime/i
            # remtime   = avgtime*(Nphi-i)

            print("-----------------------------------------------\n")
            print(" Iteration = $i / $Nphi \n")
            print("-----------------------------------------------\n")
            print(" phi  = $(phi[i]) \n")
            print("-----------------------------------------------\n")
            print("  c    = $c\n")
            print("  acpt = $acpt\n")
            print("  ESS  = $ESS  ($nresamp total resamples.)\n")
            print("-----------------------------------------------\n")
            # print("  time elapsed   = $totaltime\n")
            # print("  time average   = $avgtime\n")
            # print("  time remained  = $remtime\n")
            # print("-----------------------------------------------\n")
            print("para      mean    std\n")
            print("------    ----    ----\n")
            for k in 1:Nparam
                print("$(param_names[k])     $(mu[k])    $(sig[k])\n")
            end

        end

    end

	end

    print("-----------------------------------------------\n")
    println("logML = $(sum(log.(constMat)))")
    print("-----------------------------------------------\n")

end

function SMC(;Nblock = 1,Npart = 1000,Nphi = 100,lambda = 2)

    Γ₀ = zeros(n_y,n_y)
    Γ₁ = zeros(n_y,n_y)
    Ψ = zeros(n_y,n_x)
    Π = zeros(n_y,n_η)
    ZZ = zeros(n_obs,n_y)
    SDX = zeros(n_x,n_x)

    RR = zeros(n_y,n_x)
    GG = zeros(n_y,n_y)
    eu = [0, 0]

    set_obs_eq!(ZZ)

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
    id = zeros(Int64,Npart)
    loglh   = zeros(Npart)
    logpost = zeros(Npart)
    nresamp = 0

    cMat    = zeros(Nphi)
    ESSMat  = zeros(Nphi)
    acptMat = zeros(Nphi)
    rsmpMat = zeros(Nphi)
    acpt_part = zeros(Npart)
	tune = Array{Any,1}(undef,4)

	f(x1,x2) = Likelihoods!(x1,x2,Y,bound,param,Γ₀,Γ₁,Ψ,Π,SDX,GG,RR,ZZ,eu)

    SMCInit!(weightsMat,constMat,drawsMat,logpost,loglh,Npart,phi,f)
    SMCIter!(c, acpt, weightsMat, constMat, id, drawsMat, logpost, loglh, nresamp, tune, cMat, ESSMat, acptMat, rsmpMat, acpt_part, Nphi, Npart, Nparam, phi,f, trgt)

    return (drawsMat[Nphi,:,:],weightsMat[:,Nphi])

end
