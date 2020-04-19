module SMC_main

using Statistics
using Distributed
using SharedArrays
using LinearAlgebra
@everywhere using SMC_aux

export estimate!, simul_draws

"Initializes a SMC estimation structure `S` the specified using prior distributions."
function prior_draw!(S::EstSMC)
	@sync @distributed for n in 1:S.n_part
		valid = false
		while !valid
			S.draws[n,:] .= [rand(S.EstM.prior[i]) for i in 1:S.EstM.model.n_param]
			S.loglh[n], S.logpost[n], _, valid = likelihood!(S.draws[n,:], 0., S.EstM)
		end
	end
end

"Writes in `id` the multinomial resampling of the integer interval 1:`n_part` according to weights `W`"
function multinomial_resampling!(id, W, n_part)
    U = rand(n_part)
    CumW = [sum(W[1:i]) for i in 1:n_part]
    @sync @distributed for i in 1:n_part
        id[i] = findfirst(x->(U[i]<x),CumW)
    end
end

"Carries out one step of a Metropolis Hastings sampler in a SMC context."
function mutation_RWMH!(S::EstSMC, i::Int, n_param::Int)
	wght = repeat(S.weights[:, i], 1, n_param)
	z = S.draws - repeat(sum(S.draws.*wght, dims=1), S.n_part, 1)
	R = (z.*wght)'*z
	sqrtR = transpose(cholesky(Hermitian(R)).L)
	@sync @distributed for j in 1:S.n_part

	    valid = false
		px = zeros(n_param)
		lx = -Inf
		postx = 0.

	    while !valid
	    	px .= S.draws[j,:] + S.c*transpose(sqrtR)*randn(n_param)
	        postx, lx, _, valid = likelihood!(px, S.ϕ[i], S.EstM)
	    end

	    post0 = S.logpost[j] + (S.ϕ[i]-S.ϕ[i-1])*S.loglh[j]

	    alp = exp(postx - post0)

	    u = rand()
	    if u < alp
	        S.draws[j,:] .= px
	        S.loglh[j] = lx
	        S.logpost[j] = postx
	        S.acpt_part[j] = 1
	    else
	        S.logpost[j] = post0
	        S.acpt_part[j] = 0
	    end

	end

end

"Estimates the SMC `S`."
function estimate!(S::EstSMC; display=1)
	@time begin
		println("SMC starts....")

		prior_draw!(S)
		S.weights[:,1] .= 1/S.n_part
		S.const_vec[1] = sum(S.weights[:,1])

		println("SMC recursion starts...")

		for i in 2:S.n_ϕ

			n_param = S.EstM.model.n_param

			#-----------------------------------
			# (a) Correction
			#-----------------------------------

			S.weights[:, i] .= S.weights[:, i-1].*exp.((S.ϕ[i]-S.ϕ[i-1]).*S.loglh)
			S.const_vec[i] = sum(S.weights[:, i])
			S.weights[:, i] .=  S.weights[:, i] ./ S.const_vec[i]

			#-----------------------------------
			# (b) Selection
			#-----------------------------------

			ESS = 1/sum(S.weights[:, i].^2)

			if (ESS < 0.5*S.n_part)

				multinomial_resampling!(S.id, S.weights[:, i], S.n_part)
				S.draws .= S.draws[S.id, :]

				S.loglh .= S.loglh[S.id]
				S.logpost .= S.logpost[S.id]
				S.weights[:, i] .= 1/S.n_part
				S.n_resamp = S.n_resamp + 1

			end

			#--------------------------------------------------------
			# (c) Mutation
			#--------------------------------------------------------

			S.c = S.c*(0.95 + 0.10*exp(16*(S.acpt-S.trgt))/(1 + exp(16*(S.acpt-S.trgt))))

			@everywhere $(S).c = $(S.c)

			mutation_RWMH!(S, i, n_param)

			S.acpt = mean(S.acpt_part)

			@everywhere $(S).acpt = $(S.acpt)

			S.c_vec[i]    = S.c
			S.ESS_vec[i]  = ESS
			S.acpt_vec[i] = S.acpt

			wght = repeat(S.weights[:, i], 1, n_param)

			S.μ .= reshape(sum(S.draws.*wght,dims=1),n_param)
			S.σ .= reshape(sum((S.draws - repeat(reshape(S.μ,1,n_param),S.n_part,1)).^2 .*wght,dims=1),n_param)
			S.σ .= sqrt.(S.σ)

			# totaltime = totaltime + toc()
			# avgtime   = totaltime/i
			# remtime   = avgtime*(n_phi-i)

			if mod(i, display) == 0

				print("-----------------------------------------------\n")
				print(" Iteration = $i / $(S.n_ϕ) \n")
				print("-----------------------------------------------\n")
				print(" ϕ  = $(S.ϕ[i]) \n")
				print("-----------------------------------------------\n")
				print("  c    = $(S.c)\n")
				print("  acpt = $(S.acpt)\n")
				print("  ESS  = $ESS  ($(S.n_resamp) total resamples.)\n")
				print("-----------------------------------------------\n")
			# 	# print("  time elapsed   = $totaltime\n")
			# 	# print("  time average   = $avgtime\n")
			# 	# print("  time remained  = $remtime\n")
				print("-----------------------------------------------\n")
				print("para      mean    std\n")
				print("------    ----    ----\n")
				for k in 1:n_param
					print("$(S.EstM.model.param_names[k])     $(S.μ[k])    $(S.σ[k])\n")
				end

			end

		end

		print("-----------------------------------------------\n")
		println("logML = $(sum(log.(S.const_vec)))")
		print("-----------------------------------------------\n")

	end

end

end
