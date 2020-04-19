module Kamenik

    using LinearAlgebra, SparseArrays

    export kamenik_solve, kron_pow, vector_solve

    function kron_pow(A::AbstractArray,k::Int64)
        if k == 0
            return I(size(A,1))
        else
            out = A
            for i in 2:k
                out = kron(A,out)
            end
            return out
        end
    end

    import LinearAlgebra.kron

    function kron(x...)
        n = length(x)
        output = x[1]
        for k in 2:n
            output = kron(output,x[k])
        end
        return output
    end

    function vector_solve(A::Array{Float64,2}, B::Array{Float64,2}, C::Array{Float64,2}, D::Array{Float64,2}, k::Int64)

        n = size(A,1)
        m = size(C,1)

        F0 = schur(A\B)
        U = sparse(F0.Z')
        K = sparse(F0.T)
        F1 = schur(C)
        V = sparse(F1.Z')
        F = sparse(F1.T)

        D_bar = U*(A\D)*kron_pow(V',k)

        Temp = kron(kron_pow(F',k),K)
        vec_Y = ( I(n*m^k) + Temp )\vec(D_bar)
        Y = reshape(vec_Y,(n,m^k))
        return U'*Y*kron_pow(V,2)

    end

    function set_F_k(F_t, K, k::Int64)
        if k == 0
            return K
        else
            return kron(kron_pow(F_t,k),K)
        end
    end

    function eig_blocks(F)
        m = size(F,1)
        j_bar = Int64[]
        blocks = Bool[]
        j = 1
        while j < m+1
            push!(j_bar,j)
            if isreal(F[j,j])
                push!(blocks,true)
                j += 1
            else
                push!(blocks,false)
                j += 2
            end
        end
        return (j_bar,blocks)
    end

    function kamenik_solve(A::Array, B::Array, C::Array, D::AbstractArray, k::Int64)

        n = size(A,1)
        m = size(C,1)

        F0 = schur(A\B)
        U = sparse(F0.Z')
        K = sparse(F0.T)
        F1 = schur(C)
        V = sparse(F1.Z')
        F = sparse(F1.T')

        G = F*F
        j_bar,blocks = eig_blocks(F)
        F_k = [set_F_k(F,K,j) for j in 0:2]
        F_k2 = [M*M for M in F_k]

        vec_D_bar = vec(U*(A\Array(D))*kron_pow(V', k))
        vec_Y = zeros(n*m^k)
        solv1!(vec_Y, vec_D_bar, k, 1., F, G, K, F_k, F_k2, j_bar, blocks, n, m)
        Y = reshape(vec_Y, (n, m^k))
        return U'*Y*kron_pow(V, k)

    end

    function solv1!(y, d, k::Int64, r::Float64, F, G, K, F_k, F_k2, j_bar::Vector{Int64}, blocks::Vector{Bool}, n::Int64, m::Int64)
        if k == 0
            y .= (I(n) + r*K)\Array(d)
        else
            s = n*m^(k-1)
            for (b,j) in enumerate(j_bar)
                if blocks[b]
                    ind_b = ((j-1)*s+1):j*s
                    solv1!(view(y,ind_b), view(d,ind_b), k-1, r*F[j,j], F, G, K, F_k, F_k2, j_bar, blocks, n, m)
                    z = r*F_k[k]*y[ind_b]
                    for i in j+1:m
                        if i > j
                            d[(i-1)*s+1:i*s] .= d[(i-1)*s+1:i*s] - F[i,j]*z
                        end
                    end
                else
                    ind_b = ((j-1)*s+1):(j+1)*s
                    α = F[j,j]
                    β1 = F[j,j+1]
                    β2 = -F[j+1,j]
                    solv2!(view(y,ind_b), view(d,ind_b), k-1, r*α, r*β1, r*β2, F, G, K, F_k, F_k2, j_bar, blocks, n, m)
                    z1 = r*F_k[k]*y[((j-1)*s+1):j*s]
                    z2 = r*F_k[k]*y[(j*s+1):(j+1)*s]
                    for i in j+2:m
                        d[(i-1)*s+1:i*s] .= d[(i-1)*s+1:i*s] - F[i,j]*z1 - F[i,j+1]*z2
                    end
                end
            end
        end
    end

    function solv2!(y, d, k::Int64, α::Float64, β1::Float64, β2::Float64, F, G, K, F_k, F_k2, j_bar::Vector{Int64}, blocks::Vector{Bool}, n::Int64, m::Int64)
        s = n*m^k
        d_hat = (kron(I(2),I(n*m^k))+kron([α β1 ; -β2 α], F[k+1]))*d
        solv2p!(view(y,1:s), view(d_hat,1:s), k, α, β1*β2, F, G, K, F_k, F_k2, j_bar, blocks, n, m)
        solv2p!(view(y,s+1:2*s), view(d_hat,s+1:2*s), k, α, β1*β2, F, G, K, F_k, F_k2, j_bar, blocks, n, m)
    end

    function solv2p!(y, d, k::Int64, α::Float64, β2::Float64, F, G, K, F_k, F_k2, j_bar::Vector{Int64}, blocks::Vector{Bool}, n::Int64, m::Int64)
        if k == 0
            y .= (I(n)+2*α*K+(α^2+β2)*F_k[k+1])\d
        else
            s = n*m^(k-1)
            for (b,j) in enumerate(j_bar)
                if blocks[b]
                    ind_b = ((j-1)*s+1):j*s
                    solv2p!(view(y,ind_b), view(d,ind_b), k-1, F[j,j]*α, F[j,j]^2*β2, F, G, K, F_k, F_k2, j_bar, blocks, n, m)
                    z = 2*α*F_k[k]*y[ind_b]
                    w = (α^2+β2)*F_k[k]^2*y[ind_b]
                    for i in j+1:m
                        d[(i-1)*s+1:i*s] .= d[(i-1)*s+1:i*s] - F[i,j]*z - G[i,j]*w
                    end
                else
                    ind_b = ((j-1)*s+1):(j+1)*s
                    γ = F[j,j]
                    δ1 = F[j,j+1]
                    δ2 = -F[j+1,j]
                    M = [γ δ_1 ; -δ_2 ]
                    d_hat = (kron(I(2),I(n*m^(k-1)))+2*α*kron(M,F_k[k]) + (α^2 + β2)*kron(M^2,F_k[k]^2))*d[ind_b]
                    δ = sqrt(δ1*δ2)
                    β = sqrt(β2)
                    a1 = α*γ - β*δ
                    b1 = α*δ + γ*β
                    a2 = α*γ + β*δ
                    b2 = α*δ - γ*β
                    ind_b1 = ((j-1)*s+1):j*s
                    ind_b2 = (j*s+1):(j+1)*s
                    solv2p!(view(y,ind_b1), view(d_hat,ind_b1), k-1, a2, b2^2, F, G, K, F_k, F_k2, j_bar, blocks, n, m)
                    solv2p!(view(y,ind_b1), view(y,ind_b1), k-1, a1, b1^2, F, G, K, F_k, F_k2, j_bar, blocks, n, m)
                    solv2p!(view(y,ind_b2), view(d_hat,ind_b2), k-1, a2, b2^2, F, G, K, F_k, F_k2, j_bar, blocks, n, m)
                    solv2p!(view(y,ind_b2), view(y,ind_b2), k-1, a1, b1^2, F, G, K, F_k, F_k2, j_bar, blocks, n, m)
                    z1 = 2*α*F_k[k]*y[ind_b1]
                    z2 = 2*α*F_k[k]*y[ind_b2]
                    w1 = (α^2+β2)*F_k2[k]*y[ind_b1]
                    w2 = (α^2+β2)*F_k2[k]*y[ind_b2]
                    for i in j+2:m
                        d[(i-1)*s+1:i*s] .= d[(i-1)*s+1:i*s] - F[i,j]*z1 - F[i,j+1]*z2 - G[i,j]*w1 - G[i,j+1]*w2
                    end
                end
            end
        end
    end

end
