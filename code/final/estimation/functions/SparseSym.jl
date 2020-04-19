module SparseSym

    export SparseMDArray, vector_jacobian, sparse_md, subs_array, to_matrix_csc, expected, spI, S

    using SparseArrays
    using SymEngine
    import Base.IteratorsMD.CartesianIndices
    import Base.iszero
    using Espresso

    iszero(x::Basic) = (x == 0) || (x == 0.0)

    struct SparseMDArray{Tv,Ti<:Integer}
        s::Tuple{Vararg{Ti}}
        ind::Vector{Tuple{Vararg{Ti}}}
        nzval::Vector{Tv}
    end

    function sparse_md(A::Array)
        s = size(A)
        ind = Tuple{Vararg{eltype(s)}}[]
        nzval = eltype(A)[]
        for i in CartesianIndices(s)
            if !iszero(A[i])
                push!(ind,i)
                push!(nzval,A[i])
            end
        end
        return SparseMDArray(s,ind,nzval)
    end

    function vector_jacobian(f::SparseMDArray{Basic}, v::Array{Basic})

        n_v = size(v,1)
        s = tuple(f.s...,n_v)
        ind = Tuple{Vararg{Int64}}[]
        nzval = Basic[]

        n_dim = length(f.s)+1
        for (j,var) in enumerate(v)
            for (i,expr) in enumerate(f.nzval)
                derivative = diff(expr,var)
                if !iszero(derivative)
                    push!(ind,tuple(f.ind[i]...,j))
                    push!(nzval,derivative)
                end
            end
        end

        return SparseMDArray(s,ind,nzval)

    end

    function subs(f::Basic, x::Vector{Basic}, x_ss::Vector)
        n = length(x)
        output = copy(f)
        for i in 1:n
            output = SymEngine.subs(output, x[i]=>x_ss[i])
        end
        return output
    end

    function subs_array(f::SparseMDArray{Basic}, x::Vector{Basic}, x_ss::Vector)
        ind = eltype(f.ind)[]
        nzval = eltype(f.nzval)[]
        for (k,c) in enumerate(f.ind)
            output = subs(f.nzval[k], x, x_ss)
            if !iszero(output)
                push!(nzval,output)
                push!(ind,c)
            end
        end
        return SparseMDArray(f.s,ind,nzval)
    end

    function subs_array(f::SparseMatrixCSC{Basic}, x::Vector{Basic}, x_ss::Vector)
        colptr = ones(eltype(f.colptr),f.n+1)
        rowval = eltype(f.rowval)[]
        nzval = eltype(f.nzval)[]
        j_km1 = 1
        j_k = 1
        j = 1
        k = 0
        while j < f.n+1
            if f.colptr[j] != f.colptr[j+1]
                for i in f.colptr[j]:f.colptr[j+1]-1
                    output = subs(f.nzval[i], x, x_ss)
                    if !iszero(output)
                        k += 1
                        j_k = j
                        push!(rowval, f.rowval[i])
                        push!(nzval, output)
                        if j == j_km1
                            colptr[j+1] += 1
                        else
                            colptr[j_km1+2:j] .= colptr[j_km1+1]
                            colptr[j+1] = colptr[j]+1
                            j_km1 = j
                        end
                    end
                end
            end
            j += 1
        end

        if j_k < f.n
            colptr[j_k+2:end] .= colptr[j_k+1]
        end

        return SparseMatrixCSC(f.m, f.n, colptr, rowval, nzval)
    end

    function to_matrix_csc(A::SparseMDArray)
        @assert length(A.s) > 1

        m = A.s[1]
        n = prod(A.s[2:end])
        colptr = ones(eltype(A.s),n+1)
        rowval = zeros(eltype(A.s),length(A.nzval))

        linear_from = LinearIndices(A.s)
        cartesian_to = CartesianIndices((m,n))

        j_km1 = 1
        j_k = 1

        for (k,c) in enumerate(A.ind)
            c_to = cartesian_to[linear_from[c...]]
            rowval[k] = c_to[1]
            j_k = c_to[2]

            if j_k == j_km1
                colptr[j_k+1] += 1
            else
                colptr[j_km1+2:j_k] .= colptr[j_km1+1]
                colptr[j_k+1] = colptr[j_k]+1
                j_km1 = j_k
            end
        end

        if j_k < n
            colptr[j_k+2:end] .= colptr[j_k+1]
        end

        return SparseMatrixCSC(m, n, colptr, rowval, A.nzval)

    end

    function expected(A::SparseMatrixCSC{Basic}, ϵ::Vector{Basic} ; order = 4)
        colptr = ones(eltype(A.colptr),A.n+1)
        rowval = eltype(A.rowval)[]
        nzval = Float64[]
        j_km1 = 1
        j = 1
        k = 0
        while j < A.n+1
            if A.colptr[j] != A.colptr[j+1]
                for i in A.colptr[j]:A.colptr[j+1]-1
                    expr = expected(A.nzval[i], ϵ, order)
                    if !iszero(expr)
                        k += 1
                        push!(rowval, A.rowval[i])
                        push!(nzval, expr)
                        if j == j_km1
                            colptr[j+1] += 1
                        else
                            colptr[j_km1+2:j] .= colptr[j_km1+1]
                            colptr[j+1] = colptr[j]+1
                            j_km1 = j
                        end
                    end
                end
            end
            j += 1
        end

        return SparseMatrixCSC(A.m, A.n, colptr, rowval, nzval)

    end

    function expected(A::SparseVector{Basic}, ϵ::Vector{Basic} ; order = 4)
        nzind = eltype(A.nzind)[]
        nzval = Float64[]
        for (j,basic)  in enumerate(A.nzval)
            expr = expected(basic, ϵ, order)
            if !iszero(expr)
                push!(nzind, j)
                push!(nzval, expr)
            end
        end

        return SparseVector(length(nzval), nzind, nzval)

    end

    function expected(expr::Basic,  ϵ::Vector{Basic} , order::Int64)
        E_expr = expr
        k = order
        go_on = typeof(E_expr) != Float64
        while go_on && k > 1
            var = 1
            while go_on && var < length(ϵ)+1
                E_expr = Espresso.subs(convert(Expr, E_expr), Dict(:($(Symbol(ϵ[var]))^$k) => :($(m(k)))))
                go_on = typeof(E_expr) != Float64
                var += 1
            end
            k -= 1
        end
        if go_on && k == 1
            var = 1
            while go_on && var < length(ϵ)+1
                E_expr = Espresso.subs(convert(Expr, E_expr), Dict(:($(Symbol(ϵ[var]))) => :($(m(k)))))
                go_on = typeof(E_expr) != Float64
                var += 1
            end
        end

        return eval(E_expr)

    end

    function m(k::Int64)
        if k % 2 == 1
            return 0.
        else
            q = k // 2
            return factorial(k)/((2^q)*factorial(q))
        end
    end


    function spI(n::Int64)
        return SparseMatrixCSC(n,n,collect(1:n+1),collect(1:n),ones(n))
    end

    function S(A::AbstractArray)
        return S(size(A)...)
    end

    function S(p::Int64,q::Int64)
        r = p*q
        colptr = collect(1:r+1)
        rowval = zeros(Int64,r)
        for i in 1:p
            rowval[(i-1)*q+1:i*q] .= i:p:r
        end
        return SparseMatrixCSC(r,r,colptr,rowval,ones(r))
    end

end
