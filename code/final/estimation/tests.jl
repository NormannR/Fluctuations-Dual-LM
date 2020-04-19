

S.loglh[461]
px = S.draws[1,461,:]
# %%
likelihood!(px, S.ϕ[1], S.EstM)
# %%
i = 2
S.weights[:, i] .= S.weights[:, i-1].*exp.((S.ϕ[i]-S.ϕ[i-1]).*S.loglh)
findall(isnan.(S.weights[:, i]))
findall(isnan.(S.loglh))
S.const_vec[i] = sum(S.weights[:, i])
# %%

# %%
# %%
param = [0.985705345582615, 0.011715368959866202, 0.4742977399572977, 1.2247838204074117, 0.0241500509271413, 0.7341390832282619, 0.22904208124287684, 0.17972596915997935, 0.11918645860091112, 0.29855144324221883]
# %%
param = [0.47266689478277424, 0.6465121868810297, 0.48047904774542594, 3.5168904994182957, 0.19203963005951655, 0.7549649726944838, 0.26040889312227644, 0.5544878236490646, 0.6821632506240077, 0.19240913469659435]
EstM = init_EstLinMod(prior(), Y, bound, M)

M = EstM.model
n_x = length(M.ss_x)
n_y = length(M.ss_y)
n_eq = n_x + n_y
f_x_t = zeros(n_eq,n_x)
f_x_tp1 = zeros(n_eq,n_x)
f_y_t = zeros(n_eq,n_y)
f_y_tp1 = zeros(n_eq,n_y)
g_x = zeros(n_y, n_x)
h_x = zeros(n_x, n_x)
eu = [true,true]

G = zeros(n_x, n_x)
R = zeros(n_x, M.n_ϵ)
Z = zeros(M.n_ϵ, n_x)

M.set_lin_mod!(f_y_tp1, f_y_t, f_x_tp1, f_x_t, param)

A = hcat(f_x_tp1, f_y_tp1)
B = -hcat(f_x_t, f_y_t)

using LinearAlgebra

fill!(g_x, 0.)
fill!(h_x, 0.)
fill!(eu, true)

F = eigen(B, A)
perm = sortperm(abs.(F.values))
V = F.vectors[:,perm]
D = F.values[perm]
m = findlast(abs.(D) .< 1)

V[11,:]

perm[10]
D[10]

n_x = length(M.ss_x)
n_y = length(M.ss_y)

h1 = real.(V[1:m,1:m]*Diagonal(D)[1:m,1:m]*inv(V[1:m,1:m]))
g1 = real.(V[m+1:end,1:m]*inv(V[1:m,1:m]))
# %%
param = [0.7295330689618175, 0.5039165983932837, 0.311251624945484, 1.5057286528897953, 0.17086794750240183, 0.732836639462445, 0.2333846538298944, 0.2999108857902937, 0.3525635094896005, 0.337908730489954]

EstM = init_EstLinMod(prior(), Y, bound, M)

M = EstM.model
n_x = length(M.ss_x)
n_y = length(M.ss_y)
n_eq = n_x + n_y
f_x_t = zeros(n_eq,n_x)
f_x_tp1 = zeros(n_eq,n_x)
f_y_t = zeros(n_eq,n_y)
f_y_tp1 = zeros(n_eq,n_y)
g_x = zeros(n_y, n_x)
h_x = zeros(n_x, n_x)
eu = [true,true]

G = zeros(n_x, n_x)
R = zeros(n_x, M.n_ϵ)
Z = zeros(M.n_ϵ, n_x)

M.set_lin_mod!(f_y_tp1, f_y_t, f_x_tp1, f_x_t, param)

A = hcat(f_x_tp1, f_y_tp1)
B = -hcat(f_x_t, f_y_t)

using LinearAlgebra

fill!(g_x, 0.)
fill!(h_x, 0.)
fill!(eu, true)

F = eigen(B, A)
perm = sortperm(abs.(F.values))
V = F.vectors[:,perm]
D = F.values[perm]
m = findlast(abs.(D) .< 1)

n_x = length(M.ss_x)
n_y = length(M.ss_y)

h2 = real.(V[1:m,1:m]*Diagonal(D)[1:m,1:m]*inv(V[1:m,1:m]))
g2 = real.(V[m+1:end,1:m]*inv(V[1:m,1:m]))
# %%
function foo()
  S = SMC_dual
  n = 100000
  valid = 0
  invalid = 0
  valid_p = zeros(0,S.EstM.model.n_param)
  invalid_p = zeros(0,S.EstM.model.n_param)
  for k in 1:n
    p = [rand(S.EstM.prior[i]) for i in 1:S.EstM.model.n_param]
    _, _, _, flag = likelihood!(p, 0., S.EstM)
    p = reshape(p,1,S.EstM.model.n_param)
    if flag
      valid_p=vcat(valid_p, p)
    else
      invalid_p=vcat(invalid_p, p)
    end
  end
  return (n,valid,invalid,valid_p,invalid_p)
end
# %%
n,valid,invalid,valid_p,invalid_p = foo()
# %%
m1 = mean(valid_p, dims = 1)
m2 = mean(invalid_p, dims = 1)

m1-m2

# %%
likelihood!(param, 0., EstM)

SMC_dual.draws[1,:,:]

ϕ = 0.
x = [ 0.84,
0.90,
0.74,
1.82,
0.66,
0.79,
0.13,
0.26,
4.75,
0.08 ]
# %%
using BenchmarkTools
@btime likelihood!(x, ϕ, EstM)

# %%

# %%
using SparseSym
# %%
xp, x, yp, y, Eq = M.sym_mod(param)
v = vcat(xp, x, yp, y)
v_ss = zeros(length(v))
# %%
subs_array(sparse_md(Eq),v,v_ss)
# %%
f_x_tp1_2 = to_matrix_csc(subs_array(vector_jacobian(sparse_md(Eq),xp),v,v_ss))
f_y_tp1_2 = to_matrix_csc(subs_array(vector_jacobian(sparse_md(Eq),yp),v,v_ss))
f_x_t_2 = to_matrix_csc(subs_array(vector_jacobian(sparse_md(Eq),x),v,v_ss))
f_y_t_2 = to_matrix_csc(subs_array(vector_jacobian(sparse_md(Eq),y),v,v_ss))
# %%
using SparseArrays
f_x_tp1_2 - sparse(f_x_tp1)
f_x_t_2 - sparse(f_x_t)
Float64.(f_y_tp1_2 - sparse(f_y_tp1))
abs.(Float64.(f_y_t_2 - sparse(f_y_t)) .> 1e-13)
Float64.(f_y_t_2)
sparse(f_y_t)
# %%
using DualTiming
using SMC
using FilterData
using SparseSym
# %%
M = lin_dual()
# %%
n_x = length(M.ss_x)
n_y = length(M.ss_y)
n_eq = n_x + n_y
f_x_t = zeros(n_eq,n_x)
f_x_tp1 = zeros(n_eq,n_x)
f_y_t = zeros(n_eq,n_y)
f_y_tp1 = zeros(n_eq,n_y)
# %%
g_x = zeros(n_y, n_x)
h_x = zeros(n_x, n_x)
eu = [true,true]
solve!(g_x, h_x, eu, f_y_tp1, f_y_t, f_x_tp1, f_x_t, param, M)
# %%
M = sym_dual()
xp, x, yp, y, Eq = M.sym_mod(param)
v = vcat(xp, x, yp, y)
v_ss = zeros(length(v))
# %%
subs_array(sparse_md(Eq),v,v_ss)
# %%
f_x_tp1_2 = to_matrix_csc(subs_array(vector_jacobian(sparse_md(Eq),xp),v,v_ss))
f_y_tp1_2 = to_matrix_csc(subs_array(vector_jacobian(sparse_md(Eq),yp),v,v_ss))
f_x_t_2 = to_matrix_csc(subs_array(vector_jacobian(sparse_md(Eq),x),v,v_ss))
f_y_t_2 = to_matrix_csc(subs_array(vector_jacobian(sparse_md(Eq),y),v,v_ss))
# %%
using SparseArrays
f_x_tp1_2 - sparse(f_x_tp1)
# %%
f_x_t_2 - sparse(f_x_t)
# %%
abs.(Float64.(f_y_tp1_2 - sparse(f_y_tp1))) .> 1e-12
# %%
abs.(Float64.(f_y_t_2 - sparse(f_y_t)) .> 1e-12)
# %%
EstM = init_EstLinMod(prior(), Y, bound, M)
# %%
n_x = length(M.ss_x)
n_y = length(M.ss_y)
n_eq = n_x + n_y
f_x_t = zeros(n_eq,n_x)
f_x_tp1 = zeros(n_eq,n_x)
f_y_t = zeros(n_eq,n_y)
f_y_tp1 = zeros(n_eq,n_y)
g_x = zeros(n_y, n_x)
h_x = zeros(n_x, n_x)
eu = [true,true]

G = zeros(n_x, n_x)
R = zeros(n_x, M.n_ϵ)
Z = zeros(M.n_ϵ, n_x)

param = [ 0.84,
0.90,
0.74,
1.82,
0.66,
0.79,
0.13,
0.26,
4.75,
0.08 ]
# %%
M.set_lin_mod!(f_y_tp1, f_y_t, f_x_tp1, f_x_t, param)
# %%

# %%
using SparseSym
# %%
M = sym_classic()
xp, x, yp, y, Eq = M.sym_mod(param)
v = vcat(xp, x, yp, y)
v_ss = zeros(length(v))
# %%
Eq
# %%
subs_array(sparse_md(Eq),v,v_ss)
# %%
f_x_tp1_2 = to_matrix_csc(subs_array(vector_jacobian(sparse_md(Eq),xp),v,v_ss))
f_y_tp1_2 = to_matrix_csc(subs_array(vector_jacobian(sparse_md(Eq),yp),v,v_ss))
f_x_t_2 = to_matrix_csc(subs_array(vector_jacobian(sparse_md(Eq),x),v,v_ss))
f_y_t_2 = to_matrix_csc(subs_array(vector_jacobian(sparse_md(Eq),y),v,v_ss))
# %%
using SparseArrays
# %%
sparse(f_x_tp1_2-f_x_tp1)
# %%
sparse(f_x_t_2-f_x_t)
# %%
Float64.(sparse(f_y_tp1_2-f_y_tp1))
# %%
Float64.(sparse(f_y_t_2-f_y_t)).nzval
# %%
A = hcat(f_x_tp1, f_y_tp1)
B = -hcat(f_x_t, f_y_t)

using LinearAlgebra

F = eigen(B, A)
perm = sortperm(abs.(F.values))
V = F.vectors[:,perm]
D = F.values[perm]
m = findlast(abs.(D) .< 1)

n_x = length(M.ss_x)
n_y = length(M.ss_y)

solve_eig!(g_x, h_x, eu, A, B, M)
