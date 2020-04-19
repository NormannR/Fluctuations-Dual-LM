import Pkg; Pkg.activate("../../../../")
push!(LOAD_PATH,pwd())
using Revise
using JLD2, FileIO, CSV, DataFrames,  Plots, FilterData, VAR
# %%
pyplot()
# %%
path = "../input/var.csv"
vars = [:CAC40,:VOL,:R,:W,:CPI,:N,:Y]
detrend = [:CAC40,:VOL,:W,:CPI,:N,:Y]
demean = [:R]
t1 = "1960Q1"
t2 = "2020Q1"
Y = hamilton_detrend(path,vars,detrend,demean,t1,t2)
# %%
plot(Y, label=["CAC40" "VOL" "R" "W" "CPI" "N" "Y"])
# %%
VAR_estimation(Y',1)
# %%
IRF = irf_var(Y,1)
# %%
plot(IRF[1,2,:])
plot(IRF[6,2,:])
plot(IRF[7,2,:])
# %%
vars = [:VOL,:N,:Y]
detrend = vars
demean = []
Y = hamilton_detrend(path,vars,detrend,demean,t1,t2)
IRF = irf_var(Y,1)
plot(IRF[3,1,:])
# %%
vars = [:Y,:N,:VOL]
detrend = vars
demean = []
Y = hamilton_detrend(path,vars,detrend,demean,t1,t2)
IRF = irf_var(Y,1)
plot(IRF[1,3,:])
# %%
vars = [:CAC40,:VOL,:CPI,:W,:MUFSMU,:JC,:JD,:N]
detrend = vars
demean = []
Y = hamilton_detrend(path,vars,detrend,demean,t1,t2)
IRF = irf_var(Y,1)
plot(IRF[5,2,:])
# %%
vars = [:VOL,:MUFSMU,:JC,:JD,:N]
detrend = vars
demean = []
Y = hamilton_detrend(path,vars,detrend,demean,t1,t2)
IRF = irf_var(Y,1)
plot(IRF[2,1,:])
# %%
vars = [:CAC40,:VOL,:W,:MUFSMU,:JC,:JD,:N]
detrend = vars
demean = []
Y = hamilton_detrend(path,vars,detrend,demean,t1,t2)
IRF = irf_var(Y,1)
plot(IRF[4,2,:])
# %%
vars = [:CAC40,:VOL,:W,:MUFSMU,:JC,:JD]
detrend = vars
demean = []
Y = hamilton_detrend(path, vars, detrend, demean, t1, t2)
IRF = irf_var(Y,1)
plot(IRF[4,2,:])
# %%
vars = [:VOL,:W,:MUFSMU,:JC,:JD,:N]
detrend = vars
demean = []
Y = hamilton_detrend(path, vars, detrend, demean, t1, t2)
IRF = irf_var(Y,1)
plot(IRF[3,1,:])
# %%
path = "../input/var.csv"
vars = [:CAC40,:VOL,:R,:W,:CPI,:N,:Y]
detrend = [:CAC40,:VOL,:W,:CPI,:N,:Y]
demean = [:R]
t1 = "1960Q1"
t2 = "2020Q1"
Y = hamilton_detrend(path,vars,detrend,demean,t1,t2)
IRF = sim_irf_var(Y, 1)
# %%
plot(IRF[7,2,:,1],ribbon=(IRF[7,2,:,2], IRF[7,2,:,3]), linestyle = :solid, linewidth=2, color = "black")
# %%
