function Sylvester(P,S,T,F)
    # Solves PY + SYT' = F
    n = size(P,1)
    Y = zeros(n,n)
    f = copy(F)
    k = n
    A = zeros(2*n,2*n)
    B = zeros(2*n)
    X = zeros(2*n)

    while k > 0
        if  (k == 1) || ( abs(T[k,k-1]) < eps(eltype(T)) )

            Y[:,k] = (P + T[k,k]*S)\f[:,k]

        else

            A[1:n,1:n] = P + T[k-1,k-1]*S
            A[1:n,n+1:end] = T[k-1,k]*S
            A[n+1:end,1:n] = T[k,k-1]*S
            A[n+1:end,n+1:end] = P + T[k,k]*S
            B[1:n] = f[:,k-1]
            B[n+1:end] = f[:,k]
            X = A\B
            Y[:,k-1] = X[1:n]
            Y[:,k] = X[n+1:end]

        end

        for j in 1:k-1
            f[:,j] -= (k == j)*P*Y[:,k] + T[j,k]*S*Y[:,k]
        end

        k -= 1

    end

    return Y
end
function vech(A::AbstractMatrix{T}) where T
    m = LinAlg.checksquare(A)
    v = Vector{T}((m*(m+1))>>1)
    k = 0
    for j = 1:m, i = j:m
        @inbounds v[k += 1] = A[i,j]
    end
    return v
end
function AIM(param,ss)

    NY = 15
    NX = 4

    GAM0 = zeros(NY,NY)
    GAM1 = zeros(NY,NY)
    GAM2 = zeros(NY,NY)
    GAM3 = zeros(NY,NX)

    rho = param[1]
    sigma = param[2]
    eta = param[3]
    m = param[4]
    sigz = param[5]
    epsilon = param[6]
    beta = param[7]
    gamma = param[8]
    F = param[9]
    b = param[10]
    xif = param[11]
    s = param[12]
    gsY = param[13]

    rho_A = param[14]
    rho_mu = param[15]
    rho_g = param[16]
    r_i = param[17]
    r_pi = param[18]
    r_y = param[19]
    psi = param[20]

    sig_A = param[21]
    sig_mu = param[22]
    sig_g = param[23]
    sig_m = param[24]

    #Distribution

    Ez = exp(sigz^2/2)
    CDF(x) = normcdf(0,sigz,log(max(x,0)))
    PDF(x) = normpdf(0,sigz,log(max(x,0)))/max(x,0)
    II(x) = exp(sigz^2/2)*normcdf(0,1,(sigz^2-log(max(x,0)))/sigz)
    int(x) = II(x) - x*(1-CDF(x))

    #Endogenous variables

    i = 1
    c = 2
    np = 3
    nf = 4
    zp = 5
    zf = 6
    zs = 7
    theta = 8
    phi = 9
    pi = 10
    Y = 11
    v = 12

    A = 13
    mu = 14
    g = 15

    #Perturbations

    e_A = 1
    e_mu = 2
    e_g = 3
    e_m = 4

    #Euler equation

    GAM0[1,c] = 1
    GAM1[1,c] = -1
    GAM0[1,i] = 1
    GAM1[1,pi] = -1

    #Definition of zp

    GAM0[2,A] = ss[phi]*(ss[zp] + beta*(1-s)*rho_A*int(ss[zp]))
    GAM0[2,phi] = ss[phi]*ss[zp]
    GAM0[2,zp] = ss[phi]*ss[zp]
    GAM0[2,c] = b - ss[phi]*ss[zp] - F
    GAM1[2,c] = -(b - ss[phi]*ss[zp] - F)
    GAM1[2,phi] = beta*(1-s)*ss[phi]*int(ss[zp])
    GAM1[2,theta] = -beta*(1-s)*(1-CDF(ss[zp]))*eta*gamma*ss[theta]/(1-eta)
    GAM1[2,zp] = -beta*(1-s)*((1-CDF(ss[zp]))*ss[phi] - eta*gamma*ss[theta]*PDF(ss[zp])/(1-eta))*ss[zp]

    #Definition of zf

    GAM0[3,A] = rho*ss[phi]*(ss[zf] + beta*(1-xif)*(Ez - ss[zf])*rho_A)
    GAM0[3,phi] = rho*ss[phi]*ss[zf]
    GAM0[3,zf] = rho*ss[phi]*ss[zf]
    GAM1[3,phi] = rho*beta*ss[phi]*(1-xif)*(Ez-ss[zf])
    GAM0[3,c] = b - rho*ss[phi]*ss[zf]
    GAM1[3,c] = -(b - rho*ss[phi]*ss[zf])
    GAM1[3,zf] = -rho*ss[phi]*beta*(1-xif)*ss[zf]
    GAM1[3,theta] = -beta*(1-xif)*eta*gamma*ss[theta]/(1-eta)

    #Definition of zs

    GAM0[4,zs] = (1-rho)*ss[zs]
    GAM0[4,zp] = -ss[zp]
    GAM0[4,phi] = F/ss[phi]
    GAM0[4,A] = F/ss[phi]
    GAM0[4,zf] = rho*ss[zf]

    #Job creation condition

    GAM0[5,A] = -gamma/((1-eta)*ss[phi]*m*ss[theta]^(-sigma))
    GAM0[5,phi] = -gamma/((1-eta)*ss[phi]*m*ss[theta]^(-sigma))
    GAM0[5,theta] = sigma*gamma/((1-eta)*ss[phi]*m*ss[theta]^(-sigma))
    GAM0[5,zs] = (1-rho)*(1-CDF(ss[zs]))*ss[zs]
    GAM0[5,zf] = rho*(1-CDF(ss[zf]))*ss[zf]

    #Permanent employment

    GAM0[6,np] = 1
    GAM2[6,np] = (1-s)*(1-CDF(ss[zp]))
    GAM0[6,zp] = (1-s)*ss[zp]*PDF(ss[zp])
    GAM0[6,v] = -(1-CDF(ss[zs]))*m*ss[theta]^(-sigma)*ss[v]/ss[np]
    GAM0[6,theta] = sigma*(1-CDF(ss[zs]))*m*ss[theta]^(-sigma)*ss[v]/ss[np]
    GAM0[6,zs] = ss[zs]*PDF(ss[zs])*m*ss[theta]^(-sigma)*ss[v]/ss[np]

    #Temporary employment

    GAM0[7,nf] = 1
    GAM2[7,nf] = (1-xif)
    GAM0[7,v] = -(CDF(ss[zs])-CDF(ss[zf]))*m*ss[theta]^(-sigma)*ss[v]/ss[nf]
    GAM0[7,theta] = sigma*(CDF(ss[zs])-CDF(ss[zf]))*m*ss[theta]^(-sigma)*ss[v]/ss[nf]
    GAM0[7,zs] = -ss[zs]*PDF(ss[zs])*m*ss[theta]^(-sigma)*ss[v]/ss[nf]
    GAM0[7,zf] = ss[zf]*PDF(ss[zf])*m*ss[theta]^(-sigma)*ss[v]/ss[nf]

    #Production function

    GAM0[8,Y] = 1.
    GAM0[8,A] = -1.
    GAM2[8,np] = (1-s)*II(ss[zp])*ss[np]/ss[Y]
    GAM2[8,nf] = rho*(1-xif)*Ez*ss[nf]/ss[Y]
    GAM0[8,zp] = (1-s)*ss[np]*ss[zp]^2*PDF(ss[zp])/ss[Y]
    GAM0[8,zs] = (1-rho)*ss[zs]^2*PDF(ss[zs])*ss[v]*m*ss[theta]^(-sigma)/ss[Y]
    GAM0[8,zf] = rho*ss[zf]^2*PDF(ss[zf])*ss[v]*m*ss[theta]^(-sigma)/ss[Y]
    GAM0[8,v] = -(II(ss[zs]) + rho*(II(ss[zf]) - II(ss[zs])))*ss[v]*m*ss[theta]^(-sigma)/ss[Y]
    GAM0[8,theta] = sigma*(II(ss[zs]) + rho*(II(ss[zf]) - II(ss[zs])))*ss[v]*m*ss[theta]^(-sigma)/ss[Y]

    #NK PC

    GAM0[9,pi] = 1
    GAM1[9,pi] = - beta
    GAM0[9,mu] = - 1
    GAM0[9,phi] = - (1-beta*psi)*(1-psi)/psi

    #Taylor rule

    GAM0[10,i] = 1
    GAM1[10,pi] = -(1-r_i)*r_pi
    GAM0[10,Y] = -(1-r_i)*r_y
    GAM2[10,i] = r_i
    GAM3[10,e_m] = 1

    #Definition of v

    GAM0[11,v] = 1.
    GAM0[11,theta] = -1.
    GAM2[11,np] = -(1-s)*(1-CDF(ss[zp]))*ss[theta]*ss[np]/ss[v]
    GAM2[11,nf] = -(1-xif)*ss[theta]*ss[nf]/ss[v]
    GAM0[11,zp] = -(1-s)*ss[theta]*PDF(ss[zp])*ss[zp]*ss[np]/ss[v]

    #Ressource constraint

    GAM0[12,Y] = 1
    GAM0[12,c] = -ss[c]/ss[Y]
    GAM0[12,g] = -ss[g]/ss[Y]
    GAM0[12,v] = -gamma*ss[v]/ss[Y]

    #Shock processes

    GAM0[13,A] = 1
    GAM2[13,A] = rho_A
    GAM3[13,e_A] = 1

    GAM0[14,g] = 1
    GAM2[14,g] = rho_g
    GAM3[14,e_g] = 1

    GAM0[15,mu] = 1
    GAM2[15,mu] = rho_mu
    GAM3[15,e_mu] = 1

    GAM1 *= -1.

    return (GAM0,GAM1,GAM2,GAM3)
end
function dAIM!(param,indposest,ss,dGAM0,dGAM1,dGAM2,dGAM3)

    NY = 15
    NX = 4

    rho = param[1]
    sigma = param[2]
    eta = param[3]
    m = param[4]
    sigz = param[5]
    epsilon = param[6]
    beta = param[7]
    gamma = param[8]
    F = param[9]
    b = param[10]
    xif = param[11]
    s = param[12]
    gsY = param[13]

    rho_A = param[14]
    rho_mu = param[15]
    rho_g = param[16]
    r_i = param[17]
    r_pi = param[18]
    r_y = param[19]
    psi = param[20]

    sig_A = param[21]
    sig_mu = param[22]
    sig_g = param[23]
    sig_m = param[24]
    
    #Distribution

    Ez = exp(sigz^2/2)
    CDF(x) = normcdf(0,sigz,log(max(x,0)))
    PDF(x) = normpdf(0,sigz,log(max(x,0)))/max(x,0)
    II(x) = exp(sigz^2/2)*normcdf(0,1,(sigz^2-log(max(x,0)))/sigz)
    int(x) = II(x) - x*(1-CDF(x))

    #Endogenous variables

    i = 1
    c = 2
    np = 3
    nf = 4
    zp = 5
    zf = 6
    zs = 7
    theta = 8
    phi = 9
    pi = 10
    Y = 11
    v = 12

    A = 13
    mu = 14
    g = 15

    #Perturbations

    e_A = 1
    e_mu = 2
    e_g = 3
    e_m = 4

    Nparam = size(indposest,1)

    for i in 1:Nparam

        dGAM0[i] = zeros(NY,NY)
        dGAM1[i] = zeros(NY,NY)
        dGAM2[i] = zeros(NY,NY)
        dGAM3[i] = zeros(NY,NX)

    end

    #rho_A

    dGAM0[1][2,A] = ss[phi]*beta*(1-s)*int(ss[zp])
    dGAM0[1][3,A] = rho*ss[phi]*beta*(1-xif)*(Ez - ss[zf])
    dGAM2[1][13,A] = 1.

    #rho_mu

    dGAM2[2][15,mu] = 1.

    #rho_g

    dGAM2[3][14,g] = 1.

    #r_i

    dGAM1[4][10,pi] = r_pi
    dGAM1[4][10,Y] = r_y
    dGAM2[4][10,i] = 1.

    #r_pi

    dGAM1[5][10,pi] = -(1-r_i)

    #r_y

    dGAM0[6][10,Y] = -(1-r_i)

    #psi

    dGAM0[7][9,phi] = (1-beta*psi^2)/psi^2

    #sig_A

    dGAM3[8][13,e_A] = 1

    #sig_mu

    dGAM3[9][15,e_mu] = 1

    #sig_g

    dGAM3[10][14,e_g] = 1

    #sig_m

    dGAM3[11][10,e_m] = 1

    for i in 1:Nparam
        dGAM1[i] *= -1.
    end
end
function Iskrev!(A,B,SDX,C,param,indposest,ss,Y,dGAM0,dGAM1,dGAM2,dGAM3,dA,dB,dO,dSIG0,dSIG,J1,J2)

    ##################################J2########################################

    GAM0,GAM1,GAM2,GAM3 = AIM(param,ss)
    dAIM!(param,indposest,ss,dGAM0,dGAM1,dGAM2,dGAM3)

    A = A[1:NY,1:NY]
    B = B[1:NY,:]
    B = B*SDX
    C = C[:,1:NY]
    O = B*B'

    F1 = schurfact(GAM0 - GAM1*A,-GAM1)
    F2 = schurfact(A)

    T = F2[:T]
    Q2 = F2[:Z]'
    Z2 = Q2'

    P = F1[:S]
    S = F1[:T]

    Q1 = F1[:Q]'
    Z1 = F1[:Z]

    for i in 1:Nparam
        F = Q1*(dGAM2[i] - (dGAM0[i]-dGAM1[i]*A)*A)*Q2'
        dA[i] = Z1*Sylvester(P,S,T,F)*Z2'
        dB[i] = (GAM0 - GAM1*A)\(dGAM3[i] - (dGAM0[i] - dGAM1[i]*A - GAM1*dA[i])*B)
        dO[i] = dB[i]*B' + B*dB[i]'
    end

    for i in 1:Nparam
        J2[1:NY^2,i] = vec(dA[i])
        J2[NY^2+1:end,i] = vech(dO[i])
    end

    id = [true,true]

    if rank(J2) < Nparam
        id[1] = false
    end

    ##################################J1########################################

    T = size(Y,1)
    SIG0 = solve_discrete_lyapunov(A, O)

    for i in 1:Nparam
        dSIG0[i] = solve_discrete_lyapunov(A, dO[i] + dA[i]*SIG0*A' + A*SIG0*dA[i]' )
    end
    A_p_im1 = eye(A)

    for i in 1:Nparam
        dSIG[1,i] = C*SIG0*C'
    end
    for t in 1:T-1
        for i in 1:Nparam
            dSIG[t+1,i] = C*A_p_im1*(t*SIG0 + A*dSIG0[i])*C'
        end
        A_p_im1 *= A
    end

    for i in 1:Nparam
        for t in 0:T-1
            J1[trunc(Int,t*NX*(NX+1)/2)+1:trunc(Int,(t+1)*NX*(NX+1)/2),i] = vech(dSIG[t+1,i])
        end
    end

    if rank(J1) < Nparam
        id[2] = false
    end

    return id
end
