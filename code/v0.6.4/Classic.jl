
function steady_state_eval(param)

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

    Ez = exp(sigz^2/2)
    G(x) = normcdf(0,sigz,log(max(x,0)))
    II(x) = exp(sigz^2/2)*normcdf(0,1,(sigz^2-log(max(x,0)))/sigz)
    int(x) = II(x) - x*(1-G(x))

    phi = (epsilon-1)/epsilon
    i = 1/beta - 1
    mu = 1/phi

    eq_zp(zp,theta) = phi*zp+(1-beta*(1-s))*phi*F+beta*(1-s)*phi*int(zp)-b-beta*(1-s)*(1-G(zp))*eta*gamma*theta/(1-eta)
    fun_zp(theta) = find_zero(zp -> eq_zp(zp,theta), (0.1,1.5), Bisection())
    LHS(theta) = gamma/((1-eta)*phi*m*theta^(-sigma))
    RHS(theta) = int(fun_zp(theta)+F*phi)

    theta = find_zero(theta -> LHS(theta) - RHS(theta), (0.2,1.4), Bisection())
    zp = fun_zp(theta)
    zc = zp+F

    q = m*theta^(-sigma)
    p = theta*q

    mup = p*(1-G(zc))
    xip = s+(1-s)*G(zp)
    np = mup/(xip*m) + xif*mup*(1-xip) + xip*xif)
    u = 1-np-nf

    e = u+xip*np
    v = theta*e

    Y = np*((1-xip)*II(zp)/(1-G(zp)) + xip*II(zs)/(1-G(zs))) + nf*rho*((1-xif)*Ez +
        xif*(II(zf)-II(zs))/(G(zs)-G(zf)) )

    g = gsY*Y
    c = Y-g-gamma*v

    A = 1
    pi = 1
    Delta = 1

    return [ 1+i, c, np, nf, zp, zf, zs, theta, phi, pi, Y, v, A, mu, g, c, zp, zf, theta, phi, pi]

end

function linearized_model_ss(param,ss)

    NNY = 15
    NX = 4
    NETA = 6
    NY = NNY + NETA

    GAM0 = zeros(NY,NY)
    GAM1 = zeros(NY,NY)
    PSI = zeros(NY,NX)
    PPI = zeros(NY,NETA)
    C = zeros(NY)

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

    E_c = 16
    E_zp = 17
    E_zf = 18
    E_theta = 19
    E_phi = 20
    E_pi = 21

    #Perturbations

    e_A = 1
    e_mu = 2
    e_g = 3
    e_m = 4

    #Expectation errors

    eta_c = 1
    eta_zp = 2
    eta_zf = 3
    eta_theta = 4
    eta_phi = 5
    eta_pi = 6

    #Euler equation

    GAM0[1,c] = 1
    GAM0[1,E_c] = -1
    GAM0[1,i] = 1
    GAM0[1,E_pi] = -1

    #Definition of zp

    GAM0[2,A] = ss[phi]*(ss[zp] + F*(1-beta*(1-s)*rho_A) + beta*(1-s)*rho_A*int(ss[zp]))
    GAM0[2,phi] = ss[phi]*(ss[zp]+F)
    GAM0[2,zp] = ss[phi]*ss[zp]
    GAM0[2,c] = b - ss[phi]*ss[zp] - ss[phi]*F
    GAM0[2,E_c] = -(b - ss[phi]*ss[zp] - ss[phi]*F)
    GAM0[2,E_phi] = beta*(1-s)*ss[phi]*(int(ss[zp])-F)
    GAM0[2,E_theta] = -beta*(1-s)*(1-CDF(ss[zp]))*eta*gamma*ss[theta]/(1-eta)
    GAM0[2,E_zp] = -beta*(1-s)*((1-CDF(ss[zp]))*ss[phi] - eta*gamma*ss[theta]*PDF(ss[zp])/(1-eta))*ss[zp]

    #Definition of zf

    GAM0[3,A] = rho*ss[phi]*(ss[zf] + beta*(1-xif)*(Ez - ss[zf])*rho_A)
    GAM0[3,phi] = rho*ss[phi]*ss[zf]
    GAM0[3,zf] = rho*ss[phi]*ss[zf]
    GAM0[3,E_phi] = rho*beta*ss[phi]*(1-xif)*(Ez-ss[zf])
    GAM0[3,c] = b - rho*ss[phi]*ss[zf]
    GAM0[3,E_c] = -(b - rho*ss[phi]*ss[zf])
    GAM0[3,E_zf] = -rho*ss[phi]*beta*(1-xif)*ss[zf]
    GAM0[3,E_theta] = -beta*(1-xif)*eta*gamma*ss[theta]/(1-eta)

    #Definition of zs

    GAM0[4,zs] = (1-rho)*ss[zs]
    GAM0[4,zp] = -ss[zp]
    GAM0[4,zf] = rho*ss[zf]

    #Job creation condition

    GAM0[5,A] = -gamma/((1-eta)*ss[phi]*m*ss[theta]^(-sigma))
    GAM0[5,phi] = -gamma/((1-eta)*ss[phi]*m*ss[theta]^(-sigma))
    GAM0[5,theta] = sigma*gamma/((1-eta)*ss[phi]*m*ss[theta]^(-sigma))
    GAM0[5,zs] = (1-rho)*(1-CDF(ss[zs]))*ss[zs]
    GAM0[5,zf] = rho*(1-CDF(ss[zf]))*ss[zf]

    #Permanent employment

    GAM0[6,np] = 1
    GAM1[6,np] = (1-s)*(1-CDF(ss[zp]))
    GAM0[6,zp] = (1-s)*ss[zp]*PDF(ss[zp])
    GAM0[6,v] = -(1-CDF(ss[zs]))*m*ss[theta]^(-sigma)*ss[v]/ss[np]
    GAM0[6,theta] = sigma*(1-CDF(ss[zs]))*m*ss[theta]^(-sigma)*ss[v]/ss[np]
    GAM0[6,zs] = ss[zs]*PDF(ss[zs])*m*ss[theta]^(-sigma)*ss[v]/ss[np]

    #Temporary employment

    GAM0[7,nf] = 1
    GAM1[7,nf] = (1-xif)
    GAM0[7,v] = -(CDF(ss[zs])-CDF(ss[zf]))*m*ss[theta]^(-sigma)*ss[v]/ss[nf]
    GAM0[7,theta] = sigma*(CDF(ss[zs])-CDF(ss[zf]))*m*ss[theta]^(-sigma)*ss[v]/ss[nf]
    GAM0[7,zs] = -ss[zs]*PDF(ss[zs])*m*ss[theta]^(-sigma)*ss[v]/ss[nf]
    GAM0[7,zf] = ss[zf]*PDF(ss[zf])*m*ss[theta]^(-sigma)*ss[v]/ss[nf]

    #Production function

    GAM0[8,Y] = 1.
    GAM0[8,A] = -1.
    GAM1[8,np] = (1-s)*II(ss[zp])*ss[np]/ss[Y]
    GAM1[8,nf] = rho*(1-xif)*Ez*ss[nf]/ss[Y]
    GAM0[8,zp] = (1-s)*ss[np]*ss[zp]^2*PDF(ss[zp])/ss[Y]
    GAM0[8,zs] = (1-rho)*ss[zs]^2*PDF(ss[zs])*ss[v]*m*ss[theta]^(-sigma)/ss[Y]
    GAM0[8,zf] = rho*ss[zf]^2*PDF(ss[zf])*ss[v]*m*ss[theta]^(-sigma)/ss[Y]
    GAM0[8,v] = -(II(ss[zs]) + rho*(II(ss[zf]) - II(ss[zs])))*ss[v]*m*ss[theta]^(-sigma)/ss[Y]
    GAM0[8,theta] = sigma*(II(ss[zs]) + rho*(II(ss[zf]) - II(ss[zs])))*ss[v]*m*ss[theta]^(-sigma)/ss[Y]

    #NK PC

    GAM0[9,pi] = 1
    GAM0[9,E_pi] = - beta
    GAM0[9,mu] = - 1
    GAM0[9,phi] = - (1-beta*psi)*(1-psi)/psi

    #Taylor rule

    GAM0[10,i] = 1
    GAM0[10,E_pi] = -(1-r_i)*r_pi
    GAM0[10,Y] = -(1-r_i)*r_y
    GAM1[10,i] = r_i
    PSI[10,e_m] = 1

    #Definition of v

    GAM0[11,v] = 1.
    GAM0[11,theta] = -1.
    GAM1[11,np] = -(1-s)*(1-CDF(ss[zp]))*ss[theta]*ss[np]/ss[v]
    GAM1[11,nf] = -(1-xif)*ss[theta]*ss[nf]/ss[v]
    GAM0[11,zp] = -(1-s)*ss[theta]*PDF(ss[zp])*ss[zp]*ss[np]/ss[v]

    #Ressource constraint

    GAM0[12,Y] = 1
    GAM0[12,c] = -ss[c]/ss[Y]
    GAM0[12,g] = -ss[g]/ss[Y]
    GAM0[12,v] = -gamma*ss[v]/ss[Y]

    #Shock processes

    GAM0[13,A] = 1
    GAM1[13,A] = rho_A
    PSI[13,e_A] = 1

    GAM0[14,g] = 1
    GAM1[14,g] = rho_g
    PSI[14,e_g] = 1

    GAM0[15,mu] = 1
    GAM1[15,mu] = rho_mu
    PSI[15,e_mu] = 1

    #Expectation errors

    GAM0[16,c] = 1
    GAM1[16,E_c] = 1
    PPI[16,eta_c] = 1

    GAM0[17,zp] = 1
    GAM1[17,E_zp] = 1
    PPI[17,eta_zp] = 1

    GAM0[18,zf] = 1
    GAM1[18,E_zf] = 1
    PPI[18,eta_zf] = 1

    GAM0[19,theta] = 1
    GAM1[19,E_theta] = 1
    PPI[19,eta_theta] = 1

    GAM0[20,phi] = 1
    GAM1[20,E_phi] = 1
    PPI[20,eta_phi] = 1

    GAM0[21,pi] = 1
    GAM1[21,E_pi] = 1
    PPI[21,eta_pi] = 1

    #Sims

    GG, CC, RR, _, _, _, _, eu, _ = gensysdt(GAM0, GAM1, C, PSI, PPI)

    #Standard Deviations

    stdev = [sig_A,sig_mu,sig_g,sig_m]
    SDX = diagm(stdev)

    #Observables

    Nobs = 4
    ZZ = zeros(Nobs,NY)

    #Aggregate

    ZZ[1,Y] = 1
    ZZ[2,pi] = 1
    ZZ[3,i] = 1

    sigz = param[5]

    #Share of TC in JC

    # ZZ[4,zs] = PDF(ss[zs])*ss[zs]/(CDF(ss[zs])-CDF(ss[zf]))
    # ZZ[4,zf] = -PDF(ss[zf])*ss[zf]/(1-CDF(ss[zf]))/(CDF(ss[zs])-CDF(ss[zf]))
    # println(ZZ[4,:])

    # Unemployment

    ZZ[4,np] = +ss[np]/(ss[np]+ss[nf])
    ZZ[4,nf] = +ss[nf]/(ss[np]+ss[nf])

    return (GG,RR,SDX,ZZ,eu,NY,NNY,NETA,NX)

end
