import pandas as pd
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

os.chdir(r"C:\Users\jpark\vscode\Caliendo_Parro_2015_Python\\")

####################
# Original Code
####################
# https://www.google.com/url?q=https%3A%2F%2Fspinup-000d1a-wp-offload-media.s3.amazonaws.com%2Ffaculty%2Fwp-content%2Fuploads%2Fsites%2F40%2F2019%2F06%2FData_and_Codes_CP.zip&sa=D&sntz=1&usg=AOvVaw1aBDWNSnxgj3WXRaEFwimj


vfactor = -0.2
tol = 1E-07
maxit = 1E+10

J = 40
N = 31

Countries = ['ARG', 'AUS', 'AUT', 'BRA', 'CAN', 'CHL', 'CHN', 'DNK', 'FIN', 'FRA',
             'DEU', 'GRC', 'HUN', 'IND', 'IDN', 'IRL', 'ITA', 'JPN', 'KOR', 'MEX',
             'NLD', 'NZL', 'NOR', 'PRT', 'ZAF', 'ESP', 'SWE', 'TUR', 'GBR', 'USA',
             'ROW'];

products_all = ['Agriculture','Mining','Food','Textile','Wood','Paper','Petroleum','Chemicals','Plastic','Minerals','Basicmetals','Metalproducts','Machineryn.e.c','Office',
            'Electrical','Communication','Medical','Auto','OtherTransport','Othr','Electricity','Construction','Retail','Hotels','LandTransport',
            'WaterTransport','AirTransport','AuxTransport','Post','Finance','RealState','RentingMach','Computer','R&D','OtherBusiness','Public',
            'Education','Health','Otherservices','Private'];

final = ['Agriculture','Mining','Food','Textile','Wood','Paper','Petroleum','Chemicals','Plastic','Minerals','Basicmetals','Metalproducts','Machineryn.e.c','Office',
            'Electrical','Communication','Medical','Auto','OtherTransport','Othr']

intermediate = ['Electricity','Construction','Retail','Hotels','LandTransport','WaterTransport','AirTransport','AuxTransport','Post','Finance','RealState','RentingMach',
                'Computer','R&D','OtherBusiness','Public', 'Education','Health','Otherservices','Private']


def bilaterExports():

    rows1 = [x + "_" + y for x in final for y in Countries]
    # load trade flows
    xbilat1993 = pd.read_csv(r"data\original_data\xbilat1993.txt", sep='\t', header=None, names=Countries)
    xbilat1993.index = rows1

    xbilat1993 = xbilat1993 * 1000

    # zeros for intermediary
    rows1 = [x + "_" + y for x in intermediate for y in Countries]
    intermediate_df = pd.DataFrame(0, index = rows1, columns=Countries)

    # combine
    xbilat1993 = pd.concat([xbilat1993, intermediate_df])

    return xbilat1993

xbilat1993 = bilaterExports()

def tradeFlows():

    tau1993 = pd.read_csv(r"data\original_data\tariffs1993.txt", sep="\t", header=None, names=Countries)
    rows1 = [x + "_" + y for x in final for y in Countries]
    tau1993.index = rows1

    tau2005 = pd.read_csv(r"data\original_data\tariffs2005.txt", sep="\t", header=None, names=Countries)
    rows1 = [x + "_" + y for x in final for y in Countries]
    tau2005.index = rows1

    # tau1993
    tau = tau1993/100
    tau += 1
    rows1 = [x + "_" + y for x in intermediate for y in Countries]
    tau_df = pd.DataFrame(1, index = rows1, columns=Countries)
    # combine
    tau = pd.concat([tau, tau_df])

    # tau2005
    taup = tau2005/100
    taup += 1
    rows1 = [x + "_" + y for x in intermediate for y in Countries]
    tau_df = pd.DataFrame(0, index = rows1, columns=Countries)
    # combine
    taup = pd.concat([tau, tau_df])

    # weird
    taup = tau
    tau_hat = taup/tau

    return tau_hat, tau, taup

tau_hat, tau, taup = tradeFlows()


def readParameters():
    G = pd.read_csv(r"data\original_data\IO.txt", sep="\t", header=None, names=products_all)
    G.index = [x + "_" + y for x in Countries for y in products_all]

    B = pd.read_csv(r"data\original_data\B.txt", sep="\t", header=None, names=Countries)
    B.columns = Countries
    B.index = products_all

    GO = pd.read_csv(r"data\original_data\GO.txt", sep="\t", header=None, names=Countries)
    GO.columns = Countries
    GO.index = products_all

    T = pd.read_csv(r"data\original_data\T.txt", sep="\t", header=None)
    T = 1/T
    T_df = pd.Series([1]*20).T
    T_df = T_df *1/8.22
    T = pd.concat([T, T_df])

    T.index = products_all

    return G, B, GO, T

G, B, GO, T = readParameters()


def calculateExpenditures(xbilat1993, tau):
    # calculating expenditures
    xbilat1993 = xbilat1993 * tau
    #print("Expenditures \n",xbilat1993)

    return xbilat1993

xbilat = calculateExpenditures(xbilat1993, tau)

def domesticSales(xbilat, tau, GO):

    #################################################
    # Gross Output = GDP (Value added) + intermediate production
    #################################################

    # Domestic sales
    X = pd.DataFrame(0, index=products_all, columns=Countries)

    # tau is the tariff matrix in percentage form
    xbilat_domestic = xbilat/tau

    xbilat_domestic.to_csv("tmpxbilat_domestic.csv")
    #print("bilat domestic \n", xbilat_domestic)

    for i in products_all:
        product = [x for x in xbilat_domestic.index if x.startswith(i)]

        X.loc[i] = xbilat_domestic.loc[product].sum(axis=0)

    # is X intermediate goods? I think so
    lst = [X, GO]

    # if X expenditure greater than GO gross output, choose GO, else choose expenditure
    # are you buying more or less than producing?

    df1 = pd.concat(lst,keys=products_all).groupby(level=1, sort=False)
    GO = df1.max()
    #print("GO: \n", GO)

    domsales = GO - X

    #print("domestic sales :\n", domsales)

    return domsales, domsales.T

domsales, domsales_aux = domesticSales(xbilat, tau, GO)

# Bilateral trade matrix

def bilateral_trade_matrix(domsales_aux, xbilat):

    rows1 = [x + "_" + y for x in products_all  for y in Countries]

    aux2 = pd.DataFrame(0, index=rows1, columns=Countries)

    sections = []
    for i in products_all:
        country1 = domsales_aux[i]
        product = [x for x in aux2.index if i in x]

        #print(aux2.loc[product, :])
        section = aux2.loc[product, :]

        np.fill_diagonal(section.values, country1.values)
        sections.append(section)

    all_aux = pd.concat(sections)

    # combine internal and external trade

    xbilat = all_aux + xbilat

    return xbilat

xbilat = bilateral_trade_matrix(domsales_aux, xbilat)

def cumulcativeExpenditures(xbilat):

    A = xbilat.T.sum(axis=0)
    #print("A: \n", A)

    X0 = pd.DataFrame(0, index=products_all, columns=Countries)

    for i in products_all:
        product = [x for x in A.index if i in x]
        X0.loc[i,:] = A.loc[product].values

    return X0

X0 = cumulcativeExpenditures(xbilat)


def calculateExpenditureShares(xbilat):

    # Calculating Expenditure shares
    Xjn_values = xbilat.sum(axis=1)

    rows1 = [x + "_" + y for x in products_all  for y in Countries]
    Xjn = pd.DataFrame(0, index=rows1, columns=Countries)
    for i in np.arange(N):
        Xjn.iloc[:,i-1] = Xjn_values.values

    Din = xbilat/Xjn

    return Din

Din = calculateExpenditureShares(xbilat)


def calcSuperavits(xbilat, tau):

    # Calculating superavits
    xbilattau=xbilat/tau

    M = pd.DataFrame(0, index=products_all, columns=Countries)
   
    for i in products_all:
        wantThese = [x for x in xbilattau.index if i in x]
        sum1 = xbilattau.loc[wantThese, :]
        M.loc[i,:] = sum1.sum(axis=1).T.values
   
    E = pd.DataFrame(0, index=products_all, columns=Countries)

    for i in products_all:
        wantThese = [x for x in xbilattau.index if i in x]
        sum1 = xbilattau.loc[wantThese, :]
        E.loc[i,:] = sum1.sum(axis=0).values

    Sn = E.sum(axis=0) - M.sum(axis=0)

    Sn = Sn.values.reshape(31,1)
   
    return M, E, Sn

M, E, Sn = calcSuperavits(xbilat, tau)


def valueAdded(GO, B):
    # Calculating Value Added
    VAjn = GO * B
    VAn = VAjn.sum()
      
    VAn = VAn.values.reshape(31,1)  

    return VAn

VAn =valueAdded(GO, B)

def moreValAdded(G,B,E):

    df1 = pd.DataFrame(0, index=products_all, columns=Countries)
    for i in Countries:
        wantThese = [x for x in G.index if i in x]

        x1 =  X0.loc[:,i].values.reshape(-1,1)
        x2 = G.loc[wantThese,:]
        x3 = B.loc[:,i].values.reshape(-1,1)
        x4 = E.loc[:,i].values.reshape(-1,1)
        x5 = (1-B.loc[:,i].values.reshape(-1,1)) * E.loc[:,i].values.reshape(-1,1)
        x6 = x1 - np.matmul(x2, x5)

        df1.loc[:,i] = x6.values

    return df1
num = moreValAdded(G,B,E)


def alphas(X0, Din, tau, VAn, Sn):

    F = pd.DataFrame(0, index=products_all, columns=Countries)
    for f in products_all:
        wantThese = [x for x in Din.index if f in x]
        F.loc[f, :] = (Din.loc[wantThese, :]/tau.loc[wantThese, :]).sum(axis=1).values

    # alphas
    a1 = (X0*(1-F)).sum(axis=0).values.reshape(1,31)
    a2 = Sn
    a3 = VAn + a1.T - a2
    a4 = np.repeat(a3.T, repeats=40, axis=0)
    alphas = num/a4

    ####
    alphas[alphas < 0] = 0

    #print("alphas: \n", alphas)

    #alphas = alphas.sum().values.reshape(1,-1)
    #alphas = np.repeat(alphas, repeats=40, axis=0)

    return alphas

alphas = alphas(X0, Din, tau, VAn, Sn)
#print("alphas: ", alphas)

#####################################
# Main Equilibrium Function
#####################################
# Definitions of parameters
# tau_hat: (1240,31), original tariff position, matrix of ones
# taup: (1240,31), new tariff position, matrix of values to one or slightly greater
# alphas: (40,31), not sure what these are
# T (thetas), dispersion of productivity
# B (40,31), share of value added
# G (1240,40), IO coefficients
# Din (1240,31), expenditure shares
# J number of products, 40
# N number of countries, 31
# maxit 1E+10
# tol 1E-07
# VAn./100000: (31,1) value added
# Sn./100000: (31,1)
# vfactor


###################################
### PH
###################################

def PH(wages_N,tau_hat,T,B,G,Din,J,N,maxit,tol):

    # reformat theta vector
    LT = np.repeat(T, repeats=N, axis=0)

    # initizlize vectors of ex-post wages and price factors
    wf0 = wages_N
    pf0 = np.ones((J, N))

 
    pfmax = 1
    it = 1

    while (it <= 2): #and (pfmax > tol):
        
        # calculating log cost
        lw = np.log(wf0)
        lp = np.log(pf0)

        #
        lc = pd.DataFrame(0, index = products_all, columns = Countries)
        for i, country in enumerate(Countries):
            wantThese = [x for x in G.index if country in x]

            a1 = np.matmul(B.loc[:,country].values.reshape(-1,1), lw[i].reshape(1,-1))
            a2 = (1 - B.loc[:,country]).values.reshape(-1,1)
            a3 = np.matmul((G.loc[wantThese,:]).T, lp[:,i].reshape(-1,1))

            a100 = a1 + (a2 *  a3)

            lc.loc[:,country] = a100.values

        ##print("lc: \n", lc)

        #####################
        c = np.exp(lc)
        x7 = np.repeat(LT, repeats=N, axis=1)
        x8 = tau_hat**x7
        Din_om = Din * x8
        ######################

        phat = pd.DataFrame(0, index=products_all, columns=Countries)        
       
        for j in np.arange(J):
            for n in np.arange(N):
                 rows1 = n + j*N
                 x9 = Din_om.iloc[rows1,].values.reshape(1,-1)
                 ##print(x9)

                 x10 = c.iloc[j,:].values.reshape(1,-1)
                 ##print(x10)

                 x11 = T.iloc[j].values.reshape(1,-1)
                 ##print(x11)

                 x12 = (x10**(-1/x11)).T
                 ##print(x12)

                 phat.iloc[j,n] = np.matmul(x9, x12) 
                 ##print(phat)

                 if phat.iloc[j,n] == 0:
                     phat.iloc[j,n] = 1 
                 else:
                     phat.iloc[j,n] = (phat.iloc[j,n])**(-x11)

            if j == J:
                print("j: = \n", j)

        pfdev = np.abs(phat - pf0)
        pf0 = phat.values
        pfmax = pfdev.max().max()
        it += 1

        # #print("pf0: \n", pf0)
        # #print("c: \n", c)
    return pf0, c


def Dinprime(Din, tau_hat, c, T, J, N):

    #####
    rows1 = [x + "_" + y for x in products_all  for y in Countries]
    data = np.repeat(T.values.reshape(-1,1), repeats=N, axis=0)
    LT = pd.Series(data.flatten(), index=rows1, name="LP")

    #####
    cp = pd.DataFrame(0, index = products_all, columns = Countries)
    for i, country in enumerate(Countries):
        y1 = c.iloc[:,i].values
        y2 = (-1/T.values.flatten())
        cp.iloc[:,i] = y1**y2

    #####
    x0 = np.repeat(LT.values.reshape(-1,1), repeats=N, axis=1)
    x1 = tau_hat**(-1/x0)
    Din_om = np.multiply(Din.values, x1)
   
    ######
    idx = list(range(0, (J*N), N))

    DD = pd.DataFrame(0, index=rows1, columns=Countries)
    for i in range(0, N):
        idex = [x+i for x in idx]
        DD.iloc[idex, :] = Din_om.iloc[idex, :].values * cp

    ######
    phat = (DD.T.sum(axis=0).T)**(1/LT)
    phat = phat.values.reshape(-1,1)


    ######
    Dinp = pd.DataFrame(0, index=rows1, columns=Countries)
    for i in range(0, N):
        x1 = phat**(1/LT.values.reshape(-1,1))
        Dinp.iloc[:, i] = DD.iloc[:, i].values.reshape(-1,1) * x1


    ######
    return Dinp

def expenditure(alphas,B,G,Dinp,taup,Fp,VAn,wf0,Sn,J,N):

    IA = pd.DataFrame(0, index=np.arange(J*N), columns=np.arange(J*N)) 
    I_F = 1 - Fp.values

    idx = list(range(0, (J*N), J))

    for i in range(0, N):
        kr = np.kron(alphas.iloc[:,i], I_F[:,i].T).reshape(J,J) #40x40
        IA.iloc[i*J:(i+1)*J, i*J:(i+1)*J] = kr
        #print(IA)
    IA.to_csv("tmp_IA.csv")

    ########

    Pit = Dinp/taup

    rows1 = [x + "_" + y for x in products_all  for y in Countries]
    Bt = 1-B
    BP = pd.DataFrame(0, index=rows1, columns=Countries)


 
    for i in range(0, J):
        x1 = np.kron(np.ones(N,), Bt.iloc[i,:]).reshape(N,N)
        x2 = Pit.iloc[i*N:(i+1)*N, :].values.reshape(N,N)
        BP.iloc[i*N:(i+1)*N, :] = np.multiply(x1, x2)

    
    #########

    NBP = pd.DataFrame(0, index=rows1, columns=Countries)
    NBP = NBP.T

    for enum1, i in enumerate(Countries):
        allcountries = []
        for enum2, j in enumerate(Countries):
            x1 = BP.iloc[enum2::N, enum1]
            x2 = allcountries.append(x1)
        NBP.iloc[enum1, :] = np.concatenate(allcountries, axis=0)

    NNBP = np.kron(NBP.values, np.ones((J, 1))).reshape(J*N, J*N)
   
    # GG
    GG = np.kron(np.ones((1, N)), G.values).reshape(J*N, J*N)

    GP = np.multiply(NNBP, GG)   

    OM = np.eye(J*N, J*N) - (GP + IA)

    Vb = np.multiply(alphas,np.kron(np.ones((J,1)),np.multiply(wf0,VAn).T))
    Vb = Vb.values.reshape(J*N,1, order='F')
   
    Bb = np.multiply(-alphas, np.multiply(Sn, np.ones((1,J))).T)
    Bb = Bb.values.reshape(J*N,1, order='F')

    DD1 = np.matmul(np.linalg.inv(OM), Vb)
    DD2 = np.matmul(np.linalg.inv(OM), Bb)

    PQ = DD1 + DD2
    PQ = PQ.reshape(J, N, order='F')


    return PQ

def LMC(Xp, Dinp, J, N, B, VAL):

    PQ_vec = Xp.T.reshape(J*N,1, order='F')
    
    rows1 = [x + "_" + y for x in products_all  for y in Countries]
    
    DDinput = pd.DataFrame(0, index=rows1, columns=Countries)
    for i in range(0, N):
        x1 = Dinp.iloc[:,i].values.reshape(-1,1)
        x2 = PQ_vec
        DDinput.iloc[:, i] = np.multiply(x1, x2)

    DDinput.to_csv('tmp_DDinput.csv', index=False)

    DDDinput = pd.DataFrame(0, index=products_all, columns=Countries)
    for n in range(0, J):
        x1 = DDinput.iloc[n*N:(n+1)*N, :].values.reshape(N, -1)
        DDDinput.iloc[n, :] = x1.sum(axis=0)

    aux4 = np.multiply(B, DDDinput)
    aux5 = aux4.sum(axis=0)
    wf0 = np.multiply((1/VAL), aux5.values.reshape(-1,1))
    
    return wf0


def equilibrium(tau_hat, taup, alphas, T, B, G, Din, J, N, maxit, tol, VAn, Sn, vfactor):

    wf0 = np.ones((N, 1))
    wfmax = 1
    e = 1

    while (e <= maxit) and (wfmax > tol):

        pf0,c = PH(wf0,tau_hat,T,B,G,Din,J,N,maxit,tol)

        # #print("pf0 :\n", pf0, pf0.shape)
        # #print("c :\n", c)

        # Dinp Calculate trade shares
        Dinp = Dinprime(Din, tau_hat,c,T,J,N)
        Dinp_om = Dinp/taup
        #print("Dinp_om: \n", Dinp_om, Dinp_om.shape)

        # Fp
        Fp = pd.DataFrame(0, index=products_all, columns=Countries)
        rows1 = list(range(0, (J*N), N))
    
        for enum, i in enumerate(rows1):
            #print("i: ", i)
            Fp.iloc[enum,:] = (Dinp.iloc[i:i+31,:]/taup.iloc[i:i+31,:]).T.sum()
        #print("Fp :\n", Fp)

        PQ = expenditure(alphas,B,G,Dinp,taup,Fp,VAn,wf0,Sn,J,N)

        #print("PQ: :", PQ, PQ.shape)

        wf1 = LMC(PQ, Dinp, J, N, B, VAn)

        # Excess function
        ZW = wf1 - wf0

        PQ_vec = PQ.T.reshape(J*N,1, order='F')

        rows1 = [x + "_" + y for x in products_all  for y in Countries]
        DP = pd.DataFrame(0, index=rows1, columns=Countries)
        for n in range(0, N):
            DP.iloc[:,n] = np.multiply(Dinp_om.iloc[:,n].values.reshape(-1,1), PQ_vec)

        LHS = DP.sum(axis=0).T.values.reshape(-1,1)

        # calculate RHS (Imports) trade balance
        PF = np.multiply(PQ, Fp)
        # imports
        RHS = PF.sum(axis=0).values.reshape(-1,1)

        # excess function (trade balance)
        Snp = RHS - LHS + Sn
        ZW2 = -(RHS - LHS + Sn)/(VAn)

        #print("Snp: \n", Snp, Snp.shape)

        #interation factor prices
        wf1 = np.multiply(wf0, (1-vfactor*ZW2/wf0))

        wfmax = np.abs(wf1 - wf0).sum()
        wfmax = np.abs(Snp).sum()

        wfmax0 = wfmax.copy()
        wf0=wf1.copy()

        e += 1

        return wf0, pf0, PQ, Fp, Dinp, ZW, Snp

##############################
##############################
##############################

wf0, pf0, PQ, Fp, Dinp, ZW, Snp = equilibrium(tau_hat, taup, alphas, T, B, G, Din, J, N, maxit, tol, VAn/100000, Sn/100000, vfactor)


# expenditures Xji in long vector: PQ_vec=(X11 X12 X13...)' 
print("wf0: \n", wf0)

PQ_vec = PQ.T.reshape(1, J*N, order='F')
Dinp_om = Dinp/taup

xbilattau = np.multiply(Dinp_om, np.multiply(PQ_vec.T, np.ones(N)))
xbilatp = np.multiply(xbilattau, taup)
xbilatp.to_csv("tmp_xbilatpx.csv")

for j in range(J):
    GO.iloc[j:, :] = xbilattau.iloc[j*N:(j+1)*N, :].sum()

print("GO: \n", GO, GO.shape)

for n in range(J):
    x1 = xbilatp.iloc[n*N:(n+1)*N, :].values.reshape(N,N)
    np.fill_diagonal(x1, 0)
    print(x1)
    xbilatp.iloc[n*N:(n+1)*N, :] = x1
    

pd.DataFrame.to_csv(xbilatp, "xbilatp.csv")
pd.DataFrame.to_csv(Dinp, "Dinp.csv")
pd.DataFrame.to_csv(xbilattau, "xbilattau.csv")
pd.DataFrame.to_csv(alphas, "alphas.csv")
pd.DataFrame.to_csv(GO, "GO.csv")
