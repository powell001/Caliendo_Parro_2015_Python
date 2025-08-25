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

    # Sector_Country rows, Country columns

     # create index
    rows1 = [x + "_" + y for x in final for y in Countries]

    # load trade flows, xbilat1993 are Exports
    xbilat1993 = pd.read_csv(r"data\original_data\xbilat1993.txt", sep='\t', header=None, names=Countries)
    xbilat1993.index = rows1

    xbilat1993 = xbilat1993 * 1000

    # zeros for intermediary goods
    rows1 = [x + "_" + y for x in intermediate for y in Countries]
    intermediate_df = pd.DataFrame(0, index = rows1, columns=Countries)

    # combine
    xbilat1993 = pd.concat([xbilat1993, intermediate_df])

    xbilat1993.to_csv("tmp_xbilat1993.csv")

    return xbilat1993

xbilat1993 = bilaterExports()

def Tariffs():

    # Tariffs are per country per sector and applied to each country

    tariffs_1993 = pd.read_csv(r"data\original_data\tariffs1993.txt", sep="\t", header=None, names=Countries)
    rows1 = [x + "_" + y for x in final for y in Countries]
    tariffs_1993.index = rows1

    tariffs_2005 = pd.read_csv(r"data\original_data\tariffs2005.txt", sep="\t", header=None, names=Countries)
    rows1 = [x + "_" + y for x in final for y in Countries]
    tariffs_2005.index = rows1

    ###########################
    # tariffs_1993
    # final product tariffs 1993
    tariffs_ = tariffs_1993/100
    tariffs_ += 1

    # intermediate goods tariffs 1993
    # ? Intermediate goods are not subject to tariffs
    rows1 = [x + "_" + y for x in intermediate for y in Countries]
    tariffs__df = pd.DataFrame(1, index = rows1, columns=Countries)
    
    # combine final and intermediate goods
    tariffs_ = pd.concat([tariffs_, tariffs__df])

    #############################
    # tariffs_2005
    tariffs_policy = tariffs_2005/100
    tariffs_policy += 1

    rows1 = [x + "_" + y for x in intermediate for y in Countries]
    tariffs__df = pd.DataFrame(0, index = rows1, columns=Countries)
    
    # combine final and intermediate goods
    tariffs_policy = pd.concat([tariffs_, tariffs__df])

    # weird
    # tariffs_ is combined final and intermediate goods tariffs for 1993
    # tariffs_policy is combined final and intermediate good tariffs for 2005
    # tariffs__hat is a matrix of all ones
    tariffs_policy = tariffs_
    tariffs__hat = tariffs_policy/tariffs_

    tariffs_.to_csv("tmp_tariffs_.csv")
    tariffs__hat.to_csv("tmp_tariffs__hat.csv")
    tariffs_policy.to_csv("tmp_tariffs__p.csv")

    return tariffs__hat, tariffs_, tariffs_policy

tariffs__hat, tariffs_, tariffs_policy = Tariffs()

# Iceberg model. Exporting country must ship more than is expected to arrive in the importer (a percentage of the exports melt away during transport).  In addition,
# there is the cost of the tariff.  Formula

def readParameters():

    # This is the make matrix for all of the countries and products (rows) against products (columns)
    # Each country stacked upon one another
    # Diagonal values will be high, an industry uses a lot of it's own production as an input.
    # Shares of a product value from itself and other products (inputs)
    IO_share = pd.read_csv(r"data\original_data\IO.txt", sep="\t", header=None, names=products_all)
    IO_share.index = [x + "_" + y for x in Countries for y in products_all]
    IO_share.to_csv("tmp_IO_share.csv")

    # Share of value added for each country and each product
    # A number between 0 and 1, Private is 1
    # Products (rows) and countries (columns)
    # Percentage of value that a sector adds directly to an economy
    Value_Added_Share = pd.read_csv(r"data\original_data\B.txt", sep="\t", header=None, names=Countries)
    Value_Added_Share.columns = Countries
    Value_Added_Share.index = products_all
    Value_Added_Share.to_csv("tmp_Value_Added_Share.csv")

    # Gross Output: is a macroeconomic measure representing the total value of goods and services produced in an economy or 
    # by an industry during an accounting period, including sales to final consumers (which contribute to Gross Domestic Product)
    # and sales to other industries (intermediate inputs). 
    GrossOutput = pd.read_csv(r"data\original_data\GO.txt", sep="\t", header=None, names=Countries)
    GrossOutput.columns = Countries
    GrossOutput.index = products_all
    GrossOutput.to_csv("tmp_GrossOutput.csv")

    # Thetas, dispersion of productivity
    # Twenty values, one for each final industry, however, 8.22 is used for the intermediate goods
    Thetas = pd.read_csv(r"data\original_data\T.txt", sep="\t", header=None)
    Thetas = 1/Thetas
    Thetas_df = pd.Series([1]*20).T
    Thetas_df = Thetas_df *1/8.22
    Thetas = pd.concat([Thetas, Thetas_df])
    Thetas.to_csv("tmp_Thetas.csv")
    # add product names, all 40
    Thetas.index = products_all

    return IO_share, Value_Added_Share, GrossOutput, Thetas

IO_share, Value_Added_Share, GrossOutput, Thetas = readParameters()

def calculateExpenditures(xbilat1993, tariffs_):
    # calculating expenditures
    xbilat1993 = xbilat1993 * tariffs_
    #print("Expenditures \n",xbilat1993)

    return xbilat1993

xbilat = calculateExpenditures(xbilat1993, tariffs_)

def domesticSales(xbilat, tariffs_, GrossOutput):

    #################################################
    # Gross Output = GDP (Value added) + intermediate production
    #################################################

    # domestic sale should be production in the country including exports

    # Domestic sales
    X = pd.DataFrame(0, index=products_all, columns=Countries)

    # tariffs_ is the tariff matrix in values
    xbilat_domestic = xbilat/tariffs_

    xbilat_domestic.to_csv("tmpxbilat_domestic.csv")
    #print("bilat domestic \n", xbilat_domestic)

    # X is the total value of exports for each product per country
    for i in products_all:
        product = [x for x in xbilat_domestic.index if x.startswith(i)]

        # sum of exports
        X.loc[i] = xbilat_domestic.loc[product].sum(axis=0)

    # X is exports
    lst = [X, GrossOutput]

    X.to_csv("tmp_X.csv")
    GrossOutput.to_csv("tmp_GrossOutput.csv")

    # if X expenditure greater than GrossOutput gross output, choose GrossOutput, else choose expenditure
    # if Exports greater than Gross Output?
    # are you buying more or less than producing?

    df1 = pd.concat(lst,keys=products_all).groupby(level=1, sort=False)
    GrossOutput = df1.max()
    #print("GrossOutput: \n", GrossOutput)

    # can exports (adjusted) be greater than gross output?
    # I think some GrossOutput are zero

    df1 = pd.concat(lst,keys=products_all).groupby(level=1, sort=False)
    GrossOutput_tmp = df1.max()

    GrossOutput_tmp.to_csv("tmp_GrossOutput_tmp.csv")

    # Gross Output - Exports
    # Exports are valued at gross value added!!!!!!!!!!
    # Domestic sales are the remaining gross value added output after exports
    ##################################
    domsales = GrossOutput_tmp - X
    domsales.to_csv("tmp_domsales.csv")
    ##################################

    return domsales, domsales.T

domsales, domsales_aux = domesticSales(xbilat, tariffs_, GrossOutput)

# Bilateral trade matrix

def bilateral_trade_matrix(domsales_aux, xbilat):

    rows1 = [x + "_" + y for x in products_all  for y in Countries]

    aux2 = pd.DataFrame(0, index=rows1, columns=Countries)

    sections = []
    for i in products_all:
        # select domestic sales for one product
        country1 = domsales_aux[i]
        product = [x for x in aux2.index if i in x]

        ###########################
        # This calculates NN, so gross output used/consumed in own country
        ###########################
        section = aux2.loc[product, :]

        np.fill_diagonal(section.values, country1.values)
        sections.append(section)

    all_aux = pd.concat(sections)

    ##############
    # combine internal and external trade
    ##############
    xbilat = all_aux + xbilat

    xbilat.to_csv("tmp_xbilat_100.csv")

    return xbilat

xbilat = bilateral_trade_matrix(domsales_aux, xbilat)

def cumulcativeExpenditures(xbilat):

    A = xbilat.T.sum(axis=0)
    #print("A: \n", A)

    ############
    # Cummulative_Expenditures is total expenditures + exports
    ############

    Cummulative_Expenditures = pd.DataFrame(0, index=products_all, columns=Countries)

    for i in products_all:
        product = [x for x in A.index if i in x]
        Cummulative_Expenditures.loc[i,:] = A.loc[product].values

    return Cummulative_Expenditures

Cummulative_Expenditures = cumulcativeExpenditures(xbilat)


def calculateExpenditureShares(xbilat):

    # Calculating Expenditure shares
    Xjn_values = xbilat.sum(axis=1)

    rows1 = [x + "_" + y for x in products_all  for y in Countries]
    Xjn = pd.DataFrame(0, index=rows1, columns=Countries)
    for i in np.arange(N):
        Xjn.iloc[:,i-1] = Xjn_values.values

    ###############
    # Calculating Expenditure shares
    ###############

    ###############
    # xbilat include own consumption
    ###############

    Expenditure_shares = xbilat/Xjn

    Expenditure_shares.to_csv("tmp_Expenditure_shares.csv")

    return Expenditure_shares

Expenditure_shares = calculateExpenditureShares(xbilat)


def calcSuperavits(xbilat, tariffs_):

    # tariffs_ is tariffs
    # xbilat is the bilateral trade matrix, values

    # Calculating superavits
    # are these the values after tariffs are applied?
    # tariffs_ is greater than or equal to 1, so xbilattariffs_ will be less than or equal to xbilat, value
    # so xbilattariffs_ < xbilat for foreign countries, internally, value will be the same

    # xbilat is all expenditures
    # xbilattariffs_ is all expenditures reduced by tariffs, own = 1, others greater than one, so 
    # these are the values after applying tariffs
    xbilattariffs_ = xbilat / tariffs_
    xbilattariffs_.to_csv("tmp_xbilattariffs_.csv")

    # M are imports!
    M = pd.DataFrame(0, index=products_all, columns=Countries)
   
    for i in products_all:
        wantThese = [x for x in xbilattariffs_.index if i in x]
        
        # selecting rows e.g. Agriculture_ARG, sum across all country columns,
        # so imports from each country and the production in own country
        # select all agriculture rows
        sum1 = xbilattariffs_.loc[wantThese, :]
        # M contains the sum of values for each product, e.g. Agriculture on row and sum for each country on the columns
        # The sum of values per sector at foreign prices going to each country
        M.loc[i,:] = sum1.sum(axis=1).T.values

    M.to_csv("tmp_M.csv")
    # E are exports!
    E = pd.DataFrame(0, index=products_all, columns=Countries)

    for i in products_all:
        wantThese = [x for x in xbilattariffs_.index if i in x]
        sum1 = xbilattariffs_.loc[wantThese, :]
        # Exports to each country
        E.loc[i,:] = sum1.sum(axis=0).values


    # What is E, I think domestic expenditures at tariff adjusted prices
    E.to_csv("tmp_E.csv")
    Sn = E.sum(axis=0) - M.sum(axis=0)

    # net exports of tradeable goods
    Sn = Sn.values.reshape(31,1)
   
    return M, E, Sn

M, E, Sn = calcSuperavits(xbilat, tariffs_)


def valueAdded(GrossOutput, Value_Added_Share):
    # Calculating Value Added
    VAjn = GrossOutput * Value_Added_Share
    VAn = VAjn.sum()
      
    VAn = VAn.values.reshape(31,1)  

    return VAn

VAn =valueAdded(GrossOutput, Value_Added_Share)

def moreValAdded(Cummulative_Expenditures,IO_share,Value_Added_Share,E):

    # Cummulative_Expenditures cummulative expenditures
    # IO_share, IO table, intermediate goods, shares
    # Value_Added_Share value added, shares
    # E domestic sales

    df1 = pd.DataFrame(0, index=products_all, columns=Countries)
    for i in Countries:
        wantThese = [x for x in IO_share.index if i in x]

        x1 =  Cummulative_Expenditures.loc[:,i].values.reshape(-1,1)
        x2 = IO_share.loc[wantThese,:]
        x3 = Value_Added_Share.loc[:,i].values.reshape(-1,1)
        x4 = E.loc[:,i].values.reshape(-1,1)
        x5 = (1-Value_Added_Share.loc[:,i].values.reshape(-1,1)) * E.loc[:,i].values.reshape(-1,1)
        x6 = x1 - np.matmul(x2, x5)

        df1.loc[:,i] = x6.values

    return df1
num = moreValAdded(Cummulative_Expenditures,IO_share,Value_Added_Share,E)


def alphas(Cummulative_Expenditures, Expenditure_shares, tariffs_, VAn, Sn):

    F = pd.DataFrame(0, index=products_all, columns=Countries)
    for f in products_all:
        wantThese = [x for x in Expenditure_shares.index if f in x]
        F.loc[f, :] = (Expenditure_shares.loc[wantThese, :]/tariffs_.loc[wantThese, :]).sum(axis=1).values

    # alphas
    a1 = (Cummulative_Expenditures*(1-F)).sum(axis=0).values.reshape(1,31)
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

alphas = alphas(Cummulative_Expenditures, Expenditure_shares, tariffs_, VAn, Sn)
#print("alphas: ", alphas)

#####################################
# Main Equilibrium Function
#####################################
# Definitions of parameters
# tariffs__hat: (1240,31), original tariff position, matrix of ones
# tariffs_policy: (1240,31), new tariff position, matrix of values to one or slightly greater
# alphas: (40,31), not sure what these are
# Thetas (thetas), dispersion of productivity
# Value_Added_Share (40,31), share of value added
# IO_share (1240,40), IO coefficients
# Expenditure_shares (1240,31), expenditure shares
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

def PH(wages_N,tariffs__hat,Thetas,Value_Added_Share,IO_share,Expenditure_shares,J,N,maxit,tol):

    # reformat theta vector
    LThetas = np.repeat(Thetas, repeats=N, axis=0)

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
            wantThese = [x for x in IO_share.index if country in x]

            a1 = np.matmul(Value_Added_Share.loc[:,country].values.reshape(-1,1), lw[i].reshape(1,-1))
            a2 = (1 - Value_Added_Share.loc[:,country]).values.reshape(-1,1)
            a3 = np.matmul((IO_share.loc[wantThese,:]).T, lp[:,i].reshape(-1,1))

            a100 = a1 + (a2 *  a3)

            lc.loc[:,country] = a100.values

        ##print("lc: \n", lc)

        #####################
        c = np.exp(lc)
        x7 = np.repeat(LThetas, repeats=N, axis=1)
        x8 = tariffs__hat**x7
        Expenditure_shares_om = Expenditure_shares * x8
        ######################

        phat = pd.DataFrame(0, index=products_all, columns=Countries)        
       
        for j in np.arange(J):
            for n in np.arange(N):
                 rows1 = n + j*N
                 x9 = Expenditure_shares_om.iloc[rows1,].values.reshape(1,-1)
                 ##print(x9)

                 x10 = c.iloc[j,:].values.reshape(1,-1)
                 ##print(x10)

                 x11 = Thetas.iloc[j].values.reshape(1,-1)
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


def Expenditure_sharesprime(Expenditure_shares, tariffs__hat, c, Thetas, J, N):

    #####
    rows1 = [x + "_" + y for x in products_all  for y in Countries]
    data = np.repeat(Thetas.values.reshape(-1,1), repeats=N, axis=0)
    LThetas = pd.Series(data.flatten(), index=rows1, name="LP")

    #####
    cp = pd.DataFrame(0, index = products_all, columns = Countries)
    for i, country in enumerate(Countries):
        y1 = c.iloc[:,i].values
        y2 = (-1/Thetas.values.flatten())
        cp.iloc[:,i] = y1**y2

    #####
    x0 = np.repeat(LThetas.values.reshape(-1,1), repeats=N, axis=1)
    x1 = tariffs__hat**(-1/x0)
    Expenditure_shares_om = np.multiply(Expenditure_shares.values, x1)
   
    ######
    idx = list(range(0, (J*N), N))

    DD = pd.DataFrame(0, index=rows1, columns=Countries)
    for i in range(0, N):
        idex = [x+i for x in idx]
        DD.iloc[idex, :] = Expenditure_shares_om.iloc[idex, :].values * cp

    ######
    phat = (DD.T.sum(axis=0).T)**(1/LThetas)
    phat = phat.values.reshape(-1,1)


    ######
    Expenditure_sharesp = pd.DataFrame(0, index=rows1, columns=Countries)
    for i in range(0, N):
        x1 = phat**(1/LThetas.values.reshape(-1,1))
        Expenditure_sharesp.iloc[:, i] = DD.iloc[:, i].values.reshape(-1,1) * x1


    ######
    return Expenditure_sharesp

def expenditure(alphas,Value_Added_Share,IO_share,Expenditure_sharesp,tariffs_policy,Fp,VAn,wf0,Sn,J,N):

    IA = pd.DataFrame(0, index=np.arange(J*N), columns=np.arange(J*N)) 
    I_F = 1 - Fp.values

    idx = list(range(0, (J*N), J))

    for i in range(0, N):
        kr = np.kron(alphas.iloc[:,i], I_F[:,i].T).reshape(J,J) #40x40
        IA.iloc[i*J:(i+1)*J, i*J:(i+1)*J] = kr
        #print(IA)
    IA.to_csv("tmp_IA.csv")

    ########

    Pit = Expenditure_sharesp/tariffs_policy

    rows1 = [x + "_" + y for x in products_all  for y in Countries]
    Value_Added_Sharet = 1-Value_Added_Share
    Value_Added_ShareP = pd.DataFrame(0, index=rows1, columns=Countries)

    for i in range(0, J):
        x1 = np.kron(np.ones(N,), Value_Added_Sharet.iloc[i,:]).reshape(N,N)
        x2 = Pit.iloc[i*N:(i+1)*N, :].values.reshape(N,N)
        Value_Added_ShareP.iloc[i*N:(i+1)*N, :] = np.multiply(x1, x2)

    
    #########

    NValue_Added_ShareP = pd.DataFrame(0, index=rows1, columns=Countries)
    NValue_Added_ShareP = NValue_Added_ShareP.T

    for enum1, i in enumerate(Countries):
        allcountries = []
        for enum2, j in enumerate(Countries):
            x1 = Value_Added_ShareP.iloc[enum2::N, enum1]
            x2 = allcountries.append(x1)
        NValue_Added_ShareP.iloc[enum1, :] = np.concatenate(allcountries, axis=0)

    NNValue_Added_ShareP = np.kron(NValue_Added_ShareP.values, np.ones((J, 1))).reshape(J*N, J*N)
   
    # GG
    GG = np.kron(np.ones((1, N)), IO_share.values).reshape(J*N, J*N)

    GP = np.multiply(NNValue_Added_ShareP, GG)   

    OM = np.eye(J*N, J*N) - (GP + IA)

    Vb = np.multiply(alphas,np.kron(np.ones((J,1)),np.multiply(wf0,VAn).T))
    Vb = Vb.values.reshape(J*N,1, order='F')
   
    Value_Added_Shareb = np.multiply(-alphas, np.multiply(Sn, np.ones((1,J))).T)
    Value_Added_Shareb = Value_Added_Shareb.values.reshape(J*N,1, order='F')

    DD1 = np.matmul(np.linalg.inv(OM), Vb)
    DD2 = np.matmul(np.linalg.inv(OM), Value_Added_Shareb)

    PQ = DD1 + DD2
    PQ = PQ.reshape(J, N, order='F')


    return PQ

def LMC(Xp, Expenditure_sharesp, J, N, Value_Added_Share, VAL):

    PQ_vec = Xp.T.reshape(J*N,1, order='F')
    
    rows1 = [x + "_" + y for x in products_all  for y in Countries]
    
    DExpenditure_sharesput = pd.DataFrame(0, index=rows1, columns=Countries)
    for i in range(0, N):
        x1 = Expenditure_sharesp.iloc[:,i].values.reshape(-1,1)
        x2 = PQ_vec
        DExpenditure_sharesput.iloc[:, i] = np.multiply(x1, x2)

    DExpenditure_sharesput.to_csv('tmp_DExpenditure_sharesput.csv', index=False)

    DDExpenditure_sharesput = pd.DataFrame(0, index=products_all, columns=Countries)
    for n in range(0, J):
        x1 = DExpenditure_sharesput.iloc[n*N:(n+1)*N, :].values.reshape(N, -1)
        DDExpenditure_sharesput.iloc[n, :] = x1.sum(axis=0)

    aux4 = np.multiply(Value_Added_Share, DDExpenditure_sharesput)
    aux5 = aux4.sum(axis=0)
    wf0 = np.multiply((1/VAL), aux5.values.reshape(-1,1))
    
    return wf0


def equilibrium(tariffs__hat, tariffs_policy, alphas, Thetas, Value_Added_Share, IO_share, Expenditure_shares, J, N, maxit, tol, VAn, Sn, vfactor):

    wf0 = np.ones((N, 1))
    wfmax = 1
    e = 1

    while (e <= maxit) and (wfmax > tol):

        pf0,c = PH(wf0,tariffs__hat,Thetas,Value_Added_Share,IO_share,Expenditure_shares,J,N,maxit,tol)

        # #print("pf0 :\n", pf0, pf0.shape)
        # #print("c :\n", c)

        # Expenditure_sharesp Calculate trade shares
        Expenditure_sharesp = Expenditure_sharesprime(Expenditure_shares, tariffs__hat,c,Thetas,J,N)
        Expenditure_sharesp_om = Expenditure_sharesp/tariffs_policy
        #print("Expenditure_sharesp_om: \n", Expenditure_sharesp_om, Expenditure_sharesp_om.shape)

        # Fp
        Fp = pd.DataFrame(0, index=products_all, columns=Countries)
        rows1 = list(range(0, (J*N), N))
    
        for enum, i in enumerate(rows1):
            #print("i: ", i)
            Fp.iloc[enum,:] = (Expenditure_sharesp.iloc[i:i+31,:]/tariffs_policy.iloc[i:i+31,:]).T.sum()
        #print("Fp :\n", Fp)

        PQ = expenditure(alphas,Value_Added_Share,IO_share,Expenditure_sharesp,tariffs_policy,Fp,VAn,wf0,Sn,J,N)

        #print("PQ: :", PQ, PQ.shape)

        wf1 = LMC(PQ, Expenditure_sharesp, J, N, Value_Added_Share, VAn)

        # Excess function
        ZW = wf1 - wf0

        PQ_vec = PQ.T.reshape(J*N,1, order='F')

        rows1 = [x + "_" + y for x in products_all  for y in Countries]
        DP = pd.DataFrame(0, index=rows1, columns=Countries)
        for n in range(0, N):
            DP.iloc[:,n] = np.multiply(Expenditure_sharesp_om.iloc[:,n].values.reshape(-1,1), PQ_vec)

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

        return wf0, pf0, PQ, Fp, Expenditure_sharesp, ZW, Snp

##############################
##############################
##############################

wf0, pf0, PQ, Fp, Expenditure_sharesp, ZW, Snp = equilibrium(tariffs__hat, tariffs_policy, alphas, Thetas, Value_Added_Share, IO_share, Expenditure_shares, J, N, maxit, tol, VAn/100000, Sn/100000, vfactor)


# expenditures Xji in long vector: PQ_vec=(X11 X12 X13...)' 
print("wf0: \n", wf0)

PQ_vec = PQ.T.reshape(1, J*N, order='F')
Expenditure_sharesp_om = Expenditure_sharesp/tariffs_policy

xbilattariffs_ = np.multiply(Expenditure_sharesp_om, np.multiply(PQ_vec.T, np.ones(N)))
xbilatp = np.multiply(xbilattariffs_, tariffs_policy)
xbilatp.to_csv("tmp_xbilatpx.csv")

for j in range(J):
    GrossOutput.iloc[j:, :] = xbilattariffs_.iloc[j*N:(j+1)*N, :].sum()

print("GrossOutput: \n", GrossOutput, GrossOutput.shape)

for n in range(J):
    x1 = xbilatp.iloc[n*N:(n+1)*N, :].values.reshape(N,N)
    np.fill_diagonal(x1, 0)
    print(x1)
    xbilatp.iloc[n*N:(n+1)*N, :] = x1
    

pd.DataFrame.to_csv(xbilatp, "xbilatp.csv")
pd.DataFrame.to_csv(Expenditure_sharesp, "Expenditure_sharesp.csv")
pd.DataFrame.to_csv(xbilattariffs_, "xbilattariffs_.csv")
pd.DataFrame.to_csv(alphas, "alphas.csv")
pd.DataFrame.to_csv(GrossOutput, "GrossOutput.csv")
