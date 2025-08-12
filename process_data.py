import pandas as pd
import os
import numpy as np

os.chdir(r"C:\Users\jpark\vscode\Caliendo_Parro_2015_Python\\")

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

print(len(Countries), len(final), 31*40)


def bilaterExports():

    rows1 = [x + "_" + y for x in final for y in Countries]
    # load trade flows
    xbilat1993 = pd.read_csv(r"data\original_data\xbilat1993.txt", sep='\t', header=None, names=Countries)
    xbilat1993.index = rows1
    print(xbilat1993)
    xbilat1993.to_csv("tmp.csv")

    xbilat1993 = xbilat1993 * 1000

    # zeros for intermediary
    rows1 = [x + "_" + y for x in intermediate for y in Countries]
    intermediate_df = pd.DataFrame(0, index = rows1, columns=Countries)

    # combine
    xbilat1993 = pd.concat([xbilat1993, intermediate_df])

    print(xbilat1993)

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

    print(tau_hat)

    return tau_hat, tau

tau_hat, tau = tradeFlows()

def readParameters():
    G = pd.read_csv(r"data\original_data\IO.txt", sep="\t", header=None, names=products_all)
    G.index = [x + "_" + y for x in Countries for y in products_all]
    print("(G) IO Table: \n", G)

    B = pd.read_csv(r"data\original_data\B.txt", sep="\t", header=None, names=Countries)
    B.columns = Countries
    B.index = products_all
    print("B Table: \n", B)

    GO = pd.read_csv(r"data\original_data\GO.txt", sep="\t", header=None, names=Countries)
    GO.columns = Countries
    GO.index = products_all
    print("GO Table: \n", GO)

    T = pd.read_csv(r"data\original_data\T.txt", sep="\t", header=None)
    T = 1/T
    T_df = pd.Series([1]*20).T
    T_df = T_df *1/8.22
    T = pd.concat([T, T_df])

    T.index = products_all
    print("T Table: \n", T)

    return G, B, GO, T

G, B, GO, T = readParameters()


def calculateExpenditures(xbilat1993, tau):
    # calculating expenditures
    xbilat1993 = xbilat1993 * tau
    print("Expenditures \n",xbilat1993)

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
    print("bilat domestic \n", xbilat_domestic)

    for i in products_all:
        product = [x for x in xbilat_domestic.index if x.startswith(i)]

        X.loc[i] = xbilat_domestic.loc[product].sum(axis=0)

    # is X intermediate goods? I think so
    lst = [X, GO]

    # if X expenditure greater than GO gross output, choose GO, else choose expenditure
    # are you buying more or less than producing?

    df1 = pd.concat(lst,keys=products_all).groupby(level=1, sort=False)
    GO = df1.max()
    domsales = GO - X

    print("domestic sales :\n", domsales)

    return domsales, domsales.T

domsales, domsales_aux = domesticSales(xbilat, tau, GO)

print(domsales_aux)

# Bilateral trade matrix

def bilateral_trade_matrix(domsales_aux, xbilat):

    rows1 = [x + "_" + y for x in products_all  for y in Countries]

    aux2 = pd.DataFrame(0, index=rows1, columns=Countries)

    sections = []
    for i in products_all:
        country1 = domsales_aux[i]
        product = [x for x in aux2.index if i in x]

        print(aux2.loc[product, :])
        section = aux2.loc[product, :]

        np.fill_diagonal(section.values, country1.values)
        sections.append(section)

    all_aux = pd.concat(sections)

    all_aux.to_csv("tmp_all_aux.csv")

    # combine internal and external trade
    print(all_aux.shape, xbilat.shape)

    xbilat = all_aux + xbilat

    return xbilat

xbilat = bilateral_trade_matrix(domsales_aux, xbilat)
xbilat.to_csv("tmp_xbilat.csv")

def cumulcativeExpenditures(xbilat):

    xbilat = bilateral_trade_matrix(domsales_aux, xbilat)

    A = xbilat.T.sum()

    X0 = pd.DataFrame(0, index=products_all, columns=Countries)

    for i in products_all:
        product = [x for x in A.index if i in x]
        X0.loc[i,:] = A.loc[product].values
    print(X0)

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
print(Din)


def calcSuperavits(xbilat, tau):

    # Calculating superavits
    xbilattau=xbilat/tau

    M = pd.DataFrame(0, index=products_all, columns=Countries)
   
    for i in products_all:
        wantThese = [x for x in xbilattau.index if i in x]
        sum1 = xbilattau.loc[wantThese, :]
        M.loc[i,:] = sum1.sum(axis=1).T.values
    print(M)

    E = pd.DataFrame(0, index=products_all, columns=Countries)

    for i in products_all:
        wantThese = [x for x in xbilattau.index if i in x]
        sum1 = xbilattau.loc[wantThese, :]
        E.loc[i,:] = sum1.sum(axis=0).values
    print(E)

calcSuperavits(xbilat, tau)