import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d
from itertools import chain
import datetime
from dateutil import parser


def calcT(d1, d2):
    date1 = parser.parse(d1)
    date2 = parser.parse(d2)
    diff = date2 - date1
    t=(diff.days/365)
    return t, diff.days

d1="2022-05-12"    
df = pd.read_csv("aapl_df.csv" ) #aapl option data for June expiration

def calcprice(df):
    if df['Volume'].iloc[0]== 21369: #mark modified csv to improve efficiency
        return 0
    price=[]
    und=[]
    div=[]
    r=[]
    hisvol=[]
    time=[]
    day=[]
    for i in range(len(df)):
            mid=(df['Bid'].iloc[i]+df['Ask'].iloc[i])/2
            price.append(mid)
            und.append(143.8)
            div.append(0.0065)
            r.append(0.0256)
            hisvol.append(0.4463)
            time.append(calcT(d1,df['Expdate'].iloc[i])[0])
            day.append(calcT(d1,df['Expdate'].iloc[i])[1])
    df['price']=pd.DataFrame(price)
    df['und']=pd.DataFrame(und)
    df['div']=pd.DataFrame(div)
    df['r']=pd.DataFrame(r)
    df['hisvol']=pd.DataFrame(hisvol)
    df['day']=pd.DataFrame(day)
    df['T']=pd.DataFrame(time)
    df['Volume'].iloc[0]= 21369
    return 0

calcprice(df)
def bisection(f, a, b, tol=1e-3, maxiter=100):
    c = (a+b)*0.5  # Declare c as the midpoint ab
    n = 0  
    while n <= maxiter:
        c = (a+b)*0.5
        if f(c) == 0 or abs(a-b)*0.5 < tol:
            # Root is found or is very close
            return c
        n += 1
        if f(c) < 0:
            a = c
        else:
            b = c                
    return c
 # 3 times per trading day dt 
class Option:
    def __init__(self, r, div, vol, K, T,N,dt,realP):
        self.K = K
        self.T = T         
        self.Smax = 3 * K  #maximum range 
        self.N = N   #steps
        self.dt = dt
        self.S = np.linspace(0, self.Smax, self.N) #delta x in this case 
        self.A = np.zeros((self.N, self.N))
        self.b = np.zeros((self.N, 1))
        self.X = np.maximum(self.K - self.S, 0)
        self.r = r
        self.q = div
        self.sigma = vol
        self.price=realP
        self.tol = 1e-5
        self.kl = 1.2
        self.err = 0
        self.iter = 0

    def solve(self, S0,vol):
        self.solvePDE(vol)
        x = self.S.flatten()
        y = self.X.flatten()
        return np.interp(S0, x, y)

    def solvePDE(self,vol):
        t = self.T
        while t > 0:
            dt = min(t, self.dt)
            self.setCoeff(dt,vol)
            self.solveCN()
            t -= dt

    def setCoeff(self, dt,vol):
        N = self.N
        r = self.r
        q = self.q
        S = self.S
        X = self.X
        sigma = vol
        dS = S[1] - S[0]
        for i in range(0, N-1):
            alpha = 0.25 * dt * (np.square(sigma*S[i]/dS) - (r - q) * S[i]/dS)
            beta = 0.5 * dt * (r + np.square(sigma * S[i]/dS))
            gamma = 0.25 * dt * (np.square(sigma*S[i]/dS) + (r - q) * S[i]/dS)
            if i == 0:
                self.b[i] = X[i] * (1 - beta)
                self.A[i][i] = 1 + beta
            else:
                self.b[i] = alpha * X[i-1] + (1 - beta) * X[i] + gamma * X[i+1]
                self.A[i][i-1] = -alpha
                self.A[i][i] = 1 + beta
                self.A[i][i+1] = -gamma
        self.A[-1][N-4] = -1
        self.A[-1][N-3] = 4
        self.A[-1][N-2] = -5
        self.A[-1][N-1] = 2
        self.b[-1] = 0


    def solveCN(self):
        N = self.N
        ite = 0
        kl = self.kl
        self.err = 1000
        while self.err > self.tol and ite <= 1000:
            ite += 1
            x_old = self.X.copy()
            for i in range(N-1):
                self.X[i] = (1 - kl) * self.X[i] + kl * self.b[i] / self.A[i][i]
                self.X[i] -= self.A[i][i+1] * self.X[i+1] * kl / self.A[i][i]
                self.X[i] -= self.A[i][i-1] * self.X[i-1] * kl / self.A[i][i]
            self.X[N-1] = (1 - kl) * self.X[i] + kl * self.b[i] / self.A[i][i]  #boundary condition

            for j in range(N-4, N):
                self.X[N-1] -= self.A[N-1][j] * self.X[j] * kl / self.A[N-1][N-1]
          
            self.X = np.maximum(self.X, self.K - self.S)
            self.err = np.linalg.norm(x_old - self.X)
            self.iter = ite
class impmodel:

    def __init__(self, S0, r, K, T, div, N, price ):
        self.S = S0
        self.r = r
        self.T = T
        self.K=K
        self.div = div
        self.N = N
        self.price=price

    def optionval(self, K, sigma):
        opt = Option(K=self.K, r=self.r, T=self.T, N=self.N, vol=sigma, div=self.div,dt=(1/365),realP=self.price)
        return opt.solve(self.S,sigma)

    def imp_vol(self):
            f = lambda sigma: \
                self.optionval(self.K, sigma)-\
                self.price
            impv = bisection(f, 0.01, 2.0)
            return impv
N=38
dt=(1/365) #1 time a day
div=0.0065 #dividend
r=0.0256 #interest rate
hisvol=0.4463 #historical vol
estimate=[]
err=[]
impvol=[]
strike=df['Strike'].values.tolist()
day=df['day'].values.tolist()
for i in range(len(df)):
    solver = Option(r, div, hisvol, df['Strike'].iloc[i], df['T'].iloc[i],N,dt,df['price'].iloc[i])
    price = 0
    impmod=impmodel(143.8, r,df['Strike'].iloc[i], df['T'].iloc[i], div, N, df['price'].iloc[i] )
    impv= impmod.imp_vol()
    print("number of options calculated:" + str(i))
    impvol.append(impv)
    estimate.append(price)

df['estimate']=pd.DataFrame(estimate) # constant vol curve assumption, then use bisection to estimate implied vol

# plot reference:https://medium.com/@rasmus.sparre/plot-volatility-surface-in-python-using-5d8b9ad10de9
# initiate figure
fig = plt.figure(figsize=(7,7))
# set projection to 3d
axs = plt.axes(projection="3d")
# use plot_trisurf from mplot3d to plot surface and cm for color scheme
axs.plot_trisurf(strike, day, impvol, cmap=cm.jet)
# change angle
axs.view_init(30, 65)
# add labels
plt.xlabel("Strike")
plt.ylabel("Days to maturity")
plt.title("Volatility Surface for AAPL: IV as a Function of K and T")
plt.show()


print("done")









    
