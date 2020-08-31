import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import seaborn as sns
import datetime
from dateutil.relativedelta import relativedelta

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
#from scipy.stats import gmean

import cvxpy as cp

import empyrical as emp
import time
from tabulate import tabulate



from multiprocessing import Pool
import os


def cvx_opt(df):
    # convert sharpe maximization to standard quadratic program according to:
    # http://people.stat.sc.edu/sshen/events/backtesting/reference/maximizing%20the%20sharpe%20ratio.pdf
    
    # solve QP using cvxpy
    
    # define avg. returns and cov matrix
    u = np.asmatrix(df.values.mean(axis=0)).reshape(3,1)
    cov = np.cov(df.values.T)
    
    # define adjusted constraint matrix
    # min weight of each asset = .1
    A = np.asmatrix([[1,0,0],[0,1,0],[0,0,1]])
    bounds = np.asmatrix([0.1,.1,0.1])
    A_mod = A - bounds.T

    y = cp.Variable(len(cov))
    funct = cp.quad_form(y, cov)
    prob = cp.Problem(cp.Minimize(funct), 
                   [y@u == 1, 
                    y >= 0,
                    A_mod@y.T >= 0 # additional linear constraints
                   ])
    r = prob.solve()
    
    if np.isinf(r):
        x = np.full((len(cov),),1/len(cov)) # equal weight, inf occurs if all mean returns are negative
    else:
        x = y.value/y.value.sum()

    return x

def cvx_opt_minv(df):
    
    # define avg. returns and cov matrix
    u = np.asmatrix(df.values.mean(axis=0)).reshape(3,1)
    cov = np.cov(df.values.T)
    
    #set up bounds
    A = np.identity(len(cov))
    bounds = np.array([0.1,0.1,0.1])
    
    
    w = cp.Variable(len(cov)) 
    risk = cp.quad_form(w, cov)
    prob = cp.Problem(cp.Minimize(risk), # different objective function
                   [cp.sum(w) == 1, 
                    w >= 0,
                   A@w.T >= bounds])

    
    
    r = prob.solve()
    
    if np.isinf(r):
        x = np.full((len(cov),),1/len(cov)) # equal weight, inf occurs if all mean returns are negative
    else:
        x = w.value

    return x


df_feat_train = pd.read_csv("../data/processed_features_train_exp.csv",index_col=0,parse_dates=True)
returns_train = pd.read_csv("../data/processed_returns_train_exp.csv",index_col=0,parse_dates=True)

df_feat_test = pd.read_csv("../data/processed_features_test_exp.csv",index_col=0,parse_dates=True)
returns_test = pd.read_csv("../data/processed_returns_test_exp.csv",index_col=0,parse_dates=True)

max_r = returns_train['Stock_Returns'].rolling(21).std().quantile(.95)
max_b = returns_train['Bond_Returns'].rolling(21).std().quantile(.95)
max_c = returns_train['Commodity_Returns'].rolling(21).std().quantile(.95)

class port_opt:
    
    def __init__(self,model):
        self.model = model
        self.wts = []
        self.wt0 = np.array([[.6,.3,.1]]).reshape(3,1)
        self.wts_opt = None
        self.OptimizeFunct_1 = None
        self.OptimizeFunct_2 = None
        self.d = []
        self.dst =[]
        self.nearest = None
        self.rets = None
        self.stds = None
        self.measures = None
        self.all_rets = None
        self.dailyM = None
        
        
        
    def predict_port(self,df_feat_train,df_feat_test,returns_train,returns_test,verbose=False):
        
        for i in range(len(df_feat_test)-1):
            self.dst, nearest = self.model.kneighbors([df_feat_test.values[i]],return_distance=True)

            # we should get weights from similar period + 1 to match what we are trying to predict 
            mask = returns_train.index.strftime('%m-%Y').isin(df_feat_train.iloc[np.fmin(nearest+1,np.full((self.model.n_neighbors),[len(df_feat_train)-1]))[0],:].index.strftime('%m-%Y'))
            f = returns_train[mask]

            # check if greater than max std
            if np.any(returns_test.loc[df_feat_test.index[i].strftime('%m-%Y')].std() > [max_r,99,max_c]):
                if verbose:
                    print('min var: {}'.format(i))
                wt = self.OptimizeFunct_2(f)
            else:
                wt = self.OptimizeFunct_1(f)
            self.d.append(self.dst[0])
            self.wts.append(wt)
        self.d = np.array(self.d)
        self.wts = np.array(self.wts)
        return None
        
    def calc_results(self):
        self.rets=[]
        self.stds=[]
        self.measures=[]
        self.all_rets=np.empty((0,3))
        self.dailyM=None

        for i in range(len(self.wts)):
            wts = np.asarray(self.wts[i]).reshape(3,1)
            next_month = df_feat_test.index[i] + relativedelta(months=1)
            next_month = datetime.datetime.strftime(next_month,'%m-%Y')

            ret = np.asarray(returns_test.loc[next_month].mean()).dot(wts)
            std = np.sqrt(wts.T.dot(returns_test.loc[next_month].cov().values).dot(wts))
            shrp = ret/std

            ret2 = returns_test.loc[next_month].mean().dot(self.wt0)
            std2 = np.sqrt(self.wt0.T.dot(returns_test.loc[next_month].cov().values).dot(self.wt0))
            shrp2 = ret2/std2

            ret3 = returns_test.loc[next_month].mean().dot(self.wts_opt)
            std3 = np.sqrt(self.wts_opt.T.dot(returns_test.loc[next_month].cov().values).dot(self.wts_opt))
            shrp3 = ret3/std3

            self.rets.append((ret[0],ret2[0],ret3))
            self.stds.append((std[0,0],std2[0,0],std3))
            self.measures.append((shrp[0,0],shrp2[0,0],shrp3))
            self.all_rets = np.concatenate((self.all_rets,np.column_stack([returns_test.loc[next_month].values.dot(wts),returns_test.loc[next_month].values.dot(self.wt0),returns_test.loc[next_month].values.dot(self.wts_opt)])))


        self.rets = np.array(self.rets)
        self.stds = np.array(self.stds)
        self.measures = np.array(self.measures)
        
        self.dailyM = np.mean(self.all_rets,axis=0)/np.std(self.all_rets,axis=0)
        
        return None
    
    def summary(self):
        
        strategy_list = ['','Dynamic Algo','Base: {}'.format(self.wt0.flatten()),'Training_Opt_Static']

        print(tabulate(
                [strategy_list,
                 ['CAGR',*list(emp.cagr(self.all_rets))],
                 ['Returns',*list(np.mean(self.all_rets,axis=0)*252)],
                 ['Std',*list(np.std(self.all_rets,axis=0)*np.sqrt(252))],
                 ['Sharpe',*list(self.dailyM*np.sqrt(252))],
                 ['Max Drawdown',*list(emp.max_drawdown(self.all_rets))],
                 ['Sortino',*list(emp.sortino_ratio(self.all_rets))],
                 ['Calmar',emp.calmar_ratio(self.all_rets[:,0]),emp.calmar_ratio(self.all_rets[:,1]),emp.calmar_ratio(self.all_rets[:,2])]
                ],
                headers="firstrow",tablefmt="github"))
        return None
    
    def plots(self):
        
        
        fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1,figsize=(16,28))
        

        l = len((np.insert((self.all_rets+1).cumprod(axis=0),0,1,axis=0)))
        ax1.plot(pd.DataFrame(index=returns_test.iloc[-l:].index,data=(np.insert((self.all_rets+1).cumprod(axis=0),0,1,axis=0))))
        ax1.set_title('Growth of a dollar')
        ax1.legend(loc='best',labels=['Dynamic Algo','Base: {}'.format(self.wt0.flatten()),'Training_Opt_Static'])

        l = len(emp.roll_max_drawdown(self.all_rets[:,0], window=252*1))

        datadf = pd.DataFrame({0:emp.roll_max_drawdown(self.all_rets[:,0], window=252*1),
                                1:emp.roll_max_drawdown(self.all_rets[:,1], window=252*1),
                                2:emp.roll_max_drawdown(self.all_rets[:,2], window=252*1)},
                             index = returns_test.iloc[-l:].index )

        ax2.plot(datadf)
        ax2.legend(loc='best',labels=['Dynamic Algo','Base: {}'.format(self.wt0.flatten()),'Training_Opt_Static'])
        ax2.set_title('Rolling 1y Max Drawdown')
        
        ax3.plot((pd.DataFrame(self.all_rets,index=returns_test.iloc[-len(self.all_rets):].index).rolling(252*3).apply(lambda x: x.mean()/x.std())*np.sqrt(252)).dropna())
        ax3.legend(labels=['Dynamic Algo','Base: {}'.format(self.wt0.flatten()),'Training_Opt_Static'])
        ax3.set_title('Rolling 3y Sharpe')
        
        r0 = pd.DataFrame(self.wts,index=df_feat_test.iloc[-len(self.wts):].index)
        ax4.stackplot(r0.index,r0.T)
        ax4.legend(loc='center right',labels=['Stock','Bond','Commodity'],bbox_to_anchor=(0,0))
        ax4.set_title('Dynamic Weights')
        
        return None

def run_multi(i):
    final = port_opt(NearestNeighbors())
    final.model.n_neighbors=i
    final.OptimizeFunct_1 = cvx_opt
    final.OptimizeFunct_2 = cvx_opt_minv
    final.wts_opt = final.OptimizeFunct_1(returns_train)
    final.model.fit(df_feat_train.values)
    final.predict_port(df_feat_train,df_feat_test,returns_train,returns_test)
    final.calc_results()
    return (i,final.dailyM*np.sqrt(252))