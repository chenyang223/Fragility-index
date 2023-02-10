#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 16:29:23 2021

@author: tomorr
"""


import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB


def Data_load(data_name):
    """
    Import data
    """
        
    if data_name == 'BreastCancerCoimbra':
        data = pd.read_excel('Data/Breast Cancer Coimbra Data Set .xlsx')
        x = data.values[:,:-1]
        y = np.array([1 if data.values[i,-1]==1 else -1 for i in range(data.values[:,-1].shape[0])])
        
    elif data_name == 'BreastCancerPrognostic':
        data = pd.read_excel('Data/Breast Cancer Wisconsin Prognostic .xlsx',header=None)
        data.fillna(method= 'ffill', inplace=True)
        x = data.values[:,1:]
        y = np.array([1 if data.values[i,0]=='N' else -1 for i in range(data.values[:,0].shape[0])])

    elif data_name == 'LiverPatient':
        data = pd.read_csv('Data/Indian Liver Patient Dataset (ILPD).csv',engine="python",header=None)
        data[1] = [1 if data.values[i,1] == 'Male' else 0 for i in range(data.values.shape[0])]
        data.fillna(method= 'ffill', inplace=True)
        x = data.values[:,:-1]
        y = np.array([1 if data.values[i,-1]==1 else -1 for i in range(data.values[:,-1].shape[0])])
    
    elif data_name == 'GermanCredit':
        data = pd.read_excel('Data/GermanCredit.xlsx',header=None)
        x = data.values[:,:-1]
        y = np.array([1 if data.values[i,-1] ==1 else -1 for i in range(data.values[:,-1].shape[0])])
    
    elif data_name == 'LiverDisorders':
        data = pd.read_excel('Data/LiverDisorders.xlsx',header=None)
        x = data.values[:,:-1]
        y = np.array([1 if data.values[i,-1] ==1 else -1 for i in range(data.values[:,-1].shape[0])])
    elif data_name=="Australian":
        data = pd.read_csv('Data/Australian Credit Approval.csv')
        x = data.values[:,:-1]
        y = np.array([1 if data.values[i,-1] ==1 else -1 for i in range(data.values[:,-1].shape[0])])
    
    elif data_name=="Diabetes":
        data = pd.read_csv('Data/Diabetes dataset.csv')
        x = data.values[:,:-1]
        y = np.array([1 if data.values[i,-1] ==1 else -1 for i in range(data.values[:,-1].shape[0])])
    
    elif data_name=="Ionosphere":
        data = pd.read_csv('Data/Johns Hopkins University Ionosphere database.csv',header=None)
        x = data.values[:,:-1]
        y = np.array([1 if data.values[i,-1] =='g' else -1 for i in range(data.values[:,-1].shape[0])])
    elif data_name=='SpamBase':
        data = pd.read_csv('Data/SPAM E-MAIL DATABASE ATTRIBUTES.csv',header=None)
        x = data.values[:,:-1]
        y = np.array([1 if data.values[i,-1] ==1 else -1 for i in range(data.values[:,-1].shape[0])])
    elif data_name=='Sonar':
        data = pd.read_csv('Data/Sonar.csv',header=None)
        x = data.values[:,:-1]
        y = np.array([1 if data.values[i,-1] =='R' else -1 for i in range(data.values[:,-1].shape[0])])
    else:
        print('No data')
        
    return x,y



    
def FI_calculation(score,y):
    """
    Calculate the KL-divergence based FI by Algorithm 1
    Input:
        score: score value h(x) of each test sample x.
        y: true class label.
        
    Output:
        value of FI
    """
    
    sample_p = score[y==1]      # positive samples
    sample_n = score[y==-1]      # negative samples
    
    N_p = sample_p.shape[0]
    N_n = sample_n.shape[0]
    
    S = N_p * N_n           #number of samples of ranking errors
    
    error = np.zeros(S)     # ranking error
    for i in range(N_p):
        for j in range(N_n):
            error[i * N_n + j] = sample_n[j] - sample_p[i]
            
            
    def expectation(error,k):
        # calculate the empirical expectation given the value of k
        x = 0
        for i in range(error.shape[0]):
            x += np.exp(error[i]/k)
            
        return x/error.shape[0]
    
            
    if (error<0).sum() == 0:
        return 0, error
    elif error.mean()>0:
        return np.Inf, error
    else:       
        k_min = 10
        e = expectation(error,k_min)
        while (e <= 1) & (k_min>1e-4) :
            k_min = k_min/2
            e = expectation(error,k_min)
            #print(k_min)
            
        if k_min <= 1e-4:
            return 0,error
        else:
            k_max = k_min * 2
            
            while k_max - k_min > 1e-3:
                k = (k_min + k_max) / 2
                e = expectation(error,k)
                if e <= 1:
                    k_max = k
                else:
                    k_min = k
                    
                #print(k_max,k_min)
            
            return k_max, error
    
    
 
def bAUC_calculation(samples, estimate=True):

    """
    Calculte bAUC given the samples of ranking error
    Formulation can be found in Norton M, Uryasev S. Maximization of auc and buffered auc in binary classification[J]. Mathematical Programming, 2019, 174(1): 575-612.
    """

    ################# estimate bPOE by simply sorting samples #################
    if estimate==True:
        N=samples.shape[0]
        sorted_samples=sorted(samples)
    
        cvar=sorted_samples[-1]
        probability_level=1
        for i in range(N):
            probability_level = probability_level - 1/float(N)
            cvar = (cvar*(i) + sorted_samples[-(i+1)])/float(i+1)
            if cvar <=0: break
    
        var=sorted_samples[int( probability_level*N ) ]
        bPOE=1-probability_level
        bAUC=probability_level
        # a=1/float(-var)
        gamma=var
        return bAUC
    ######################Get bAUC exactly by solving an LP#############################

    
    m = gp.Model()
    m.setParam("OutputFlag",0)

    E=[]
 
    for i in range(samples.shape[0]):
        E.append(m.addVar(lb=0,ub=GRB.INFINITY,obj=0,vtype=GRB.CONTINUOUS,name="E"+str(i) ))
    a=m.addVar(lb=0,ub=GRB.INFINITY,obj=0,vtype=GRB.CONTINUOUS,name="a" )  

    m.update()
    m.setObjective( (1/float(samples.shape[0]))*gp.quicksum(E[i] for i in range(samples.shape[0])  ),GRB.MINIMIZE)
    
    m.optimize()
    
    for i in range(samples.shape[0]):
        m.addConstr ( E[i]      >= a * (samples[i]) + 1            )
        m.addConstr ( E[i]      >= 0                               )
    
    m.optimize()
    bPOE=m.getObjective().getValue()
    bAUC=1-bPOE
        
    return bAUC
    
 
def FI_minimization(N,S,data_sample,lb_p,lb_n,ub_p,ub_n,LogToConsole=False):
    
    """
    Train linear classifier by minimizing wasserstein based FI.

    Input:
        N: dimension of features.
        data_sample: samples in the empirical distribution of ranking errors.
        S: number of samples in the empirical distribution of ranking errors.
        lb_p, ub_p: lower bound and upper bound for the features of positive samples.
        lb_n, ub_n: lower bound and upper bound for the features of negativee samples.

        
    Output:
        w_solu: decision variable for the linear classifier
    """
    
    MM = gp.Model()
    MM.Params.LogToConsole=LogToConsole

    
    k = MM.addVar(lb = 0,ub = GRB.INFINITY,vtype=GRB.CONTINUOUS,name="k")
    p1 = MM.addVars(S,N,lb = 0,ub = GRB.INFINITY,vtype=GRB.CONTINUOUS,name="p1")
    p2 = MM.addVars(S,N,lb = 0,ub = GRB.INFINITY,vtype=GRB.CONTINUOUS,name="p2")
    q1 = MM.addVars(S,N,lb = 0,ub = GRB.INFINITY,vtype=GRB.CONTINUOUS,name="q1")
    q2 = MM.addVars(S,N,lb = 0,ub = GRB.INFINITY,vtype=GRB.CONTINUOUS,name="q2")
    
    r = MM.addVars(S,2*N,lb = -GRB.INFINITY,ub = GRB.INFINITY,vtype=GRB.CONTINUOUS,name="r")
    r_abs = MM.addVars(S,2*N,lb = 0,ub = GRB.INFINITY,vtype=GRB.CONTINUOUS,name="r")
    
    w = MM.addVars(N,lb = -GRB.INFINITY,ub = GRB.INFINITY, vtype=GRB.CONTINUOUS,name='w')
    w_abs = MM.addVars(N,lb = 0,ub = GRB.INFINITY, vtype=GRB.CONTINUOUS,name='w')
    
    aux = MM.addVars(S,lb = -GRB.INFINITY,ub = GRB.INFINITY,vtype=GRB.CONTINUOUS,name="aux")
    
    
    MM.setObjective(k,GRB.MINIMIZE)
    
    MM.addConstr( sum( aux[s] for s in range(S) )
                  <= 0 )
    
    MM.addConstrs(sum(- p1[s,n] * lb_p[n] + p2[s,n] * ub_p[n] 
                        - q1[s,n] * lb_n[n] + q2[s,n] * ub_n[n] 
                        for n in range(N))
                    + sum(r[s,n] * data_sample[s,n]
                        for n in range(2*N))
                    == aux[s]
                  for s in range(S))
    
    MM.addConstrs(- p1[s,n] + p2[s,n] 
                  == 
                  - w[n] - r[s,n] 
                  for s in range(S) for n in range(N))
    
    MM.addConstrs(- q1[s,n] + q2[s,n] 
                  ==
                  w[n] - r[s,N+n] 
                  for s in range(S) for n in range(N))
    
    
    MM.addConstrs(r_abs[s,n] == gp.abs_(r[s,n]) for s in range(S) for n in range(2*N))
    MM.addConstrs(r_abs[s,n] <= k for s in range(S) for n in range(2*N))
    
    

    MM.addConstrs(w_abs[n] == gp.abs_(w[n]) for n in range(N))
    MM.addConstr(gp.quicksum(w_abs[n] for n in range(N)) == 1 )

    MM.optimize()

    
    w_solu = np.zeros(N)
    for n in range(N):
        w_solu[n] = w[n].x

    
    return w_solu

    




def bAUC(N,S,data_sample,LogToConsole=False):
    
    """
    Train linear classifier by maximizing bAUC
    Formulation can be found in Norton M, Uryasev S. Maximization of auc and buffered auc in binary classification[J]. Mathematical Programming, 2019, 174(1): 575-612.
    
    Input:
        N: dimension of features.
        data_sample: samples in the empirical distribution of ranking errors.
        S: number of samples in the empirical distribution of ranking errors.
        
    Output:
        w_solu: decision variable for the linear classifier
    """
    
    MM = gp.Model()
    MM.Params.LogToConsole=LogToConsole
    
    
    w = MM.addVars(N,lb = -GRB.INFINITY,ub = GRB.INFINITY, vtype=GRB.CONTINUOUS,name='w')
    w_abs = MM.addVars(N,lb = 0,ub = GRB.INFINITY, vtype=GRB.CONTINUOUS,name='w')

    r = MM.addVars(S,lb = 0,ub = GRB.INFINITY,vtype=GRB.CONTINUOUS,name="r")
    
    MM.setObjective(gp.quicksum(r[s] for s in range(S)),GRB.MINIMIZE)
    
    MM.addConstrs(r[s] >= 
                  sum(w[n]*(data_sample[s,N+n]-data_sample[s,n]) for n in range(N)) +1
                  for s in range(S))
    

    MM.addConstrs(w_abs[n] == gp.abs_(w[n]) for n in range(N))
    MM.addConstr(gp.quicksum(w_abs[n] for n in range(N)) == 1 )
    
    MM.setParam("NonConvex", 2)
    
    MM.optimize()
    
    w_solu = np.zeros(N)
    for n in range(N):
        w_solu[n] = w[n].x
    
    return w_solu

     

def performance(data_name,name,w,N,S_test,data_test):
    """
    For section 5.3 Optimization for linear classifiers.
    Calculating the statistic descriptions of ranking errors.
    
    Output: performance table
    """
    ranking_error = np.zeros(S_test)
    for s in range(S_test):
        ranking_error[s] = w.dot(data_test[s,N:]) - w.dot(data_test[s,:N])
    
    # ranking_error_pos = ranking_error[ranking_error >= 0]
    
    v_prob = np.mean(ranking_error <= 0)
    v_mean = np.mean(ranking_error[ranking_error >= 0])
    v_std = np.std(ranking_error)
    v_var95 = np.quantile(ranking_error, 0.95)
    v_var99 = np.quantile(ranking_error, 0.99)
    v_cvar95 = np.mean(ranking_error[ranking_error>=v_var95])
    v_cvar99 = np.mean(ranking_error[ranking_error>=v_var99])
    
    perf_df = pd.DataFrame({"Data_name":[data_name],
                            "Model_name": [name], 
                            "Probability": [v_prob],
                            "Mean": [v_mean],
                            "Std": [v_std],
                            "VaR%95": [v_var95],
                            "CVaR%95": [v_cvar95],
                            "VaR%99": [v_var99],
                            "CVaR%99": [v_cvar99]})
        
    return perf_df

def calculate_error(classifier, x, y):
    score = classifier.decision_function(x)

    sample_p = score[y==1]      # positive samples
    sample_n = score[y==-1]      # negative samples
    
    N_p = sample_p.shape[0]
    N_n = sample_n.shape[0]
    
    S = N_p * N_n           #number of samples of ranking errors
    
    error = np.zeros(S)     # ranking error
    for i in range(N_p):
        for j in range(N_n):
            error[i * N_n + j] = sample_n[j] - sample_p[i]
    
    return error


def performance_of_error(data_name, name, ranking_error):
    v_prob = np.mean(ranking_error <= 0)
    v_mean = np.mean(ranking_error[ranking_error >= 0])
    v_std = np.std(ranking_error)
    v_var95 = np.quantile(ranking_error, 0.95)
    v_var99 = np.quantile(ranking_error, 0.99)
    v_cvar95 = np.mean(ranking_error[ranking_error>=v_var95])
    v_cvar99 = np.mean(ranking_error[ranking_error>=v_var99])
    
    perf_df = pd.DataFrame({"Data_name":[data_name],
                            "Model_name": [name], 
                            "AUC": [v_prob],
                            "Mean": [v_mean],
                            "Std": [v_std],
                            "VaR%95": [v_var95],
                            "CVaR%95": [v_cvar95],
                            "VaR%99": [v_var99],
                            "CVaR%99": [v_cvar99]})
        
    return perf_df

    