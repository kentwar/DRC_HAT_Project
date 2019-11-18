import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SensSpecFuns import *

#### This code is designed to calculate the sensitivity and specificity of
#### the Congo Checche algorithm, DRC algorithm and MiniMobile algorithm
#### under different combinations of diagnostic tests.
#### Used in the Jones,Strong,Fox,Kent,Chatchuea - 2019

## Scenario 1: 50% for active, 42% for passive. A more 'worse case scenario'.
## Scenario 2: 74% for active, 61.2% for passive. A more optimistic scenario.
## Both scenarios have lymph node proportion in non HAT patients as 10% and
## can suggest higher in areas where a high frequency is shown.
## In the paper we disabled optimism/pessimism and ignored passive as this is not currently modelled.

## Import the data
data=pd.read_csv('/home/pkent/Documents/WokeWellness/Coding/algorithmcsv.csv')
data.iloc[:,1:7]=data.iloc[:,1:7]/100


##Worst Case scenario WCNGH=WorstCaseNodesGivenHat
WCNGH=0.5
WCNGNH=0.1
##Optimmistic scenario OCNGH=OptimisticCaseNodesGivenHat
OCNGH=0.74
OCNGNH=0.1
## Create a subset of the data for membership in phases.
## Phasem1 is Phase -1 in the literature
Phasem1=[WCNGH,WCNGNH,OCNGH,OCNGNH]
Phase0=data.loc[data['type']==0]
Phase1=data.loc[data['type']==1]
Phase2=data.loc[data['type']==2]
Phase3=data.loc[data['type']==3]
Phase4=data.loc[data['type']==4]

##Create empty dataframe for results

outputdf1=runmobile1(Phasem1,Phase0,Phase1,Phase2,Phase3)
outputdf2=runmobile2(Phasem1,Phase0,Phase1,Phase2,Phase3)
outputdf3=runminimobile(Phasem1,Phase0,Phase1,Phase2)




### Set The Lab Test
labtests=Phase4.iloc[[1,3],:]

outputdf4=runmobile1LAB(Phasem1,Phase0,Phase1,Phase2,Phase3,labtests)
outputdf5=runmobile2LAB(Phasem1,Phase0,Phase1,Phase2,Phase3,labtests)
outputdf6=runminimobileLAB(Phasem1,Phase0,Phase1,Phase2,labtests)

#This Code reorders the columns to make it prettier
outputdf1 =reorder(outputdf1)
outputdf2 =reorder(outputdf2)
outputdf3 =reorder(outputdf3)
outputdf4 =reorder(outputdf4)
outputdf5 =reorder(outputdf5)
outputdf6 =reorder(outputdf6

## Output data to csv
outputdf1.to_csv('\RealOutput\MobileAlgorithm1_SensSpec.csv')
outputdf2.to_csv('\RealOutput\MobileAlgorithm2_SensSpec.csv')
outputdf3.to_csv('\RealOutput\MinimobileAlgorithm_SensSpec.csv')
outputdf4.to_csv('\RealOutput\MobileAlgorithm1_SensSpecLABS.csv')
outputdf5.to_csv('\RealOutput\MobileAlgorithm2_SensSpecLABS.csv')
outputdf6.to_csv('\RealOutput\MinimobileAlgorithm_SensSpecLABS.csv'))

#This code Plots the Sensitivity and Specificity of the algorithms
def plotdata(data):
    plt.scatter(1-data.iloc[:,5],data.iloc[:,2])
    plt.scatter(1-data.iloc[:,6],data.iloc[:,3])
    plt.scatter(1-data.iloc[:,4],data.iloc[:,1])
    plt.legend(['upper','mean','lower'])
    plt.xlim(0,0.015)
    plt.title('Sensitivity and Specificity for algorithms - DRC alg.')
    plt.xlabel('1-specificity')
    plt.ylabel('sensitivity')
    plt.ylim(0.4,1)
    plt.show()

plotdata(outputdf1)
