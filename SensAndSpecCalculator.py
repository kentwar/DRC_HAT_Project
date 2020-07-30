import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools as it


#### This code is designed to calculate the sensitivity and specificity of
#### all possible combinations of diagnostic tests for the treatment of
#### Human African Trypanosimiasis
#### Used in the Jones,Strong,Fox,Kent,Chatchuea - 2019

## Scenario 1: 50% for active, 42% for passive. A more 'worse case scenario'.
## Scenario 2: 74% for active, 61.2% for passive. A more optimistic scenario.
## Both scenarios have lymph node proportion in non HAT patients as 10% and
## can suggest higher in areas where a high frequency is shown.
## In the paper we disabled optimism/pessimism and ignored passive as this is
## not currently modelled.

###############################################################################
############## Notes on source data ###########################################
###############################################################################

## The data originally used for this code is available from (GITHUB LINK)
## It needs to be imported to a pandas Dataframe with the following layout
##
## Column      : Content
##
## 0             String : Name of diagnostic tests
## 1
## 2
## 3             Float  : Mean Sensitivity of test
## 4
## 5
## 6             Float  : Mean Specificity of test
##
## The unlabeled columns can be empty in the dataframe and exist only
## as a possible future work


###############################################################################
############## Code Section One - General Rules ###############################
###############################################################################

### These are general rules for calculating combinations of diagnostic tests


def CAS(A, B):
	'''Function for combining the sensitivity and specificity of two tests
	CAS = Combine.And.Serial. Meaning we are combining tests that  are in serial
	and that both have to be true to be taken as a positive

    Inputs:
    A               : Numpy list    : [sensitivity,specificity]

    Outputs:
    combinedsens    : Integer       : values for sensitivity
    combinedspec    : Integer       : values for specificity
    '''

	combinedsens = A[0] * B[0]
	combinedspec = A[1] + (1 - A[1]) * B[1]

	return(combinedsens, combinedspec)

def COS(A, B):
	'''Function for combining the sensitivity and specificity of two tests
	COS = Combine.'OR'.Serial Meaning we are combining tests that  are in serial
	and if one of them is true then it is positive

    Inputs:
    A               : Numpy list    : [sensitivity,specificity]

    Outputs:
    combinedsens    : Integer       : values for sensitivity
    combinedspec    : Integer       : values for specificity
    '''
	combinedsens = A[0] + (1 - A[0]) * B[0]
	combinedspec = A[1] * B[1]

	return(combinedsens, combinedspec)

def CAP(A, B):
	'''Function for combining the sensitivity and specificity of two tests
	CAP = Combine.'AND'.Parallel Meaning we are combining tests that are in
	Parallel and if both of them is true then it is positive

    Inputs:
    A               : Numpy list    : [sensitivity,specificity]

    Outputs:
    combinedsens    : Integer       : values for sensitivity
    combinedspec    : Integer       : values for specificity
    '''

	combinedsens = A[0] * B[0]
	combinedspec = A[1] + B[1]-(A[1] * B[1])

	return(combinedsens, combinedspec)

def COP(A, B):
	'''Function for combining the sensitivity and specificity of two tests
	COP = Combine.'OR'.Parallel Meaning we are combining tests that are in
	Parallel and if one of them is true then it is positive

    Inputs:
    A               : Numpy list    : [sensitivity,specificity]

    Outputs:
    combinedsens    : Integer       : values for sensitivity
    combinedspec    : Integer       : values for specificity
    '''

	combinedsens = A[0] + B[0] - (A[0] * B[0])
	combinedspec = A[1] * B[1]

	return(combinedsens, combinedspec)

###############################################################################
############## Code Section Two - Path Analysis ###############################
###############################################################################

## The following are all the different possible algorithms constructed from
## combining all possible paths utilising all possible tests

def no_extra_paths(A, B, C, D):
    ''' This calculates the sensitivity of a path constructed without any extra
    paths

    We have the algorithm:  A and ((B and C) or D)

    Inputs:
    A-D             : List          : [sens, spec]

    Outputs:
    result3         : List          : [sens, spec]
    '''

    result1 = CAS(B, C)
    result2 = COS(result1, D)
    result3 = CAS(A, result2)

    return(result3)

def extra_path_1(A, B, C, D):
    ''' This calculates the sensitivity of a path constructed including path1

    We have the algorithm:  (B and C) or (A and D)

    Inputs:
    A-D             : List          : [sens, spec]

    Outputs:
    result3         : List          : [sens, spec]
    '''

    result1 = CAS(B, C)
    result2 = CAS(A, D)
    result3 = COS(result1, result2)

    return(result3)

def extra_path_2(A, B, C, D, E):
    ''' This calculates the sensitivity of a path constructed including path2

    We have the algorithm:  A and ((B and C) or  D or E)

    Inputs:
    A-D             : List          : [sens, spec]

    Outputs:
    result4         : List          : [sens, spec]
    '''

    result1 = CAS(B, C)
    result2 = COS(result1, D)
    result3 = COS(result2, E)
    result4 = CAS(A, result3)

    return(result4)

def extra_path_3(A, B, C, D, F, G):
    ''' This calculates the sensitivity of a path constructed including path3

    We have the algorithm:  A and ((B and C) or  D or (F and G))

    Inputs:
    A-D             : List          : [sens, spec]

    Outputs:
    result5         : List          : [sens, spec]
    '''

    result1 = CAS(B, C)
    result2 = CAS(F, G)
    result3 = COS(result1, D)
    result4 = COS(result3, result2)
    result5 = CAS(A, result4)

    return(result5)

def extra_path_2and3(A, B, C, D, E, F, G):
    ''' This calculates the sensitivity of an algorithm including path 2 & 3

    We have the algorithm:  A and ((B and C) or  D or E or (F and G))

    Inputs:
    A-D             : List          : [sens, spec]

    Outputs:
    result6         : List          : [sens, spec]
    '''

    result1 = CAS(B, C)
    result2 = CAS(F, G)
    result3 = COS(result1, D)
    result4 = COS(result3, E)
    result5 = COS(result4, result2)
    result6 = CAS(A, result5)

    return(result6)

def extra_path_1and2(A, B, C, D, E, F, G):
    ''' This calculates the sensitivity of an algorithm including path 1 & 2

    We have the algorithm:  (B and C) or  (A and (D or E))

    Inputs:
    A-D             : List          : [sens, spec]

    Outputs:
    result6         : List          : [sens, spec]
    '''

    result1 = CAS(B, C)
    result2 = COS(D, E)
    result3 = CAS(A, result2)
    result4 = COS(result1, result3)

    return(result4)

def extra_path_1and3(A, B, C, D, E, F, G):
    ''' This calculates the sensitivity of an algorithm including path 1 & 3

    We have the algorithm:  (B and C) or  (A and (D or (F and G)))

    Inputs:
    A-D             : List          : [sens, spec]

    Outputs:
    result5         : List          : [sens, spec]
    '''


    result1 = CAS(B, C)
    result2 = COS(F, G)
    result3 = COS(D, result2)
    result4 = CAS(A, result3)
    result5 = COS(result1, result4)

    return(result5)

def all_paths(A, B, C, D, E, F, G):
    ''' This calculates the sensitivity of an algorithm including Paths 1 & 2 & 3

    We have the algorithm: (B and C) or  (A and (D or E or (F and G))

    Inputs:
    A-D             : List          : [sens, spec]

    Outputs:
    result6         : List          : [sens, spec]
    '''

    result1 = CAS(B, C)
    result2 = CAS(F, G)
    result3 = COS(result2, E)
    result4 = COS(result3, D)
    result5 = CAS(A, result4)
    result6 = COS(result1, result5)

    return(result6)

###############################################################################
############## Code Section Three - ToolKit ###################################
###############################################################################

def il(list_):
    '''il is an indexlister, returning the number of items in a list as a range

    Input
    list_           : List

    Output          : Range         : returns a range up to the number of items'''

    return(range(len( list_)))

def prep(df, index):
    '''Prep function is neccesary to pluck the items from the Pandas Dataframe

    Inputs
    df              : Pandas Dataframe
    index           : Integer

    Output
    values          : list          : [sens, spec] of the individual test
    test_name       : string        : name of the test
    '''

    df          = df.iloc[index]
    values      = [df.iloc[3],df.iloc[6]]
    test_name   = str(df.iloc[0])
    return(values, test_name)

def rdtcattconflict(i):
    '''A function to test if Catt dilutions and RDT appear in the same algorithm '''
    if i[1] == 1 and i[4] != 3:
        return(True)

###############################################################################
############## Code Section Four - Implementation #############################
###############################################################################

def run_no_extra_paths(A, B, C, D):
    ''' This runs the no_extra_paths algorithm for all possibile combinations
    of tests
    '''

    output = pd.DataFrame()  ## An empty placeholder

    ## Combinations holds all iterations of viable combinations (algorithms)
    combinations = list(it.product(il(A), il(B), il(C), il(D)))

    for i in combinations:
        Bi,Bstr = prep(B, i[1])
        Ci,Cstr = prep(C, i[2])
        Di,Dstr = prep(D, i[3])
        values  = no_extra_paths(A[i[0]], Bi, Ci, Di)
        name    = Bstr +' '+ Cstr + ' ' + Dstr+' NOXP'

        temp    = pd.DataFrame([values])
        temp.loc[0,'Algorithm'] = name
        output  = output.append(temp)
    output.columns  = ['sens', 'spec', 'Algorithm']
    return(output)

def run_extra_path_1(A, B, C, D):
    ''' This runs the extra_path_1 algorithm for all possibile combinations
    of tests
    '''

    output = pd.DataFrame()  ## An empty placeholder

    ## Combinations holds all iterations of viable combinations (algorithms)
    combinations = list(it.product(il(A), il(B), il(C), il(D)))

    for i in combinations:
        Bi,Bstr = prep(B, i[1])
        Ci,Cstr = prep(C, i[2])
        Di,Dstr = prep(D, i[3])

        values  = extra_path_1(A[i[0]], Bi, Ci, Di)
        name    = Bstr +' '+ Cstr + ' ' + Dstr+' XP1'

        temp    = pd.DataFrame([values])
        temp.loc[0,'Algorithm'] = name
        output  = output.append(temp)
    output.columns  = ['sens', 'spec', 'Algorithm']

    return(output)

def run_extra_path_2(A, B, C, D, E):
    ''' This runs the extra_path_2 algorithm for all possibile combinations
    of tests
    '''

    output = pd.DataFrame()  ## An empty placeholder

    ## Combinations holds all iterations of viable combinations (algorithms)
    combinations = list(it.product(il(A), il(B), il(C), il(D), il(E)))

    for i in combinations:
        if not rdtcattconflict(i):
            Bi,Bstr = prep(B, i[1])
            Ci,Cstr = prep(C, i[2])
            Di,Dstr = prep(D, i[3])
            Ei,Estr = prep(E, i[4])

            values  = extra_path_2(A[i[0]], Bi, Ci, Di, Ei)
            name    = Bstr +' '+ Cstr + ' ' + Dstr + ' ' + Estr +' XP2'

            temp    = pd.DataFrame([values])
            temp.loc[0,'Algorithm'] = name
            output  = output.append(temp)
    output.columns  = ['sens', 'spec', 'Algorithm']
    output.drop_duplicates()

    return(output)

def run_extra_path_3(A, B, C, D, F, G):
    ''' This runs the extra_path_3 algorithm for all possibile combinations
    of tests
    '''
    output = pd.DataFrame()  ## An empty placeholder

    ## Combinations holds all iterations of viable combinations (algorithms)
    combinations = list(it.product(il(A), il(B), il(C), il(D), il(F), il(G)))

    for i in combinations:
        Bi,Bstr = prep(B, i[1])
        Ci,Cstr = prep(C, i[2])
        Di,Dstr = prep(D, i[3])
        Fi,Fstr = prep(F, i[4])
        Gi      = [G[i[5]],1-G[i[5]]]
        values  = extra_path_3(A[i[0]], Bi, Ci, Di, Fi, Gi)
        name    = Bstr +' '+ Cstr + ' ' + Dstr + ' '+Fstr+' ' +str(G[i[5]])+' XP3'

        temp    = pd.DataFrame([values])
        temp.loc[0,'Algorithm'] = name
        output  = output.append(temp)
    output.columns  = ['sens', 'spec', 'Algorithm']
    output.drop_duplicates()

    return(output)

def run_extra_path_2and3(A, B, C, D, E, F, G):
    ''' This runs the extra_path_2and3 algorithm for all possibile combinations
    of tests
    '''

    output = pd.DataFrame()  ## An empty placeholder

    ## Combinations holds all iterations of viable combinations (algorithms)
    combinations = list(it.product(il(A), il(B), il(C), il(D), il(E), il(F), il(G)))

    for i in combinations:
        if not rdtcattconflict(i):
            Bi,Bstr = prep(B, i[1])
            Ci,Cstr = prep(C, i[2])
            Di,Dstr = prep(D, i[3])
            Ei,Estr = prep(E, i[4])
            Fi,Fstr = prep(F, i[5])
            Gi      = [G[i[6]],1-G[i[6]]]
            values  = extra_path_2and3(A[i[0]], Bi, Ci, Di, Ei, Fi, Gi)
            name    = Bstr +' '+ Cstr + ' ' + Dstr + ' '+ Estr + ' '+Fstr+' ' +str(G[i[6]])+' XP23'

            temp    = pd.DataFrame([values])
            temp.loc[0,'Algorithm'] = name
            output  = output.append(temp)
    output.columns  = ['sens', 'spec', 'Algorithm']
    output.drop_duplicates()

    return(output)

def run_extra_path_1and2(A, B, C, D, E, F, G):
    ''' This runs the extra_path_1and2 algorithm for all possibile combinations
    of tests
    '''

    output = pd.DataFrame()  ## An empty placeholder

    ## Combinations holds all iterations of viable combinations (algorithms)
    combinations = list(it.product(il(A), il(B), il(C), il(D), il(E), il(F), il(G)))

    for i in combinations:
        if not rdtcattconflict(i):
            Bi,Bstr = prep(B, i[1])
            Ci,Cstr = prep(C, i[2])
            Di,Dstr = prep(D, i[3])
            Ei,Estr = prep(E, i[4])
            Fi,Fstr = prep(F, i[5])
            Gi      = [G[i[6]],1-G[i[6]]]
            values  = extra_path_1and2(A[i[0]], Bi, Ci, Di, Ei, Fi, Gi)
            name    = Bstr +' '+ Cstr + ' ' + Dstr + ' '+ Estr + ' '+Fstr+' ' +str(G[i[6]])+' XP12'

            temp    = pd.DataFrame([values])
            temp.loc[0,'Algorithm'] = name
            output  = output.append(temp)
    output.columns  = ['sens', 'spec', 'Algorithm']
    output.drop_duplicates()

    return(output)

def run_extra_path_1and3(A, B, C, D, E, F, G):
    ''' This runs the extra_path_1and3 algorithm for all possibile combinations
    of tests
    '''

    output = pd.DataFrame()  ## An empty placeholder

    ## Combinations holds all iterations of viable combinations (algorithms)
    combinations = list(it.product(il(A), il(B), il(C), il(D), il(E), il(F), il(G)))

    for i in combinations:
        if not rdtcattconflict(i):
            Bi,Bstr = prep(B, i[1])
            Ci,Cstr = prep(C, i[2])
            Di,Dstr = prep(D, i[3])
            Ei,Estr = prep(E, i[4])
            Fi,Fstr = prep(F, i[5])
            Gi      = [G[i[6]],1-G[i[6]]]
            values  = extra_path_1and3(A[i[0]], Bi, Ci, Di, Ei, Fi, Gi)
            name    = Bstr +' '+ Cstr + ' ' + Dstr + ' '+ Estr + ' '+Fstr+' ' +str(G[i[6]])+' XP13'

            temp    = pd.DataFrame([values])
            temp.loc[0,'Algorithm'] = name
            output  = output.append(temp)
    output.columns  = ['sens', 'spec', 'Algorithm']
    output.drop_duplicates()

    return(output)

def run_allpaths(A, B, C, D, E, F, G):
    ''' This runs the all_paths algorithm for all possibile combinations
    of tests
    '''

    output = pd.DataFrame()  ## An empty placeholder

    ## Combinations holds all iterations of viable combinations (algorithms)
    combinations = list(it.product(il(A), il(B), il(C), il(D), il(E), il(F), il(G)))

    for i in combinations:
        if not rdtcattconflict(i):
            Bi,Bstr = prep(B, i[1])
            Ci,Cstr = prep(C, i[2])
            Di,Dstr = prep(D, i[3])
            Ei,Estr = prep(E, i[4])
            Fi,Fstr = prep(F, i[5])
            Gi      = [G[i[6]],1-G[i[6]]]
            values  = all_paths(A[i[0]], Bi, Ci, Di, Ei, Fi, Gi)
            name    = Bstr +' '+ Cstr + ' ' + Dstr + ' '+ Estr + ' '+Fstr+' ' +str(G[i[6]])+' XP123'

            temp    = pd.DataFrame([values])
            temp.loc[0,'Algorithm'] = name
            output  = output.append(temp)
    output.columns  = ['sens', 'spec', 'Algorithm']
    output.drop_duplicates()

    return(output)


###############################################################################
############## Code Section Five - Testing Area ##############################
###############################################################################

## Ignore this section if importing the functions.

if __name__ == '__main__':
    #######
    ## Import the data
    data=pd.read_csv('algorithmcsv.csv')
    data.iloc[:,1:7]=data.iloc[:,1:7]/100
    data
    ##Worst Case scenario WCNGH=WorstCaseNodesGivenHat
    WCNGH   = 0.5
    WCNGNH  = 0.1
    ##Optimistic scenario OCNGH=OptimisticCaseNodesGivenHat
    OCNGH   = 0.74
    OCNGNH  = 0.1
    ## Create a subset of the data for membership in phases.
    ## Phasem1 is Phase -1 in the literature
    A   = [[OCNGH,OCNGNH]]
    B   = data.loc[data['type'] == 1]
    C   = data.loc[data['type'] == 0]
    D   = data.loc[data['type'] == 2]
    E   = data.loc[data['type'] == 3]
    F   = data.loc[data['type'] == 4]
    G   = [0.1 ,0.25]
    ########


    run_no_extra_paths(A, B, C, D)
    run_extra_path_1(A,B,C,D)
    run_extra_path_2(A,B,C,D,E)
    run_extra_path_3(A,B,C,D,F,G)
    run_extra_path_2and3(A,B,C,D,E,F,G)
    run_extra_path_1and2(A,B,C,D,E,F,G)
    run_extra_path_1and3(A,B,C,D,E,F,G)
    run_allpaths(A,B,C,D,E,F,G)
