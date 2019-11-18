import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#### This code is designed to calculate the sensitivity and specificity of
#### the Congo Checche algorithm, DRC algorithm and MiniMobile algorithm
#### under different combinations of diagnostic tests.
#### Used in the Jones,Strong,Fox,Kent,Chatchuea - 2019

## Scenario 1: 50% for active, 42% for passive. A more 'worse case scenario'.
## Scenario 2: 74% for active, 61.2% for passive. A more optimistic scenario.
## Both scenarios have lymph node proportion in non HAT patients as 10% and
## can suggest higher in areas where a high frequency is shown.
## In the paper we disabled optimism/pessimism and ignored passive as this is not currently modelled.

### These are general rules for calculating combinations of diagnostic tests
def CAS(sens1,spec1,sens2,spec2):
	## A function for combining the sensitivity and specificity of two tests
	## CAS = Combine.And.Serial. Meaning we are combining tests that  are in serial
	## and that both have to be true to be taken as a positive
	combinedsens=sens1*sens2
	combinedspec=spec1+(1-spec1)*spec2
	return(combinedsens,combinedspec)

def COS(sens1,spec1,sens2,spec2):
	## A function for combining the sensitivity and specificity of two tests
	## COS = Combine.'OR'.Serial Meaning we are combining tests that  are in serial
	## and if one of them is true then it is positive
	combinedsens=sens1+(1-sens1)*sens2
	combinedspec=spec1*spec2
	return(combinedsens,combinedspec)

def CAP(sens1,spec1,sens2,spec2):
	## A function for combining the sensitivity and specificity of two tests
	## CAP = Combine.'AND'.Parallel Meaning we are combining tests that  are in Parallel
	## and if one of them is true then it is positive
	combinedsens=sens1*sens2
	combinedspec=spec1+spec2-(spec1*spec2)
	return(combinedsens,combinedspec)

def COP(sens1,spec1,sens2,spec2):
	## A function for combining the sensitivity and specificity of two tests
	## COP = Combine.'OR'.Parallel Meaning we are combining tests that  are in Parallel
	## and if one of them is true then it is positive
	combinedsens=sens1+sens2-(sens1*sens2)
	combinedspec=spec1*spec2
	return(combinedsens,combinedspec)

##Checce
def original(C,A,B,D,E,F,first,second,mood):
	## A-F are the tests as they appear in order.
	## First and Second are a way of referencing the lower/upper/mean values from the table
	## mood is an indicator of whether to use the optimistic or pessimistic node values

	## We have the algorithm (A and B) OR (C and (D or E or F))
	## we break this into parts:
	## D OR E or F:
	if not E.empty:
		result0 = COS(E.iloc[first],E.iloc[second],F.iloc[first],F.iloc[second])
		result1 = COS(D.iloc[first],D.iloc[second],result0[0],result0[1])
	else:
	## D OR F
		result1 = COS(D.iloc[first],D.iloc[second],F.iloc[first],F.iloc[second])
	## C and (D OR E/ or F)
	result2 = CAS(C.iloc[first],C.iloc[second],result1[0],result1[1])
	## A and B
	if mood == 'optimistic':
		result3 = CAS(A[2],1-A[3],B.iloc[first],B.iloc[second])
	else:
		result3 = CAS(A[0],1-A[1],B.iloc[first],B.iloc[second])
	## (A AND B) OR (C AND (D OR E))
	result = (COS(result3[0],result3[1],result2[0],result2[1]))
	return (result)

def originalLAB(C,A,B,D,E,F,G,first,second,mood):
	## A-F are the tests as they appear in order.
	## First and Second are a way of referencing the lower/upper/mean values from the table
	## mood is an indicator of whether to use the optimistic or pessimistic node values

	## We have the algorith (A and B) OR (C and (D or E or F))
	## we break this into parts:
	resultm1 = COS(F.iloc[first],F.iloc[second],G.iloc[first]*0.7,G.iloc[second]+(1-G.iloc[second])*(1-0.7))
	## D OR E or F:
	if not E.empty:
		result0 = COS(E.iloc[first],E.iloc[second],resultm1[0],resultm1[1])
		result1 = COS(D.iloc[first],D.iloc[second],result0[0],result0[1])
	else:
	## D OR F
		result1 = COS(D.iloc[first],D.iloc[second],F.iloc[first],F.iloc[second])
	## C and (D OR E/ or F)
	result2 = CAS(C.iloc[first],C.iloc[second],result1[0],result1[1])
	## A and B
	if mood == 'optimistic':
		result3 = CAS(A[2],A[3],B.iloc[first],1-B.iloc[second])
	else:
		result3 = CAS(A[0],A[1],B.iloc[first],1-B.iloc[second])
	## (A AND B) OR (C AND (D OR E))
	result = (COS(result3[0],result3[1],result2[0],result2[1]))
	return (result)

##Mobile
def rural(C,A,B,D,E,F,first,second,mood):
	## A-F are the tests as they appear in order.
	## First and Second are a way of referencing the lower/upper/mean values from the table
	## mood is an indicator of whether to use the optimistic or pessimistic node values

	## We have the algorith C and ((A AND B) or (D or E or F))
	## we break this into parts:

	### This part allows for two parasite tests.
	if not E.empty:
		## E or F:
		result1 = COS(E.iloc[first],E.iloc[second],F.iloc[first],F.iloc[second])
		## D or (E OR F)
		result2 = COS(D.iloc[first],D.iloc[second],result1[0],result1[1])

	else:
		result2 = COS(D.iloc[first],D.iloc[second],F.iloc[first],F.iloc[second])

	## A AND B
	if mood == 'optimistic':
		result3 = CAS(A[2],1-A[3],B.iloc[first],B.iloc[second])
	else:
		result3 = CAS(A[0],1-A[1],B.iloc[first],B.iloc[second])
	## (A AND B) OR (D OR (E OR F))
	result4 = COS(result3[0],result3[1],result2[0],result2[1])
	## C AND (A AND B) OR (D OR (E OR F))
	result = (CAS(C.iloc[first],C.iloc[second],result4[0],result4[1]))
	return (result)

def ruralLAB(C,A,B,D,E,F,G,first,second,mood):
	## A-F are the tests as they appear in order.
	## First and Second are a way of referencing the lower/upper/mean values from the table
	## mood is an indicator of whether to use the optimistic or pessimistic node values

	## We have the algorith C and ((A AND B) or (D or E or F Or G))
	## we break this into parts:
	## F or G
	resultm1 = COS(F.iloc[first],F.iloc[second],G.iloc[first]*0.7,G.iloc[second]+(1-G.iloc[second])*(1-0.7))
	### This part allows for two parasite tests.
	if not E.empty:
		## E or (F of G):
		result1 = COS(E.iloc[first],E.iloc[second],resultm1[0],resultm1[1])
		## D or (E OR (F OR G)
		result2 = COS(D.iloc[first],D.iloc[second],result1[0],result1[1])

	else:
		result2 = COS(D.iloc[first],D.iloc[second],resultm1[0],resultm1[1])

	## A AND B
	if mood == 'optimistic':
		result3 = CAS(A[2],A[3],B.iloc[first],1-B.iloc[second])
	else:
		result3 = CAS(A[0],A[1],B.iloc[first],1-B.iloc[second])
	## (A AND B) OR (D OR (E OR F))
	result4 = COS(result3[0],result3[1],result2[0],result2[1])
	## C AND (A AND B) OR (D OR (E OR F))
	result = (CAS(C.iloc[first],C.iloc[second],result4[0],result4[1]))
	return (result)

def minimobile(A,B,C,D,Followup,first,second,mood):
	## A-D are the tests as they appear in order.
	## First and Second are a way of referencing the lower/upper/mean values from the table
	## mood is an indicator of whether to use the optimistic or pessimistic node values

	## We have the algorith A and (Followedup AND ((B AND C) OR (D)))
	## we break this into parts:
	## B and C:
	if mood == 'optimistic':
		result1 = CAS(B[2],1-B[3],C.iloc[first],C.iloc[second])
	else:
		result1 = CAS(B[0],1-B[1],C.iloc[first],C.iloc[second])
	## (B and C) OR D
	result2 = COS(result1[0],result1[1],D.iloc[first],D.iloc[second])
	## FOLLOWED UP AND ((B AND C) OR D))
	result3= CAS(Followup,1-Followup,result2[0],result2[1])
	## A AND ((B AND C) OR D)
	result = (CAS(A.iloc[first],A.iloc[second],result3[0],result3[1]))
	return (result)

def minimobileLAB(A,B,C,D,E,Followup,first,second,mood):
	## A-D are the tests as they appear in order.
	## First and Second are a way of referencing the lower/upper/mean values from the table
	## mood is an indicator of whether to use the optimistic or pessimistic node values

	## We have the algorith A and (Followedup AND ((B AND C) OR (D OR E)))
	## we break this into parts:
	## Lab Tests
	## E or F AND FollowedUp

	## D OR E
	result0 = COS(D.iloc[first],D.iloc[second],E.iloc[first]*0.7,E.iloc[second]+(1-E.iloc[second])*(1-0.7))
	## B and C:
	if mood == 'optimistic':
		result1 = CAS(B[2],1-B[3],C.iloc[first],C.iloc[second])
	else:
		result1 = CAS(B[0],1-B[1],C.iloc[first],C.iloc[second])
	## (B and C) OR D
	result2 = COS(result1[0],result1[1],result0[0],result0[1])
	## FOLLOWED UP AND ((B AND C) OR D))
	result3= CAS(Followup,1-Followup,result2[0],result2[1])
	## A AND ((B AND C) OR D)
	result = (CAS(A.iloc[first],A.iloc[second],result3[0],result3[1]))
	return (result)

### This is the code used for original
def alg(phase1,phasem1,phase0,phase2,phase22,phase3,mood):
	##calculate sensitivity and specificity for a selection of 4 tests ORIGINAL
	lowersens,lowerspec=original(phase1,phasem1,phase0,phase2,phase22,phase3,1,4,mood)
	meansens,meanspec=original(phase1,phasem1,phase0,phase2,phase22,phase3,3,6,mood)
	uppersens,upperspec=original(phase1,phasem1,phase0,phase2,phase22,phase3,2,5,mood)
	return(lowersens,uppersens,meansens,lowerspec,upperspec,meanspec)


### This is the code used for Rural
def alg2(phase1,phasem1,phase0,phase2,phase22,phase3,mood):
	##calculate sensitivity and specificity for a selection of 4 tests RURAL

	lowersens,lowerspec=rural(phase1,phasem1,phase0,phase2,phase22,phase3,1,4,mood)
	meansens,meanspec=rural(phase1,phasem1,phase0,phase2,phase22,phase3,3,6,mood)
	uppersens,upperspec=rural(phase1,phasem1,phase0,phase2,phase22,phase3,2,5,mood)
	return(lowersens,uppersens,meansens,lowerspec,upperspec,meanspec)


def alg3(phase1,phasem1,phase0,phase2,Followup,mood):
	##calculate sensitivity and specificity for a selection of 4 tests MiniMobile
	lowersens,lowerspec=minimobile(phase1,phasem1,phase0,phase2,Followup,1,4,mood)
	meansens,meanspec=minimobile(phase1,phasem1,phase0,phase2,Followup,3,6,mood)
	uppersens,upperspec=minimobile(phase1,phasem1,phase0,phase2,Followup,2,5,mood)
	return(lowersens,uppersens,meansens,lowerspec,upperspec,meanspec)

### This is the code used for original
def algLAB(phase1,phasem1,phase0,phase2,phase22,phase3,phase4,mood):
	##calculate sensitivity and specificity for a selection of 4 tests ORIGINAL
	lowersens,lowerspec=originalLAB(phase1,phasem1,phase0,phase2,phase22,phase3,phase4,1,4,mood)
	meansens,meanspec=originalLAB(phase1,phasem1,phase0,phase2,phase22,phase3,phase4,3,6,mood)
	uppersens,upperspec=originalLAB(phase1,phasem1,phase0,phase2,phase22,phase3,phase4,2,5,mood)
	return(lowersens,uppersens,meansens,lowerspec,upperspec,meanspec)

### This is the code used for Rural
def alg2LAB(phase1,phasem1,phase0,phase2,phase22,phase3,phase4,mood):
	##calculate sensitivity and specificity for a selection of 4 tests RURAL

	lowersens,lowerspec=ruralLAB(phase1,phasem1,phase0,phase2,phase22,phase3,phase4,1,4,mood)
	uppersens,upperspec=ruralLAB(phase1,phasem1,phase0,phase2,phase22,phase3,phase4,2,5,mood)
	meansens,meanspec=ruralLAB(phase1,phasem1,phase0,phase2,phase22,phase3,phase4,3,6,mood)
	return(lowersens,uppersens,meansens,lowerspec,upperspec,meanspec)

def alg3LAB(phase1,phasem1,phase0,phase2,phase4,Followup,mood):
	lowersens,lowerspec=minimobileLAB(phase1,phasem1,phase0,phase2,phase4,Followup,1,4,mood)
	meansens,meanspec=minimobileLAB(phase1,phasem1,phase0,phase2,phase4,Followup,3,6,mood)
	uppersens,upperspec=minimobileLAB(phase1,phasem1,phase0,phase2,phase4,Followup,2,5,mood)
	return(lowersens,uppersens,meansens,lowerspec,upperspec,meanspec)

def runmobile1(Phasem1,Phase0,Phase1,Phase2,Phase3):
	output=pd.DataFrame()
	for iii in range(Phase1.shape[0]):
		for i in range(Phase0.shape[0]):
			for j in range(Phase3.shape[0]):
				for k in range(Phase2.shape[0]):
					for ii in range(1,2):
						if ii == 0:
							mood='pessimistic'
						else:
							mood='optimistic'
						for xxx in range(k,Phase2.shape[0]):
							if Phase2.iloc[k][0]!=Phase2.iloc[xxx][0]:
								Alg=alg(Phase1.iloc[iii],Phasem1,Phase0.iloc[0],Phase2.iloc[k],Phase2.iloc[xxx],Phase3.iloc[j],mood)
								name=str(mood+Phase1.iloc[iii][0])+'+'+str(Phase2.iloc[k][0])+'+'+str(Phase2.iloc[xxx][0])+'+'+str(Phase3.iloc[j][0])
							else:
								Alg=alg(Phase1.iloc[iii],Phasem1,Phase0.iloc[0],Phase2.iloc[k], pd.DataFrame(),Phase3.iloc[j],mood)
								name=str(mood+Phase1.iloc[iii][0])+'+'+str(Phase2.iloc[k][0])+'+'+str(Phase3.iloc[j][0])
							#This just puts the data in a DataFrame
							temp=pd.DataFrame([Alg])
							temp.loc[0,'name']=name
							output=output.append(temp)

	return(output)

def runmobile1LAB(phasem1,phase0,phase1,phase2,phase3,phase4):
	output=pd.DataFrame()
	for labs in range(phase4.shape[0]):
		for iii in range(phase1.shape[0]):
			for i in range(phase0.shape[0]):
				for j in range(phase3.shape[0]):
					for k in range(phase2.shape[0]):
						for ii in range(1,2):
							if ii == 0:
								mood='pessimistic'
							else:
								mood='optimistic'
							for xxx in range(k,phase2.shape[0]):
								if phase2.iloc[k][0]!=phase2.iloc[xxx][0]:
									Alg=algLAB(phase1.iloc[iii],phasem1,phase0.iloc[0],phase2.iloc[k],phase2.iloc[xxx],phase3.iloc[j],phase4.iloc[labs],mood)
									name=str(mood+phase1.iloc[iii][0])+'+'+str(phase2.iloc[k][0])+'+'+str(phase2.iloc[xxx][0])+'+'+str(phase3.iloc[j][0])+'+'+str(phase4.iloc[labs][0])
								else:
									Alg=algLAB(phase1.iloc[iii],phasem1,phase0.iloc[0],phase2.iloc[k], pd.DataFrame(),phase3.iloc[j],phase4.iloc[labs],mood)
									name=str(mood+phase1.iloc[iii][0])+'+'+str(phase2.iloc[k][0])+'+'+str(phase3.iloc[j][0])+'+'+str(phase4.iloc[labs][0])
										#This just puts the data in a DataFrame
								temp=pd.DataFrame([Alg])
								temp.loc[0,'name']=name
								output=output.append(temp)

	return(output)

def runmobile2(Phasem1,Phase0,Phase1,Phase2,Phase3):
	output=pd.DataFrame()
	for iii in range(Phase1.shape[0]):
		for i in range(Phase0.shape[0]):
			for j in range(Phase3.shape[0]):
				for k in range(Phase2.shape[0]):
					for ii in range(1,2):
						if ii == 0:
							mood='pessimistic'
						else:
							mood='optimistic'
						for xxx in range(k,Phase2.shape[0]):
							if Phase2.iloc[k][0]!=Phase2.iloc[xxx][0]:
								Alg=alg2(Phase1.iloc[iii],Phasem1,Phase0.iloc[0],Phase2.iloc[k],Phase2.iloc[xxx],Phase3.iloc[j],mood)
								name=str(mood+Phase1.iloc[iii][0])+'+'+str(Phase2.iloc[k][0])+'+'+str(Phase2.iloc[xxx][0])+'+'+str(Phase3.iloc[j][0])
							else:
								Alg=alg2(Phase1.iloc[iii],Phasem1,Phase0.iloc[0],Phase2.iloc[k], pd.DataFrame(),Phase3.iloc[j],mood)
								name=str(mood+Phase1.iloc[iii][0])+'+'+str(Phase2.iloc[k][0])+'+'+str(Phase3.iloc[j][0])
							#This just puts the data in a DataFrame
							temp=pd.DataFrame([Alg])
							temp.loc[0,'name']=name
							output=output.append(temp)

	return(output)

def runmobile2LAB(phasem1,phase0,phase1,phase2,phase3,phase4):
	output=pd.DataFrame()
	for labs in range(phase4.shape[0]):
		for iii in range(phase1.shape[0]):
			for i in range(phase0.shape[0]):
				for j in range(phase3.shape[0]):
					for k in range(phase2.shape[0]):
						for ii in range(1,2):
							if ii == 0:
								mood='pessimistic'
							else:
								mood='optimistic'
							for xxx in range(k,phase2.shape[0]):
								if phase2.iloc[k][0]!=phase2.iloc[xxx][0]:
									Alg=alg2LAB(phase1.iloc[iii],phasem1,phase0.iloc[0],phase2.iloc[k],phase2.iloc[xxx],phase3.iloc[j],phase4.iloc[labs],mood)
									name=str(mood+phase1.iloc[iii][0])+'+'+str(phase2.iloc[k][0])+'+'+str(phase2.iloc[xxx][0])+'+'+str(phase3.iloc[j][0])+'+'+str(phase4.iloc[labs][0])
								else:
									Alg=alg2LAB(phase1.iloc[iii],phasem1,phase0.iloc[0],phase2.iloc[k], pd.DataFrame(),phase3.iloc[j],phase4.iloc[labs],mood)
									name=str(mood+phase1.iloc[iii][0])+'+'+str(phase2.iloc[k][0])+'+'+str(phase3.iloc[j][0])+'+'+str(phase4.iloc[labs][0])
										#This just puts the data in a DataFrame
								temp=pd.DataFrame([Alg])
								temp.loc[0,'name']=name
								output=output.append(temp)
	return(output)

def runminimobile(Phasem1,Phase0,Phase1,Phase2):
	## empty dataframe
	output=pd.DataFrame()
	levels=[0.4,0.55,0.75,0.9]
	for j in range(Phase2.shape[0]):
		for k in range(len(levels)):
			for ii in range(1,2):
				if ii == 0:
					mood='pessimistic'
				else:
					mood='optimistic'
				Alg=alg3(Phase1.iloc[1],Phasem1,Phase0.iloc[0],Phase2.iloc[j],levels[k],mood)
				name=str(mood+Phase1.iloc[1][0])+'+'+str(Phase2.iloc[j][0])+'+'+str(levels[k])+'followup'
				# This just puts the data in a DataFrame
				temp=pd.DataFrame([Alg])
				temp.loc[0,'name']=name
				output=output.append(temp)
	return(output)

def runminimobileLAB(phasem1,phase0,phase1,phase2,phase4):
	## empty dataframe
	output=pd.DataFrame()
	for labs in range(phase4.shape[0]):
		levels=[0.4,0.55,0.75,0.9]
		for j in range(phase2.shape[0]):
			for k in range(len(levels)):
				for ii in range(1,2):
					if ii == 0:
						mood='pessimistic'
					else:
						mood='optimistic'
					Alg=alg3LAB(phase1.iloc[1],phasem1,phase0.iloc[0],phase2.iloc[j],phase4.iloc[labs],levels[k],mood)
					name=str(mood+phase1.iloc[1][0])+'+'+str(phase2.iloc[j][0])+'+'+str(levels[k])+'followup'+'+'+str(phase4.iloc[labs][0])
				# This just puts the data in a DataFrame
					temp=pd.DataFrame([Alg])
					temp.loc[0,'name']=name
					output=output.append(temp)
	return(output)

def reorder(data):
	cols=['name',0,1,2,3,4,5]
	data=data[cols]
	data.columns=['name','LowerSens','UpperSens','MeanSens','LowerSpec','UpperSpec','MeanSpec']
	return(data)
