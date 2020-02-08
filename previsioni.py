import numpy as np
import pandas as pd
#from scipy import stats

def calcoloProbconTeta(testDF,collPar,tetaKDf,nTestElement,nKclass):
    probClassK = []
    for classk in range(nKclass):
        probClass=[]
        for kTE in range(nTestElement):
            testK = testDF.iloc[kTE].values
            print('valori test[',kTE,'] =',testK)
            print('numero di parametri del Test escludendo l ultimo che indica la classe: ', collPar)
            print('valore dell elemento ',collPar,' dell array test: ', testK[collPar], 'calcoliamo la probabilità che sia',classk)
            probClass.append(1)
            print(probClass, '\n',len(probClass))
            print('probClass iniziale  [',kTE,'] = ',probClass[kTE])
            for parDFi in range(collPar):
                tetaJtestVal = tetaKDf[classk][parDFi][testK[parDFi]]
                #print('tetaKDf [',classk,'] [',parDFi,'] [',testK[parDFi],'] = ',tetaJtestVal)
                probClass[kTE]= probClass[kTE]*tetaJtestVal
                #print('probClass [',kTE,'] = probClass [',kTE,'] * tetaKDf [',classk,'] [',parDFi,'] [',testK[parDFi],']')
                #print(probClass[kTE],' = ',probClass[kTE]/tetaJtestVal,' * ',tetaJtestVal)
            print('probClass ToT [',kTE,'] = ',probClass[kTE], '\n')
        probClassK.append(probClass)
    print(probClassK,'\n',len(probClassK[0]),'\n',len(probClassK[1]))
    return probClassK

def pedictionConMaxProb(probClassK,etaKDF,nTestElement):
    predDFTest = []
    thresholds = []
    for testKPred in range(nTestElement):
        print('etaKDF[0]*probClassK[0][',testKPred,']>=etaKDF[1]*probClassK[1][',testKPred,']')
        print(etaKDF[0],'*',probClassK[0][testKPred],'>=',etaKDF[1],'*',probClassK[1][testKPred])
        print(etaKDF[0] * probClassK[0][testKPred],'>=',etaKDF[1] * probClassK[1][testKPred])
        thresholds.append(etaKDF[0] * probClassK[0][testKPred])
        if etaKDF[0]*probClassK[0][testKPred]>=etaKDF[1]*probClassK[1][testKPred]:
            print('Si')
            predDFTest.append(0)
        else:
            predDFTest.append(1)
            print('No')
    thresholds.sort()
    print(thresholds)
    print(predDFTest)
    return predDFTest, thresholds

def prediction_threshold (probClassK,etaKDF, nTestElement ,threshold):
    predDFTest = []
    for testKPred in range(nTestElement):
        print('etaKDF[0]*probClassK[0][',testKPred,']>=etaKDF[1]*probClassK[1][',testKPred,']')
        print(etaKDF[0],'*',probClassK[0][testKPred],'>=',etaKDF[1],'*',probClassK[1][testKPred])
        print(etaKDF[0] * probClassK[0][testKPred],'>=',etaKDF[1] * probClassK[1][testKPred])
        if etaKDF[0]*probClassK[0][testKPred] >= threshold:
            print('Si')
            predDFTest.append(0)
        else:
            predDFTest.append(1)
            print('No')
    print(predDFTest)
    return predDFTest

def evaluation_parameters(predDFTest,classDFTest,nTestElement):
    print(list(classDFTest))
    nAccuracy=0
    trueNegative=0
    truePositive = 0
    falseNegative = 0
    falsePositive = 0
    print(' nTestElement: ',nTestElement)
    for kTest in range(nTestElement):
        if predDFTest[kTest] == classDFTest[kTest]:
            nAccuracy = nAccuracy +1
        if classDFTest[kTest]==0:
            if predDFTest[kTest] == 0:
                trueNegative = trueNegative + 1
            if predDFTest[kTest] == 1:
                falsePositive = falsePositive + 1
        if classDFTest[kTest]==1:
            if predDFTest[kTest]== 0:
                falseNegative = falseNegative +1
            if predDFTest[kTest]== 1:
                truePositive = truePositive + 1
    accuracy = (nAccuracy/nTestElement)*100
    if(truePositive != 0 and falsePositive != 0):
        precision = truePositive / (truePositive + falsePositive)
    else:
        precision = 1
    recall_tpr = truePositive / (truePositive + falseNegative)
    fpr = falsePositive / (falsePositive + trueNegative)
    print('trueNegative: ', trueNegative)
    print('truePositive: ', truePositive)
    print('falseNegative: ', falseNegative)
    print('falsePositive: ', falsePositive)
    parDf = pd.DataFrame({"accuracy":[accuracy],"precision":[precision],"recall_tpr":[recall_tpr],"fpr":[fpr]})
    return parDf

