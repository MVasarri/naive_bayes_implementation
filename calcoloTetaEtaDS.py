import numpy as np
import pandas as pd
#from scipy import stats

def CalTetaKpar(df,maxVal,numDocDFK):
    tetaKparameter = []
    #print(df)
    teta = df.value_counts()
    print('max value',maxVal,'\n',teta)
    numValparKDS = maxVal + 1
    for valJparKDS in range(numValparKDS):
        try:# la try serve perchè se noi cerchiamo un valore pel parametro che non compare mai nel dataset,
            # il comando seguente non riuscirebbe ad indicizzarlo generando un KeyError
            tetaJ = teta.loc[valJparKDS]
            tetaKparameter.append(tetaJ)
        except KeyError:
            tetaJ = (0)
            tetaKparameter.append(tetaJ)

    alphJK = np.ones(numValparKDS).astype(int) #iperParametri facciamo in modo che ogni valore teta compaia almeno una volta
    print(alphJK)
    print(tetaKparameter)
    print(tetaKparameter + alphJK)
    print(list(tetaKparameter + alphJK))
    tetaProbParameterK = list((tetaKparameter+alphJK)/(numDocDFK+sum(alphJK)))
    return tetaProbParameterK

def calTetaEtaDS(k,nKclass, df,coll,maxVar,nDocTot):
    tetaDS = []
    alphKClass = np.ones(nKclass).astype(int)
    numDocDFK=df.shape[0]
    print('\nnumero di docuenti nel df[', k, '] =',df.shape[0],'\n')
    for parKDataSet in range(coll):
        print('parametro N°: ', parKDataSet)
        tetaKparameter = CalTetaKpar(df[str(parKDataSet)], maxVar[parKDataSet],numDocDFK)
        print(tetaKparameter,'\n')
        tetaDS.append(tetaKparameter)

    print('alphKClass: ', alphKClass)
    print('sum dei alphKClass: ', sum(alphKClass))
    print('numero di docuenti tot df ',nDocTot, '\n')
    etaDS = (df.shape[0]+alphKClass[k])/(nDocTot+sum(alphKClass))
    print(etaDS, '\n')
    return tetaDS,etaDS

def insTetaEtaDS(df, coll, nKclass,maxVar,nDocTot):
    dfK = []
    tetaKDf = []
    etaKDf = []
    for k in range(nKclass):
        dfK.append(df[df[str(coll )] == k].copy())
        dfK[k].drop(str(coll), axis=1, inplace=True)
        print('parte del dataset dove la classe è: ', k, '\n', dfK[k], '\n righe e colonne: ', dfK[k].shape)
        tempTeta, tempEta = calTetaEtaDS(k, nKclass, dfK[k], coll, maxVar, nDocTot)
        tetaKDf.append(tempTeta)
        etaKDf.append(tempEta)
        print('\n', tetaKDf[k])
        print(etaKDf[k])
    return tetaKDf, etaKDf