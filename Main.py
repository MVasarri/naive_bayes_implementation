import numpy as np
import pandas as pd
import sistemaDataSet
import calcoloTetaEtaDS
import previsioni
import plot_curve_ROC_and_PR

#preparazione dei dataset
nDataSet=0
df = sistemaDataSet.sistemaDS(nDataSet)

collDF = df.shape[1]
collPar = collDF-1

maxVar = sistemaDataSet.limMaxKPar(df)
minVar = sistemaDataSet.limMinKPar(df)
print(maxVar)
print(minVar)


#randomicizzo il numero di righe del data frame
df = df.sample(frac=1).reset_index(drop=True)
print('df randomizzato \n', df)
nDocDF = df.shape[0]

#dataset pronti all'addestramento
nKclass = maxVar[-1] + 1
print('numero di classi: ',nKclass)
trainDF =df.iloc[:int(nDocDF*2/3)].copy()
print('dataset di Train:\n',trainDF)
testDF =df.iloc[int(nDocDF*2/3):].copy().reset_index(drop=True)
print('dataset di Test:\n',testDF)

#roba da fare dopo aver diviso il dataset tra train e test
print(collDF)
nDocTot = trainDF.shape[0]
tetaKDf, etaKDf = calcoloTetaEtaDS.insTetaEtaDS(trainDF, collPar, nKclass, maxVar, nDocTot)

tetaDF = pd.DataFrame(tetaKDf)
etaDf = pd.DataFrame(etaKDf)
print(tetaDF)
print(etaKDf,'\n')


#parte che riguarda le previsioni
nTestElement = testDF.shape[0]
print('DF di test \n',testDF)
print('numero di documenti nel DF di test: ',nTestElement)

probClassK = previsioni.calcoloProbconTeta(testDF, collPar, tetaKDf, nTestElement,nKclass)

print('probabilita 0 e 1')
print(probClassK[0],'\n',probClassK[1],'\n')

thresholds = []
predDFTest , thresholds = previsioni.pedictionConMaxProb(probClassK,etaKDf,nTestElement)

true_class_of_test_element =testDF[str(collPar)].values
elav_par_df = previsioni.evaluation_parameters(predDFTest,true_class_of_test_element,nTestElement)
print(elav_par_df)

#rieseguiamo le previsioni per tutte le soglie che ci siamo ricavate con la prima previsione, e calcolo dei punti per la ROC e la PR rispetto a tutte le soglie
evaluation_parameters_df = pd.DataFrame({"accuracy": [], "precision": [], "recall_tpr": [], "fpr": []})
for soglia in thresholds:
    pred_threshold_test = previsioni.prediction_threshold(probClassK,etaKDf,nTestElement, soglia)
    evaluation_parameter_df =previsioni.evaluation_parameters(pred_threshold_test,true_class_of_test_element,nTestElement)
    evaluation_parameters_df = evaluation_parameters_df.append(evaluation_parameter_df,ignore_index = True)

print(evaluation_parameters_df)

#plottiamo le curve ROC e le curve PR con le rispettive scafi convessi per la ROC e le PR raggiungibili
#ho evitato di eseguire interpolazioni tra i punti della PR querche i risultati raggiunti erano gia sufficenti per la richieta nel testo dell'esercizio

plot_curve_ROC_and_PR.plot_ROC_and_convex_hull_ROC(evaluation_parameters_df)
plot_curve_ROC_and_PR.plot_PR_and_PR_raggiungibile(evaluation_parameters_df)
