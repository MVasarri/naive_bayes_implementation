import numpy as np
import pandas as pd
#from scipy import stats #ho dovuto importare la mibreria scipy per calcolare la moda
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import plotly.graph_objects as go
import plotly.offline as play
#import cufflinks as cf

def plot_ROC_and_convex_hull_ROC(evaluation_parameters_df):
    dati_curva_ROC = evaluation_parameters_df[['fpr', 'recall_tpr']].values
    scafo_ROC = ConvexHull(dati_curva_ROC)

    print(scafo_ROC)
    print(scafo_ROC.vertices.sort())
    print(dati_curva_ROC[scafo_ROC.vertices])

    curva_ROC = go.Scatter(
        x=evaluation_parameters_df['fpr'],
        y=evaluation_parameters_df['recall_tpr'],
        mode='markers+lines',
        name='curve ROC'
    )
    curva_ROC_scafo_convesso = go.Scatter(
        x=dati_curva_ROC[scafo_ROC.vertices, 0],
        y=dati_curva_ROC[scafo_ROC.vertices, 1],
        mode='markers+lines',
        name='convex hull curve ROC'
    )
    curve_ROC_da_proiettare = [curva_ROC, curva_ROC_scafo_convesso]

    fig_curve_ROC = go.Figure(curve_ROC_da_proiettare)
    fig_curve_ROC.update_layout(
        title="curve ROC e convex hull ROC",
        xaxis_title="fpr",
        yaxis_title="tpr",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    )
    fig_curve_ROC.show()

def plot_PR_and_PR_raggiungibile(evaluation_parameters_df):
    dati_curva_PR = evaluation_parameters_df[['recall_tpr', 'precision']].values
    scafo_PR = ConvexHull(dati_curva_PR)
    dati_curva_ROC = evaluation_parameters_df[['fpr', 'recall_tpr']].values
    scafo_ROC = ConvexHull(dati_curva_ROC)

    print(scafo_ROC)
    print(scafo_ROC.vertices.sort())
    print(dati_curva_ROC[scafo_ROC.vertices])

    curva_PR = go.Scatter(
        x=evaluation_parameters_df['recall_tpr'],
        y=evaluation_parameters_df['precision'],
        mode='markers+lines',
        name='curve PR'
    )

    curva_PR_scafo_convesso = go.Scatter(
        x=dati_curva_PR[scafo_PR.vertices, 0],
        y=dati_curva_PR[scafo_PR.vertices, 1],
        mode='markers+lines',
        name='convex hull curve PR'
    )

    curva_PR_raggiungibile = go.Scatter(
        x=dati_curva_PR[scafo_ROC.vertices, 0],
        y=dati_curva_PR[scafo_ROC.vertices, 1],
        mode='markers+lines',
        name='la curva PR raggiungibile'
    )

    curve_PR_da_proiettare = [curva_PR, curva_PR_raggiungibile]

    fig_curve_PR = go.Figure(curve_PR_da_proiettare)
    fig_curve_PR.update_layout(
        title="curve PR e PR raggiungibili",
        xaxis_title="Recall",
        yaxis_title="Precision",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    )
    fig_curve_PR.show()
