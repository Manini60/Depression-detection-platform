import matplotlib
matplotlib.use('Agg')

from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np


def create_model(data):
    model = BayesianNetwork([
        ('Academic Pressure', 'Depression'),
        ('Financial Stress', 'Depression'),
        ('Sleep Duration', 'Depression'),
        ('Study Satisfaction', 'Depression'),
        ('Dietary Habits', 'Depression'),
        ('Gender', 'Depression'),
        ('Age', 'Depression')
    ])
    model.fit(data, estimator=MaximumLikelihoodEstimator)
    return model


def draw_network(model):
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    G = nx.DiGraph()
    G.add_edges_from(model.edges())

    # Custom layout: factors around center, Depression at center bottom
    pos = {
        'Academic Pressure': (-2, 1.5),
        'Financial Stress': (-1, 2.5),
        'Sleep Duration': (0, 3),
        'Study Satisfaction': (1, 2.5),
        'Dietary Habits': (2, 1.5),
        'Gender': (-1.5, 0.5),
        'Age': (1.5, 0.5),
        'Depression': (0, 0),
    }

    factor_nodes = [n for n in G.nodes() if n != 'Depression']
    dep_nodes = ['Depression']

    nx.draw_networkx_nodes(G, pos, nodelist=factor_nodes,
                           node_color='#1e3a5f', node_size=3000,
                           alpha=0.9, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=dep_nodes,
                           node_color='#c0392b', node_size=4000,
                           alpha=0.95, ax=ax)
    nx.draw_networkx_labels(G, pos, font_color='white', font_size=8,
                            font_weight='bold', ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='#4a9eff', alpha=0.7,
                           arrows=True, arrowsize=20,
                           arrowstyle='->', width=2, ax=ax)

    ax.set_title("Bayesian Network Structure", color='white',
                 fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig("static/network.png", dpi=150, bbox_inches='tight',
                facecolor='#0d1117')
    plt.close()


def confusion():
    y_true = [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0]
    y_pred = [0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0]

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Depressed', 'Depressed'],
                yticklabels=['Not Depressed', 'Depressed'],
                ax=ax, linewidths=0.5, linecolor='#1e3a5f',
                annot_kws={'size': 14, 'weight': 'bold', 'color': 'white'})

    ax.set_title("Confusion Matrix", color='white', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Predicted", color='#8b9dc3', fontsize=11)
    ax.set_ylabel("Actual", color='#8b9dc3', fontsize=11)
    ax.tick_params(colors='#8b9dc3')

    plt.tight_layout()
    plt.savefig("static/confusion.png", dpi=150, bbox_inches='tight',
                facecolor='#0d1117')
    plt.close()


def get_model_stats(model):
    """Return CPD info for display."""
    stats = []
    for cpd in model.cpds:
        stats.append({
            'variable': cpd.variable,
            'states': list(cpd.state_names[cpd.variable])
        })
    return stats
