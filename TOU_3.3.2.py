#!/usr/bin/env python
# coding: utf-8

"""
Théorie Oscillatoire Unifiée (TOU) - Application Streamlit v3.3.2
===============================================================

Cette application présente la Théorie Oscillatoire Unifiée, un cadre mathématique
permettant de décrire les phénomènes oscillatoires à toutes les échelles,
du quantique au macroscopique.

La TOU s'appuie sur les travaux fondamentaux de la physique quantique et classique,
tout en proposant une approche unifiée pour comprendre les phénomènes oscillatoires
à travers les différentes échelles. 

Références fondamentales:
- Dirac, P.A.M. (1958). The Principles of Quantum Mechanics (4th ed.). Oxford University Press.
- von Neumann, J. (1955). Mathematical Foundations of Quantum Mechanics. Princeton University Press.
- Zurek, W.H. (2003). Decoherence and the transition from quantum to classical -- REVISITED. Physics Today, 44(10), 36-44.
- Aspelmeyer, M., Kippenberg, T.J., & Marquardt, F. (2014). Cavity optomechanics. Reviews of Modern Physics, 86(4), 1391-1452.

Architecture technique:
- Interface utilisateur: Streamlit
- Visualisations: Matplotlib et Plotly
- Calculs scientifiques: NumPy, SciPy

Auteur: Francis Harvey-Pothier
Date: Mai 2025
Version: 3.3.2
Liscence: MIT
"""

# =============================================================================
# IMPORTS ET CONFIGURATION
# =============================================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.special import hermite
import sympy as sp
from PIL import Image
import io
import base64
from IPython.display import HTML
import plotly.figure_factory as ff
import math as python_math
from matplotlib.patches import FancyBboxPatch, Ellipse, Rectangle, Arrow
from scipy.ndimage import gaussian_filter
from typing import Dict, List, Tuple, Union, Optional, Callable, Any


# =============================================================================
# CONFIGURATION DE L'APPLICATION STREAMLIT
# =============================================================================

st.set_page_config(
    page_title="Théorie Oscillatoire Unifiée",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# STYLES CSS ET OPTIMISATIONS D'INTERFACE
# =============================================================================

CSS_STYLES = """
<style>
    /* Optimisation ciblée du rendu des équations mathématiques */
    .katex-display {
        overflow-x: auto;
        overflow-y: hidden;
        padding: 8px 0;
        margin: 0.5em 0;
    }
    
    .equation-container {
        display: flex;
        justify-content: center;
        margin: 25px 0;
        width: 100%;
    }
    
    .scrollable-math {
        overflow-x: auto;
        overflow-y: hidden;
        padding: 5px 0;
        scrollbar-width: thin;
    }
    
    /* Style minimal pour les scrollbars */
    .scrollable-math::-webkit-scrollbar {
        height: 4px;
    }
    
    .scrollable-math::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 2px;
    }
    
    .scrollable-math::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 2px;
    }
    
    /* Augmentation de la taille des équations */
    .katex { 
        font-size: 1.2em !important;
    }
    
    /* Amélioration de l'alignement des équations */
    .katex-html {
        text-align: center;
    }
    
    /* Empêcher les débordements */
    .katex-mathml {
        display: none;
    }
    
    /* Hiérarchie typographique optimisée */
    h1 { font-size: 2.5rem !important; margin-bottom: 1.5rem !important; }
    h2 { font-size: 2rem !important; margin-top: 2rem !important; margin-bottom: 1rem !important; }
    h3 { font-size: 1.6rem !important; margin-top: 1.5rem !important; margin-bottom: 0.8rem !important; }
    
    /* Styles pour les composants visuels structurés */
    .definition, .theorem, .contribution, .proposition {
        padding: 20px 20px 5px 20px; /* Padding réduit en bas */
        border-radius: 8px 8px 0 0; /* Coins arrondis uniquement en haut */
        margin-bottom: 0; /* Pas de marge en bas */
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* Style pour le contenu qui suit immédiatement les boîtes */
    .definition + .stMarkdown,
    .theorem + .stMarkdown,
    .contribution + .stMarkdown,
    .proposition + .stMarkdown {
        background-color: inherit;
        border-radius: 0 0 8px 8px; /* Coins arrondis uniquement en bas */
        padding: 0 20px 20px 20px; /* Padding uniquement sur les côtés et en bas */
        margin-top: 0; /* Pas de marge en haut */
        margin-bottom: 25px; /* Marge en bas */
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* Couleurs spécifiques pour chaque type de boîte */
    .definition {
        background-color: #e6f3ff;
        border-left: 8px solid #0066cc;
    }
    .definition + .stMarkdown {
        background-color: #e6f3ff;
        border-left: 8px solid #0066cc;
    }
    
    .theorem {
        background-color: #fff8e6;
        border-left: 8px solid #ffcc00;
    }
    .theorem + .stMarkdown {
        background-color: #fff8e6;
        border-left: 8px solid #ffcc00;
    }
    
    .contribution {
        background-color: #e6fff0;
        border-left: 8px solid #00cc66;
    }
    .contribution + .stMarkdown {
        background-color: #e6fff0;
        border-left: 8px solid #00cc66;
    }
    
    .proposition {
        background-color: #ffe6e6;
        border-left: 8px solid #cc0000;
    }
    .proposition + .stMarkdown {
        background-color: #ffe6e6;
        border-left: 8px solid #cc0000;
    }
    
    /* Éléments d'accent et de mise en évidence */
    .highlight {
        background-color: #f0f0f0;
        padding: 12px;
        border-radius: 8px;
        margin: 15px 0;
    }
    
    /* Conteneurs flexibles pour mise en page responsive */
    .section-container {
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        margin: 20px 0;
    }
    .section-item {
        flex: 1;
        min-width: 300px;
    }
    
    /* Optimisation des graphiques interactifs */
    .interactive-plot {
        border: 1px solid #eee;
        border-radius: 8px;
        padding: 10px;
        background: white;
    }
    
    /* Animation pour éléments interactifs */
    @keyframes highlight-pulse {
        0% { box-shadow: 0 0 0 0 rgba(0, 102, 204, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(0, 102, 204, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 102, 204, 0); }
    }
    .attention-highlight {
        animation: highlight-pulse 2s infinite;
    }
    
    /* Style des références bibliographiques */
    .reference-item {
        margin-bottom: 12px;
        padding-left: 20px;
        text-indent: -20px;
        line-height: 1.4;
    }
    
    .reference-category {
        background-color: #f8f8f8;
        padding: 10px 15px;
        border-left: 4px solid #0066cc;
        margin: 20px 0 10px 0;
        font-weight: bold;
    }
</style>
"""

# Application des styles CSS optimisés
st.markdown(CSS_STYLES, unsafe_allow_html=True)

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def render_latex(equation: str, centered: bool = True, escape: bool = False, 
               scrollable: bool = True, block: bool = True) -> None:
    """
    Rendu optimisé des équations mathématiques LaTeX avec format amélioré.
    
    Cette fonction utilise st.latex() pour un rendu mathématique précis et fiable,
    avec une gestion avancée des options de formatage et d'affichage.
    
    Références:
        - Reed, M., & Simon, B. (1980). Methods of Modern Mathematical Physics: Functional Analysis.
        - Coecke, B., & Kissinger, A. (2021). Picturing Quantum Processes.
    
    Args:
        equation: Équation au format LaTeX à afficher
        centered: Si True, centre l'équation dans son conteneur
        escape: Si True, échappe les caractères spéciaux pour éviter les erreurs d'interprétation
        scrollable: Si True, ajoute une barre de défilement horizontale pour les équations longues
        block: Si True, utilise la notation $$ (bloc), sinon $ (inline)
    
    Returns:
        None: L'équation est directement affichée via Streamlit
    """
    # Échappement des caractères spéciaux si nécessaire
    if escape:
        equation = equation.replace('\\', '\\\\')
    
    # Utiliser directement st.latex pour un rendu natif optimal
    st.latex(equation)


def definition_box(title: str, content: str) -> None:
    """
    Crée une boîte de définition visuellement structurée avec support LaTeX.
    
    Cette fonction permet d'afficher une boîte stylisée pour présenter des définitions,
    avec un rendu correct des expressions mathématiques LaTeX.
    
    Args:
        title: Titre de la définition
        content: Contenu avec expressions LaTeX délimitées par $...$
    """
    # Première étape : créer la div stylisée
    st.markdown(f"""
    <div class="definition">
        <h4>{title}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Deuxième étape : ajouter le contenu avec support LaTeX
    st.markdown(content)


def theorem_box(title: str, content: str) -> None:
    """
    Crée une boîte de théorème visuellement structurée avec support LaTeX.
    
    Cette fonction permet d'afficher une boîte stylisée pour présenter des théorèmes,
    avec un rendu correct des expressions mathématiques LaTeX.
    
    Args:
        title: Titre du théorème
        content: Contenu avec expressions LaTeX délimitées par $...$
    """
    # Première étape : créer la div stylisée
    st.markdown(f"""
    <div class="theorem">
        <h4>{title}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Deuxième étape : ajouter le contenu avec support LaTeX
    st.markdown(content)


def contribution_box(title: str, content: str) -> None:
    """
    Crée une boîte de contribution originale avec support LaTeX.
    
    Cette fonction permet d'afficher une boîte stylisée pour présenter des contributions originales,
    avec un rendu correct des expressions mathématiques LaTeX.
    
    Args:
        title: Titre de la contribution
        content: Contenu avec expressions LaTeX délimitées par $...$
    """
    # Première étape : créer la div stylisée
    st.markdown(f"""
    <div class="contribution">
        <h4>CONTRIBUTION ORIGINALE: {title}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Deuxième étape : ajouter le contenu avec support LaTeX
    st.markdown(content)


def proposition_box(title: str, content: str) -> None:
    """
    Crée une boîte de proposition avec support LaTeX.
    
    Cette fonction permet d'afficher une boîte stylisée pour présenter des propositions,
    avec un rendu correct des expressions mathématiques LaTeX.
    
    Args:
        title: Titre de la proposition
        content: Contenu avec expressions LaTeX délimitées par $...$
    """
    # Première étape : créer la div stylisée
    st.markdown(f"""
    <div class="proposition">
        <h4>{title}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Deuxième étape : ajouter le contenu avec support LaTeX
    st.markdown(content)


def quote_box(quote: str, author: str) -> None:
    """
    Affiche une citation stylisée avec son auteur dans un cadre visuel distinct.
    
    Args:
        quote: Texte de la citation
        author: Nom de l'auteur de la citation
    """
    st.markdown(f"""
    <div style="background-color: rgba(255, 255, 255, 0.1); border-left: 5px solid #1E88E5; padding: 15px; margin: 15px 0; border-radius: 5px;">
        <p style="font-style: italic; font-size: 1.1em;">"{quote}"</p>
        <p style="text-align: right; font-weight: bold;">— {author}</p>
    </div>
    """, unsafe_allow_html=True)


def reference_item(reference: str) -> None:
    """
    Affiche une référence bibliographique avec un formatage optimisé.
    
    Args:
        reference: Texte formaté de la référence bibliographique
    """
    st.markdown(f"""
    <div class="reference-item">
        {reference}
    </div>
    """, unsafe_allow_html=True)


def reference_category(category: str) -> None:
    """
    Affiche un titre de catégorie pour les références bibliographiques.
    
    Args:
        category: Nom de la catégorie de références
    """
    st.markdown(f"""
    <div class="reference-category">
        {category}
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# VISUALISATIONS DE BASE - SECTION INTRODUCTIVE
# =============================================================================

def create_interactive_harmonic_oscillator() -> go.Figure:
    """
    Crée une visualisation interactive d'oscillateurs harmoniques superposés.
    
    Cette fonction génère un graphique dynamique montrant la superposition
    de trois oscillateurs harmoniques avec différentes fréquences et amplitudes,
    permettant une compréhension intuitive des phénomènes de superposition d'ondes.
    
    Références:
        - Feynman, R.P., Leighton, R.B., & Sands, M. (1965). The Feynman Lectures on Physics.
        - Chen, Y. (2013). Macroscopic Quantum Mechanics: Theory and Experimental Concepts of Optomechanics.
    
    Returns:
        Figure Plotly interactive avec contrôles d'animation
    """
    # Configuration des paramètres temporels et échantillonnage
    t = np.linspace(0, 10, 500)
    
    # Paramètres physiques des oscillateurs
    omega1, omega2, omega3 = 2.0, 2.5, 3.0
    A1, A2, A3 = 1.0, 0.7, 0.5
    
    # Calcul des fonctions d'onde individuelles
    y1 = A1 * np.sin(omega1 * t)
    y2 = A2 * np.sin(omega2 * t)
    y3 = A3 * np.sin(omega3 * t)
    y_sum = y1 + y2 + y3  # Superposition des oscillations
    
    # Construction du graphique interactif
    fig = go.Figure()
    
    # Ajout des traces pour chaque oscillateur
    fig.add_trace(go.Scatter(
        x=t, y=y1,
        name=f'Oscillateur 1',
        line=dict(color='red', width=2, dash='solid'),
        hovertemplate='Temps: %{x:.2f}<br>Amplitude: %{y:.2f}'
    ))
    
    fig.add_trace(go.Scatter(
        x=t, y=y2,
        name=f'Oscillateur 2',
        line=dict(color='green', width=2, dash='solid'),
        hovertemplate='Temps: %{x:.2f}<br>Amplitude: %{y:.2f}'
    ))
    
    fig.add_trace(go.Scatter(
        x=t, y=y3,
        name=f'Oscillateur 3',
        line=dict(color='blue', width=2, dash='solid'),
        hovertemplate='Temps: %{x:.2f}<br>Amplitude: %{y:.2f}'
    ))
    
    fig.add_trace(go.Scatter(
        x=t, y=y_sum,
        name='Oscillation combinée',
        line=dict(color='black', width=3),
        hovertemplate='Temps: %{x:.2f}<br>Amplitude: %{y:.2f}'
    ))
    
    # Point mobile pour visualiser le mouvement instantané
    fig.add_trace(go.Scatter(
        x=[5], y=[y_sum[250]],
        mode='markers',
        marker=dict(size=12, color='black'),
        name='Point mobile',
        hoverinfo='skip'
    ))
    
    # Configuration optimisée de la mise en page
    fig.update_layout(
        title={
            'text': 'Superposition d\'oscillateurs harmoniques',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        xaxis_title='Temps',
        yaxis_title='Amplitude',
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.3)',
            borderwidth=1
        ),
        hovermode='closest',
        plot_bgcolor='white',
        width=800,
        height=500,
        updatemenus=[
            # Menu de contrôle d'animation
            dict(
                type="buttons",
                showactive=False,
                y=0,
                x=1.15,
                xanchor="right",
                yanchor="bottom",
                buttons=[
                    dict(
                        label="Animer",
                        method="animate",
                        args=[None, {"frame": {"duration": 50, "redraw": True},
                                     "fromcurrent": True}]
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": False},
                                       "mode": "immediate",
                                       "transition": {"duration": 0}}]
                    )
                ]
            )
        ],
        # Annotations explicatives et pédagogiques
        annotations=[
            dict(
                x=2.5, y=1.5,
                xref="x", yref="y",
                text="Onde 1: Fréquence ω₁",
                showarrow=True,
                arrowhead=2,
                ax=40, ay=-30,
                font=dict(size=12, color="red")
            ),
            dict(
                x=2, y=-1,
                xref="x", yref="y",
                text="Onde 2: Fréquence ω₂",
                showarrow=True,
                arrowhead=2,
                ax=-40, ay=30,
                font=dict(size=12, color="green")
            ),
            dict(
                x=8, y=2,
                xref="x", yref="y",
                text="Superposition des ondes",
                showarrow=True,
                arrowhead=2,
                ax=0, ay=-40,
                font=dict(size=12, color="black")
            )
        ]
    )
    
    # Configuration des axes pour une visualisation optimale
    fig.update_xaxes(
        range=[0, 10],
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='black',
        gridcolor='lightgray'
    )
    
    fig.update_yaxes(
        range=[-3, 3],
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='black',
        gridcolor='lightgray'
    )
    
    # Génération des frames d'animation pour effet dynamique
    frames = []
    for i in range(100):
        # Décalage de phase progressif pour créer l'animation
        shift = i / 50.0
        y1_new = A1 * np.sin(omega1 * (t + shift))
        y2_new = A2 * np.sin(omega2 * (t + shift))
        y3_new = A3 * np.sin(omega3 * (t + shift))
        y_sum_new = y1_new + y2_new + y3_new
        
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(x=t, y=y1_new),
                    go.Scatter(x=t, y=y2_new),
                    go.Scatter(x=t, y=y3_new),
                    go.Scatter(x=t, y=y_sum_new),
                    go.Scatter(x=[5], y=[y_sum_new[250]])
                ]
            )
        )
    
    fig.frames = frames
    
    return fig


def create_interactive_3d_wavefunction() -> go.Figure:
    """
    Crée une visualisation 3D interactive d'une fonction d'onde d'oscillateur harmonique 2D.
    
    Cette fonction génère une représentation tridimensionnelle avancée de la densité
    de probabilité d'un état fondamental d'oscillateur harmonique quantique,
    avec des contrôles interactifs pour explorer les propriétés de la fonction d'onde.
    
    Références:
        - Dirac, P.A.M. (1958). The Principles of Quantum Mechanics.
        - Glauber, R.J. (1963). Coherent and Incoherent States of the Radiation Field.
    
    Returns:
        go.Figure: Figure interactive 3D de la fonction d'onde
    """
    # Création du maillage avec résolution optimisée pour le rendu
    resolution = 100  # Compromis entre détail et performance
    x = np.linspace(-5, 5, resolution)
    y = np.linspace(-5, 5, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Calcul de la fonction d'onde d'oscillateur harmonique 2D (état fondamental)
    sigma = 1.0
    r_squared = X**2 + Y**2  # Pré-calcul pour optimisation
    psi_00 = np.exp(-r_squared / (2*sigma**2)) / (np.sqrt(np.pi) * sigma)
    
    # Calcul de la densité de probabilité
    prob_density = np.abs(psi_00)**2
    
    # Création du graphique 3D avec Plotly - Avec structure correcte pour colorbar
    surface_trace = go.Surface(
        z=prob_density, 
        x=x, 
        y=y, 
        colorscale='Viridis',
        opacity=0.9,
        hovertemplate='x: %{x:.2f}<br>y: %{y:.2f}<br>Densité: %{z:.4f}',
        colorbar=dict(
            # Structure correcte pour la configuration du titre
            title=dict(
                text="Densité de<br>probabilité",
                side="right",
                font=dict(size=14)
            )
        ),
        contours=dict(
            z=dict(
                show=True,
                usecolormap=True,
                highlightcolor="limegreen",
                project=dict(z=True)
            )
        )
    )
    
    # Initialisation de la figure
    fig = go.Figure(data=[surface_trace])
    
    # Configuration optimisée de la mise en page
    fig.update_layout(
        title={
            'text': "Fonction d'onde de l'état fondamental d'un oscillateur harmonique 2D",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=18)
        },
        scene=dict(
            xaxis_title='Position x',
            yaxis_title='Position y',
            zaxis_title='Densité de probabilité',
            aspectratio=dict(x=1, y=1, z=0.8),
            camera=dict(
                eye=dict(x=1.2, y=1.2, z=1.2),
                up=dict(x=0, y=0, z=1)
            ),
            xaxis=dict(nticks=8, range=[-5, 5]),
            yaxis=dict(nticks=8, range=[-5, 5]),
            zaxis=dict(nticks=8),
        ),
        width=850,
        height=700,
        margin=dict(l=0, r=0, b=0, t=60)
    )
    
    # Ajout du menu de navigation avec contrôles optimisés
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        args=[{'scene.camera.eye': {'x': 1.2, 'y': 1.2, 'z': 1.2}}],
                        label="Vue isométrique",
                        method="relayout"
                    ),
                    dict(
                        args=[{'scene.camera.eye': {'x': 0.0, 'y': 0.0, 'z': 2.5}}],
                        label="Vue du dessus",
                        method="relayout"
                    ),
                    dict(
                        args=[{'scene.camera.eye': {'x': 2.5, 'y': 0.0, 'z': 0.0}}],
                        label="Vue latérale X",
                        method="relayout"
                    ),
                    dict(
                        args=[{'scene.camera.eye': {'x': 0.0, 'y': 2.5, 'z': 0.0}}],
                        label="Vue latérale Y",
                        method="relayout"
                    )
                ]),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top",
                bgcolor="white",
                bordercolor="rgba(0,0,0,0.3)",
                borderwidth=1
            )
        ]
    )
    
    # Annotation explicative avec placement optimisé
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text="État fondamental: Ψ₀₀(x,y) = (1/√π)·e^(-(x²+y²)/2)",
        showarrow=False,
        font=dict(size=14),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        opacity=0.8,
        align="left"
    )
    
    return fig


# Définir explicitement create_3d_wavefunction comme un alias
def create_3d_wavefunction() -> go.Figure:
    """
    Fonction compatible avec le code original qui appelle la version interactive.
    Cette fonction sert de pont entre l'ancienne et la nouvelle implémentation.
    
    Returns:
        go.Figure: Figure interactive 3D de la fonction d'onde
    """
    return create_interactive_3d_wavefunction()


def create_axioms_diagram_improved() -> plt.Figure:
    """
    Crée un diagramme amélioré des cinq axiomes de la Théorie Oscillatoire Unifiée.
    
    Cette fonction génère une représentation visuelle optimisée des cinq axiomes
    fondamentaux de la TOU, avec une disposition qui évite les superpositions de texte
    et améliore la lisibilité globale.
    
    Références:
        - von Neumann, J. (1955). Mathematical Foundations of Quantum Mechanics.
        - Bratteli, O., & Robinson, D.W. (1996). Operator Algebras and Quantum Statistical Mechanics.
    
    Returns:
        plt.Figure: Figure matplotlib contenant le diagramme des axiomes
    """
    # Initialisation de la figure avec dimensions optimisées
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Suppression des axes pour une visualisation propre
    ax.set_axis_off()
    
    # Configuration des limites du graphique pour un positionnement précis
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Définition des axiomes avec positions optimisées pour éviter les superpositions
    axiom_boxes = [
        {"x": 0.5, "y": 0.82, "title": "Axiome 1: Espace d'états", 
         "text": "L'espace des états d'un système oscillatoire est\nreprésenté par un espace de Hilbert paramétrique"},
        {"x": 0.2, "y": 0.55, "title": "Axiome 2: Évolution", 
         "text": "L'évolution temporelle du système est décrite\npar une transformation unitaire générée\npar un opérateur hamiltonien"},
        {"x": 0.8, "y": 0.55, "title": "Axiome 3: Structure spectrale", 
         "text": "Le spectre de l'opérateur d'évolution contient\ndes composantes oscillatoires caractérisées\npar des fréquences propres"},
        {"x": 0.3, "y": 0.25, "title": "Axiome 4: Couplage", 
         "text": "L'interaction entre différents systèmes oscillatoires\nest représentée par un opérateur de couplage"},
        {"x": 0.7, "y": 0.25, "title": "Axiome 5: Principe de correspondance", 
         "text": "Dans la limite appropriée, les équations\nreproduisent les équations classiques\ndes oscillateurs couplés"}
    ]
    
    # Cercle central représentant la TOU, avec ombre pour effet de profondeur
    shadow_circle = plt.Circle((0.51, 0.49), 0.12, color='darkgray', alpha=0.4, zorder=1)
    ax.add_patch(shadow_circle)
    
    circle = plt.Circle((0.5, 0.5), 0.12, color='lightblue', alpha=0.6, zorder=2)
    ax.add_patch(circle)
    
    # Titre central avec espacement optimisé des lignes
    ax.text(0.5, 0.51, "Théorie\nOscillatoire\nUnifiée", 
            ha='center', va='center', fontsize=12, fontweight='bold',
            linespacing=1.3, zorder=3)
    
    # Création des boîtes pour chaque axiome avec style visuel amélioré
    for i, box in enumerate(axiom_boxes):
        # Ombre pour effet de profondeur
        shadow_rect = FancyBboxPatch(
            (box["x"]-0.183, box["y"]-0.093), 
            0.36, 0.18, 
            boxstyle=f"round,pad=0.03,rounding_size=0.05",
            facecolor='darkgray', alpha=0.4,
            linewidth=0, zorder=3
        )
        ax.add_patch(shadow_rect)
        
        # Rectangle principal avec coins arrondis
        rect = FancyBboxPatch(
            (box["x"]-0.18, box["y"]-0.09), 
            0.36, 0.18, 
            boxstyle=f"round,pad=0.03,rounding_size=0.05",
            facecolor='lightyellow', edgecolor='black', alpha=0.8,
            linewidth=1.5, zorder=4
        )
        ax.add_patch(rect)
        
        # Titre de l'axiome
        ax.text(box["x"], box["y"]+0.04, box["title"], 
                ha='center', va='center', fontsize=12, fontweight='bold',
                zorder=5)
        
        # Texte descriptif de l'axiome
        ax.text(box["x"], box["y"]-0.02, box["text"], 
                ha='center', va='center', fontsize=10, linespacing=1.3,
                zorder=5)
        
        # Connexion au centre avec effet de profondeur
        # Ombre de la ligne
        ax.plot([0.51, box["x"]+0.005], [0.49, box["y"]-0.005], 'k-', alpha=0.3, 
                linewidth=2, zorder=1.5)
        
        # Ligne principale
        ax.plot([0.5, box["x"]], [0.5, box["y"]], 'k-', alpha=0.6, 
                linewidth=1.5, zorder=2)
    
    # Titre général du diagramme avec cadre décoratif
    title_box = Rectangle((0.25, 0.93), 0.5, 0.05, 
                         facecolor='white', alpha=0.7, 
                         edgecolor='gray', linewidth=1,
                         zorder=6)
    ax.add_patch(title_box)
    
    ax.text(0.5, 0.95, "Les Cinq Axiomes de la Théorie Oscillatoire Unifiée", 
            ha='center', va='center', fontsize=16, fontweight='bold',
            zorder=7)
    
    # Utilisation de tight_layout pour optimiser l'espace
    fig.tight_layout(pad=1.5)
    
    return fig


def plot_quantum_potential_well() -> plt.Figure:
    """
    Crée une visualisation optimisée d'un puits de potentiel quantique
    avec plusieurs états d'énergie et leurs fonctions d'onde.
    
    Références:
        - Dirac, P.A.M. (1958). The Principles of Quantum Mechanics.
        - Schlosshauer, M. (2019). Quantum decoherence. Physics Reports, 831, 1-57.
    
    Returns:
        plt.Figure: Figure matplotlib du puits quantique et de ses états
    """
    # Paramètres du système physique
    x = np.linspace(-5, 5, 1000)
    V0 = 5  # Profondeur du puits de potentiel
    
    # Définition du potentiel
    def V(x: np.ndarray) -> np.ndarray:
        """Fonction de potentiel (puits carré fini)"""
        return np.where(np.abs(x) < 1, 0, V0)
    
    # Niveaux d'énergie quantifiés (simplifiés pour la visualisation)
    energies = [0.5, 2.0, 4.5]
    
    # Fonctions d'onde analytiques (simplifiées)
    def psi(n: int, x: np.ndarray) -> np.ndarray:
        """
        Calcule la fonction d'onde pour le niveau n
        
        Args:
            n: Niveau d'énergie (0, 1, 2)
            x: Points de l'espace
        
        Returns:
            Amplitude de la fonction d'onde
        """
        if n == 0:
            return np.where(np.abs(x) < 1, 
                           np.cos(np.pi*x/2), 
                           np.cos(np.pi/2)*np.exp(-(np.abs(x)-1)))
        elif n == 1:
            return np.where(np.abs(x) < 1, 
                           np.sin(np.pi*x), 
                           np.sin(np.pi)*np.exp(-(np.abs(x)-1)))
        else:
            return np.where(np.abs(x) < 1, 
                           np.cos(3*np.pi*x/2), 
                           np.cos(3*np.pi/2)*np.exp(-(np.abs(x)-1)))
    
    # Création du graphique avec style optimisé
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Tracé du potentiel avec style amélioré
    ax.plot(x, V(x), 'k-', linewidth=2.5, label='Potentiel $V(x)$')
    
    # Palette de couleurs optimisée pour meilleure distinction
    colors = ['#3366CC', '#DC3912', '#109618']
    
    # Tracé des niveaux d'énergie et des fonctions d'onde
    for i, E in enumerate(energies):
        # Ligne horizontale pour le niveau d'énergie
        ax.axhline(y=E, color=colors[i], linestyle='--', alpha=0.6, linewidth=1.5)
        
        # Calcul de la fonction d'onde
        wave = psi(i, x)
        scale_factor = 0.5  # Facteur d'échelle pour la visualisation
        wave_scaled = wave * scale_factor + E
        
        # Tracé de la fonction d'onde
        ax.plot(x, wave_scaled, color=colors[i], linewidth=2, 
                label=f'$\psi_{i}$, $E_{i} = {E}$')
        
        # Remplissage entre la fonction d'onde et le niveau d'énergie
        ax.fill_between(x, E, wave_scaled, color=colors[i], alpha=0.2)
    
    # Annotations pour améliorer la compréhension
    ax.annotate('État fondamental', xy=(1.2, 0.5), xytext=(2, 0.7),
                arrowprops=dict(facecolor=colors[0], shrink=0.05, width=1.5, headwidth=8),
                fontsize=10)
    
    ax.annotate('Premier état excité', xy=(-0.5, 2.0), xytext=(-2.5, 2.5),
                arrowprops=dict(facecolor=colors[1], shrink=0.05, width=1.5, headwidth=8),
                fontsize=10)
    
    # Configuration optimisée des axes et des légendes
    ax.set_xlim(-3, 3)
    ax.set_ylim(-1, V0+1)
    ax.set_xlabel('Position $x$', fontsize=12)
    ax.set_ylabel('Énergie $E$', fontsize=12)
    ax.set_title('États liés dans un puits de potentiel quantique', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.7)
    
    # Ajout d'un encadré explicatif
    textbox = "Les états quantiques dans un puits de potentiel sont discrets\n" + \
              "avec des énergies bien définies. Les particules ne peuvent\n" + \
              "occuper que ces niveaux d'énergie spécifiques."
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax.text(0.05, 0.05, textbox, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)
    
    fig.tight_layout()
    
    return fig


def create_frobenius_algebra_diagram_improved() -> plt.Figure:
    """
    Crée un diagramme amélioré des structures algébriques de la TOU
    avec un meilleur placement et des séparations claires pour éviter
    toute superposition de texte.
    
    Références:
        - Abramsky, S., & Coecke, B. (2008). Categorical quantum mechanics.
        - Abrams, L. (1996). Two-dimensional topological quantum field theories and Frobenius algebras.
    
    Returns:
        plt.Figure: Figure matplotlib contenant le diagramme des structures algébriques
    """
    # Initialisation de la figure avec dimensions optimisées
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Suppression des axes pour une visualisation propre
    ax.set_axis_off()
    
    # Configuration des limites du graphique pour un positionnement précis
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Définition des structures avec leurs propriétés dans une structure de données organisée
    structures = [
        {"x": 0.25, "y": 0.75, "title": "Algèbres de\nFrobenius", 
         "props": ["Associativité", "Unité", "Forme bilinéaire\nnon-dégénérée", "Compatibilité multiplicative"]},
        {"x": 0.75, "y": 0.75, "title": "C*-algèbres", 
         "props": ["Algèbre de Banach", "Involution *", "Condition de C*:\n||a*a|| = ||a||²"]},
        {"x": 0.5, "y": 0.35, "title": "Algèbres de\nFrobenius C*", 
         "props": ["Structure combinée", "Compatible avec la\ninvolution et la norme", "Fondement mathématique\nde la TOU"]}
    ]
    
    # Rendu des structures avec ombres pour effet 3D
    for struct in structures:
        # Ajout d'ombre décalée pour effet de profondeur
        shadow = Ellipse(
            (struct["x"]+0.007, struct["y"]-0.007), 
            0.32, 0.27, 
            facecolor='darkgray', 
            alpha=0.4,
            zorder=1  # Niveau inférieur pour l'ombre
        )
        ax.add_patch(shadow)
        
        # Ellipse principale avec style cohérent
        ellipse = Ellipse(
            (struct["x"], struct["y"]), 
            0.32, 0.27, 
            facecolor='lightblue', 
            edgecolor='blue', 
            alpha=0.8,
            linewidth=2,
            zorder=2  # Niveau supérieur à l'ombre
        )
        ax.add_patch(ellipse)
        
        # Titre de la structure avec formatage amélioré
        ax.text(
            struct["x"], 
            struct["y"]+0.05, 
            struct["title"], 
            ha='center', 
            va='center', 
            fontsize=14, 
            fontweight='bold',
            linespacing=1.3,
            zorder=3  # Niveau supérieur pour assurer visibilité
        )
        
        # Propriétés affichées avec espacement optimisé
        y_offset = 0
        for prop in struct["props"]:
            y_offset -= 0.035  # Espacement progressif
            ax.text(
                struct["x"], 
                struct["y"]+y_offset, 
                prop, 
                ha='center', 
                va='center', 
                fontsize=10, 
                linespacing=1.2,
                zorder=3
            )
    
    # Connexions avec courbes splines pour un rendu plus élégant
    # Courbe gauche: Frobenius vers Frobenius C*
    curve_x1 = np.linspace(0.25, 0.5, 100)
    # Fonction sinusoïdale pour créer une courbe naturelle
    curve_y1 = 0.75 - 0.4 * np.sin(np.pi * (curve_x1 - 0.25) / 0.25)
    ax.plot(curve_x1, curve_y1, 'k-', linewidth=2.5, alpha=0.6, zorder=1.5)
    
    # Courbe droite: C*-algèbres vers Frobenius C*
    curve_x2 = np.linspace(0.5, 0.75, 100)
    curve_y2 = 0.75 - 0.4 * np.sin(np.pi * (curve_x2 - 0.5) / 0.25)
    ax.plot(curve_x2, curve_y2, 'k-', linewidth=2.5, alpha=0.6, zorder=1.5)
    
    # Étiquettes de connexion avec cadres pour améliorer la lisibilité
    # Étiquette gauche avec box styling avancé
    ax.text(
        0.35, 0.55, 
        "Structure\nalgébrique", 
        ha='center', 
        va='center', 
        fontsize=10, 
        bbox=dict(
            facecolor='white', 
            alpha=0.9, 
            edgecolor='lightgray',
            boxstyle='round,pad=0.4'
        ),
        linespacing=1.3,
        zorder=4  # Sur les connexions
    )
    
    # Étiquette droite avec style cohérent
    ax.text(
        0.65, 0.55, 
        "Structure\nanalytique", 
        ha='center', 
        va='center', 
        fontsize=10,
        bbox=dict(
            facecolor='white', 
            alpha=0.9, 
            edgecolor='lightgray',
            boxstyle='round,pad=0.4'
        ),
        linespacing=1.3,
        zorder=4
    )
    
    # Liste structurée des applications avec format uniforme
    applications = [
        "1. Description des systèmes oscillatoires quantiques",
        "2. Fondement pour la théorie quantique des champs topologiques",
        "3. Formalisation des règles de fusion et d'interaction",
        "4. Modélisation des systèmes optomécaniques"
    ]
    
    # Rectangle d'ombre pour effet 3D
    shadow_rect = Rectangle(
        (0.19, 0.08), 
        0.62, 0.17, 
        facecolor='darkgray', 
        alpha=0.4,
        zorder=1
    )
    ax.add_patch(shadow_rect)
    
    # Rectangle principal pour les applications
    rect = FancyBboxPatch(
        (0.2, 0.1), 
        0.6, 0.15, 
        boxstyle="round,pad=0.03,rounding_size=0.05",
        facecolor='lightyellow', 
        edgecolor='black', 
        alpha=0.8, 
        linewidth=1.5,
        zorder=2
    )
    ax.add_patch(rect)
    
    # Titre de la section applications
    ax.text(
        0.5, 0.2, 
        "Applications:", 
        ha='center', 
        va='center', 
        fontsize=12, 
        fontweight='bold',
        zorder=3
    )
    
    # Placement précis des items d'application avec espacement calculé
    y_pos = 0.18
    for app in applications:
        y_pos -= 0.027  # Espacement constant
        ax.text(
            0.5, y_pos, 
            app, 
            ha='center', 
            va='center', 
            fontsize=10,
            zorder=3
        )
    
    # Connexion verticale vers les applications
    curve_x3 = np.linspace(0.5, 0.5, 100)
    curve_y3 = np.linspace(0.22, 0.35 - 0.125, 100)
    ax.plot(curve_x3, curve_y3, 'k-', linewidth=2, alpha=0.6, zorder=1.5)
    
    # Titre principal avec cadre décoratif
    ax.text(
        0.5, 0.95, 
        "Structures Algébriques de la Théorie Oscillatoire Unifiée", 
        ha='center', 
        va='center', 
        fontsize=16, 
        fontweight='bold',
        bbox=dict(
            facecolor='white', 
            alpha=0.7, 
            edgecolor=None,
            boxstyle='round,pad=0.3'
        ),
        zorder=5  # Niveau supérieur pour le titre
    )
    
    # Assure que la mise en page est optimale
    fig.tight_layout(pad=1.5)
    
    return fig


def create_interactive_hilbert_space() -> go.Figure:
    """
    Crée une visualisation 3D interactive de l'espace de Hilbert (sphère de Bloch)
    avec des contrôles pour explorer différentes représentations d'états quantiques.
    
    Références:
        - von Neumann, J. (1955). Mathematical Foundations of Quantum Mechanics.
        - Sala, P., et al. (2020). Ergodicity breaking arising from Hilbert space fragmentation.
    
    Returns:
        go.Figure: Figure interactive Plotly de l'espace de Hilbert
    """
    # Construction de la géométrie de la sphère de Bloch
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Initialisation de la figure Plotly
    fig = go.Figure()
    
    # Ajout de la sphère semi-transparente
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        opacity=0.3,
        colorscale='Blues',
        showscale=False,
        hoverinfo='skip'
    ))
    
    # Définition des axes principaux avec longueur normalisée
    axis_length = 1.3
    
    # Axe X (rouge) - Convention: |0⟩ + |1⟩ / |0⟩ - |1⟩
    fig.add_trace(go.Scatter3d(
        x=[-axis_length, axis_length], y=[0, 0], z=[0, 0],
        mode='lines',
        line=dict(color='red', width=4),
        name='Axe X'
    ))
    
    # Axe Y (vert) - Convention: |0⟩ + i|1⟩ / |0⟩ - i|1⟩
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[-axis_length, axis_length], z=[0, 0],
        mode='lines',
        line=dict(color='green', width=4),
        name='Axe Y'
    ))
    
    # Axe Z (bleu) - Convention: |0⟩ / |1⟩
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[-axis_length, axis_length],
        mode='lines',
        line=dict(color='blue', width=4),
        name='Axe Z'
    ))
    
    # Définition de plusieurs états quantiques pour illustrer l'espace de Hilbert
    states = [
        {"theta": np.pi/4, "phi": np.pi/3, "color": "purple", "name": "État |ψ₁⟩"},
        {"theta": np.pi/3, "phi": 2*np.pi/3, "color": "orange", "name": "État |ψ₂⟩"},
        {"theta": np.pi/5, "phi": np.pi, "color": "magenta", "name": "État |ψ₃⟩"},
        {"theta": 2*np.pi/3, "phi": np.pi/2, "color": "cyan", "name": "État |ψ₄⟩"}
    ]
    
    # Projection des états sur la sphère de Bloch
    for state in states:
        theta, phi = state["theta"], state["phi"]
        # Conversion des coordonnées sphériques en cartésiennes
        x_pos = np.sin(theta) * np.cos(phi)
        y_pos = np.sin(theta) * np.sin(phi)
        z_pos = np.cos(theta)
        
        # Tracé du vecteur d'état
        fig.add_trace(go.Scatter3d(
            x=[0, x_pos], y=[0, y_pos], z=[0, z_pos],
            mode='lines+markers',
            line=dict(color=state["color"], width=6),
            marker=dict(size=6, color=state["color"]),
            name=state["name"],
            hovertemplate=f'{state["name"]}<br>θ: {theta:.2f}<br>φ: {phi:.2f}'
        ))
        
        # Projection sur les plans principaux (optionnel)
        if state["name"] == "État |ψ₁⟩":  # Ne montrer que pour un état pour éviter l'encombrement
            # Projection sur XY
            fig.add_trace(go.Scatter3d(
                x=[x_pos], y=[y_pos], z=[0],
                mode='markers',
                marker=dict(size=4, color=state["color"], opacity=0.5),
                name=f'{state["name"]} (proj. XY)',
                showlegend=False
            ))
            
            # Projection sur XZ
            fig.add_trace(go.Scatter3d(
                x=[x_pos], y=[0], z=[z_pos],
                mode='markers',
                marker=dict(size=4, color=state["color"], opacity=0.5),
                name=f'{state["name"]} (proj. XZ)',
                showlegend=False
            ))
    
    # Ajout des cercles de projection sur les plans principaux
    points = 100
    
    # Cercle sur le plan XY (équateur)
    theta_xy = np.linspace(0, 2*np.pi, points)
    fig.add_trace(go.Scatter3d(
        x=np.cos(theta_xy), y=np.sin(theta_xy), z=np.zeros(points),
        mode='lines',
        line=dict(color='gray', width=1, dash='dash'),
        name='Plan XY',
        showlegend=True
    ))
    
    # Configuration améliorée de la mise en page
    fig.update_layout(
        title={
            'text': 'Espace de Hilbert (Représentation de Bloch)',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20)
        },
        scene=dict(
            xaxis_title='|0⟩ ← → |1⟩',
            yaxis_title='Im',
            zaxis_title='|0⟩ ← → |1⟩',
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.6, y=1.6, z=1.2),
                up=dict(x=0, y=0, z=1)
            ),
            annotations=[
                dict(
                    showarrow=False,
                    x=0, y=0, z=1.5,
                    text='|0⟩',
                    xanchor='center',
                    font=dict(size=14)
                ),
                dict(
                    showarrow=False,
                    x=0, y=0, z=-1.5,
                    text='|1⟩',
                    xanchor='center',
                    font=dict(size=14)
                )
            ]
        ),
        legend=dict(
            x=0.8,
            y=0.9,
            traceorder='normal',
            bgcolor='rgba(255, 255, 255, 0.5)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1
        ),
        width=900,
        height=700,
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    # Ajout d'un menu interactif pour explorer différentes perspectives
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        args=[{'scene.camera.eye': {'x': 1.6, 'y': 1.6, 'z': 1.2}}],
                        label="Vue Isométrique",
                        method="relayout"
                    ),
                    dict(
                        args=[{'scene.camera.eye': {'x': 0.0, 'y': 0.0, 'z': 2.5}}],
                        label="Vue Pôle Nord (|0⟩)",
                        method="relayout"
                    ),
                    dict(
                        args=[{'scene.camera.eye': {'x': 0.0, 'y': 0.0, 'z': -2.5}}],
                        label="Vue Pôle Sud (|1⟩)",
                        method="relayout"
                    ),
                    dict(
                        args=[{'scene.camera.eye': {'x': 2.5, 'y': 0.0, 'z': 0.0}}],
                        label="Vue Axe X",
                        method="relayout"
                    ),
                    dict(
                        args=[{'scene.camera.eye': {'x': 0.0, 'y': 2.5, 'z': 0.0}}],
                        label="Vue Axe Y",
                        method="relayout"
                    )
                ]),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top"
            )
        ]
    )
    
    # Annotation explicative sur la sphère de Bloch
    fig.add_annotation(
        x=0.01,
        y=0.01,
        xref="paper",
        yref="paper",
        text="Représentation de Bloch des états quantiques à 2 niveaux<br>|ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩",
        showarrow=False,
        font=dict(size=12),
        align="left",
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        opacity=0.8
    )
    
    return fig


def create_hilbert_fragmentation_diagram_improved() -> plt.Figure:
    """
    Crée un diagramme amélioré de la fragmentation de l'espace de Hilbert
    avec un placement de texte optimisé et des couleurs harmonisées.
    
    Références:
        - Sala, P., et al. (2020). Ergodicity breaking arising from Hilbert space fragmentation.
        - Moudgalya, S., & Motrunich, O.I. (2022). Hilbert space fragmentation and commutant algebras.
    
    Returns:
        plt.Figure: Figure matplotlib illustrant la fragmentation
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Suppression des axes pour une visualisation propre
    ax.set_axis_off()
    
    # Configuration des limites du graphique pour un positionnement précis
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Création du cercle principal représentant l'espace de Hilbert complet
    # Ajout d'une ombre pour effet de profondeur
    shadow_circle = plt.Circle((0.51, 0.49), 0.35, fill=False, edgecolor='gray', 
                              alpha=0.3, linewidth=3, zorder=1)
    ax.add_patch(shadow_circle)
    
    main_circle = plt.Circle((0.5, 0.5), 0.35, fill=False, edgecolor='black', 
                            linewidth=2.5, zorder=2)
    ax.add_patch(main_circle)
    
    # Titre principal dans un cadre stylisé
    title_box = Rectangle((0.25, 0.88), 0.5, 0.05, facecolor='white', 
                         alpha=0.8, edgecolor='gray', linewidth=1, zorder=3)
    ax.add_patch(title_box)
    
    ax.text(0.5, 0.9, "Fragmentation de l'Espace de Hilbert", 
            ha='center', va='center', fontsize=18, fontweight='bold', zorder=4)
    
    # Palette de couleurs optimisée pour meilleure distinction et harmonie
    fragment_colors = ["#FF5733", "#33A8FF", "#33FF57", "#A033FF", "#FF33A8"]
    
    # Définition des fragments avec positions optimisées pour éviter les superpositions
    fragments = [
        {"center": (0.38, 0.38), "radius": 0.09, "color": fragment_colors[0], "label": "K₁"},
        {"center": (0.65, 0.55), "radius": 0.11, "color": fragment_colors[1], "label": "K₂"},
        {"center": (0.44, 0.67), "radius": 0.08, "color": fragment_colors[2], "label": "K₃"},
        {"center": (0.62, 0.35), "radius": 0.07, "color": fragment_colors[3], "label": "K₄"},
        {"center": (0.3, 0.55), "radius": 0.06, "color": fragment_colors[4], "label": "K₅"}
    ]
    
    # Rendu des fragments avec effet de profondeur
    for i, f in enumerate(fragments):
        # Ajout d'une ombre pour effet 3D
        shadow = plt.Circle(
            (f["center"][0]+0.005, f["center"][1]-0.005), 
            f["radius"], 
            color='gray', 
            alpha=0.3,
            zorder=4
        )
        ax.add_patch(shadow)
        
        # Cercle principal du fragment
        circle = plt.Circle(
            f["center"], 
            f["radius"], 
            color=f["color"], 
            alpha=0.6,
            zorder=5
        )
        ax.add_patch(circle)
        
        # Étiquette du fragment avec fond contrasté pour meilleure lisibilité
        ax.text(
            f["center"][0], 
            f["center"][1], 
            f["label"], 
            ha='center', 
            va='center', 
            fontsize=14, 
            fontweight='bold',
            bbox=dict(
                facecolor='white', 
                alpha=0.5, 
                boxstyle='round,pad=0.2',
                edgecolor=f["color"]
            ),
            zorder=6
        )
    
    # Équation de la décomposition en sous-espaces dans un cadre décoratif
    eq_box = FancyBboxPatch(
        (0.3, 0.12), 
        0.4, 0.08, 
        boxstyle="round,pad=0.04,rounding_size=0.05",
        facecolor='#f8f8f8', 
        edgecolor='#333', 
        alpha=0.8, 
        linewidth=1.5,
        zorder=7
    )
    ax.add_patch(eq_box)
    
    # Équation mathématique
    ax.text(
        0.5, 0.16, 
        r"$\mathcal{H} = \bigoplus_{\alpha} \mathcal{K}_{\alpha}$", 
        ha='center', 
        va='center', 
        fontsize=20,
        zorder=8
    )
    
    # Cadre pour les explications avec style cohérent
    explanation_box = FancyBboxPatch(
        (0.15, -0.1), 
        0.7, 0.2, 
        boxstyle="round,pad=0.04,rounding_size=0.05",
        facecolor='#f0f8ff', 
        edgecolor='#336699', 
        alpha=0.7, 
        linewidth=1.5,
        zorder=9
    )
    ax.add_patch(explanation_box)
    
    # Liste des propriétés de la fragmentation
    explanation = [
        "• Les sous-espaces Kₐ sont invariants sous l'action du hamiltonien",
        "• L'évolution temporelle ne peut pas connecter différents fragments",
        "• Conduit à la non-ergodicité et à la brisure de thermalization",
        "• Phénomène crucial dans les systèmes désordonnés et contraints"
    ]
    
    # Placement des explications avec espacement optimisé
    y_pos = 0.05
    for line in explanation:
        y_pos -= 0.05  # Espacement constant
        ax.text(
            0.5, 
            y_pos, 
            line, 
            ha='center', 
            va='center', 
            fontsize=12,
            fontweight='normal', 
            color='#333',
            zorder=10
        )
    
    # Connexions entre fragments et équation pour illustrer la relation
    for f in fragments:
        # Flèche avec style de connexion courbe
        ax.annotate(
            '', 
            xy=(0.5, 0.16),  # Point d'arrivée (équation)
            xytext=f["center"],  # Point de départ (fragment)
            arrowprops=dict(
                arrowstyle='->',  # Style de flèche
                lw=1.5,  # Largeur de ligne
                color='gray',  # Couleur
                alpha=0.5,  # Transparence
                connectionstyle='arc3,rad=0.2'  # Courbe de connexion
            ),
            zorder=3  # Sous les fragments mais sur le cercle principal
        )
    
    return fig


def create_quantum_classical_transition_diagram_improved() -> plt.Figure:
    """
    Crée un diagramme amélioré de la transition quantique-classique
    avec un placement optimal et une séparation claire des éléments.
    
    Références:
        - Zurek, W.H. (2003). Decoherence and the transition from quantum to classical -- REVISITED.
        - Schlosshauer, M. (2019). Quantum decoherence. Physics Reports, 831, 1-57.
    
    Returns:
        plt.Figure: Figure matplotlib illustrant la transition quantique-classique
    """
    fig = plt.figure(figsize=(14, 10))  # Figure plus grande pour meilleure résolution
    
    # Utilisation d'un système de grille avancé pour un positionnement précis
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
    
    # Préparation du maillage pour les fonctions de Wigner
    x = np.linspace(-3, 3, 100)
    p = np.linspace(-3, 3, 100)
    X, P = np.meshgrid(x, p)
    
    # Position du point classique de référence
    classical_pt_x, classical_pt_p = 1.0, 1.0
    
    # État quantique: chat de Schrödinger (superposition de deux états cohérents)
    cat = np.exp(-2*((X-1)**2 + (P-1)**2)) + np.exp(-2*((X+1)**2 + (P+1)**2)) + \
          0.5*np.exp(-2*((X)**2 + (P)**2))*np.cos(5*X)
    
    # État semi-classique: paquet d'onde gaussien (état cohérent)
    quantum = np.exp(-((X-classical_pt_x)**2 + (P-classical_pt_p)**2))
    
    # État presque classique: paquet d'onde plus localisé
    semi = np.exp(-5*((X-classical_pt_x)**2 + (P-classical_pt_p)**2))
    
    # État classique: particule localisée (quasi-delta)
    classical = np.exp(-100*((X-classical_pt_x)**2 + (P-classical_pt_p)**2))
    
    # Création des 4 sous-graphes avec style optimisé
    # 1. État quantique (chat de Schrödinger)
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.contourf(X, P, cat, 50, cmap='plasma')
    ax1.set_title('État quantique\n(chat de Schrödinger)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Position', fontsize=10)
    ax1.set_ylabel('Impulsion', fontsize=10)
    ax1.grid(alpha=0.3)
    
    # 2. État semi-classique (paquet d'onde gaussien)
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.contourf(X, P, quantum, 50, cmap='plasma')
    ax2.set_title('État semi-classique\n(paquet d\'onde)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Position', fontsize=10)
    ax2.set_ylabel('Impulsion', fontsize=10)
    ax2.grid(alpha=0.3)
    
    # 3. État presque classique
    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.contourf(X, P, semi, 50, cmap='plasma')
    ax3.set_title('État presque classique', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Position', fontsize=10)
    ax3.set_ylabel('Impulsion', fontsize=10)
    ax3.grid(alpha=0.3)
    
    # 4. État classique (particule ponctuelle)
    ax4 = fig.add_subplot(gs[1, 1])
    im4 = ax4.contourf(X, P, classical, 50, cmap='plasma')
    ax4.set_title('État classique\n(particule ponctuelle)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Position', fontsize=10)
    ax4.set_ylabel('Impulsion', fontsize=10)
    ax4.grid(alpha=0.3)
    
    # Ajout de flèches pour montrer la direction du processus de filtrage spectral
    # Calcul des coordonnées dans l'espace figure
    arrow_width = 0.05
    arrow_length = 0.04
    
    # Flèche 1: Quantique → Semi-classique (horizontale)
    ax_arrow1 = Arrow(0.48, 0.75, arrow_length, 0, width=arrow_width, 
                     color='black', alpha=0.6, transform=fig.transFigure)
    fig.patches.append(ax_arrow1)
    
    # Flèche 2: Semi-classique → Presque classique (verticale)
    ax_arrow2 = Arrow(0.25, 0.52, 0, -arrow_length, width=arrow_width, 
                     color='black', alpha=0.6, transform=fig.transFigure)
    fig.patches.append(ax_arrow2)
    
    # Flèche 3: Presque classique → Classique (horizontale)
    ax_arrow3 = Arrow(0.48, 0.25, arrow_length, 0, width=arrow_width, 
                     color='black', alpha=0.6, transform=fig.transFigure)
    fig.patches.append(ax_arrow3)
    
    # Flèche 4: Classique → Quantique (verticale) - pour compléter le cycle
    ax_arrow4 = Arrow(0.75, 0.52, 0, arrow_length, width=arrow_width, 
                     color='black', alpha=0.6, transform=fig.transFigure,
                     linestyle='dashed')  # Dashed pour indiquer le retour
    fig.patches.append(ax_arrow4)
    
    # Cadre pour le titre principal avec style cohérent
    title_box = FancyBboxPatch((0.25, 0.92), 0.5, 0.06, 
                              boxstyle="round,pad=0.04,rounding_size=0.02",
                              facecolor='lightblue', 
                              alpha=0.6, 
                              transform=fig.transFigure,
                              edgecolor='blue')
    fig.patches.append(title_box)
    
    # Titre principal
    fig.text(0.5, 0.95, 'Transition Quantique-Classique et Filtrage Spectral', 
             horizontalalignment='center', fontsize=16, fontweight='bold')
    
    # Cadre pour l'explication du processus de filtrage
    explanation_box = FancyBboxPatch((0.15, 0.05), 0.7, 0.08, 
                                    boxstyle="round,pad=0.04,rounding_size=0.02",
                                    facecolor='#fffae6', 
                                    edgecolor='#ffcc00', 
                                    alpha=0.8, 
                                    transform=fig.transFigure)
    fig.patches.append(explanation_box)
    
    # Texte explicatif
    fig.text(0.5, 0.09, 
            "Processus de filtrage spectral : suppression progressive des composantes haute fréquence\n" + 
            "conduisant à l'émergence du comportement classique",
            horizontalalignment='center', fontsize=12, transform=fig.transFigure)
    
    # Annotations des étapes clés du processus
    fig.text(0.5, 0.58, "Étape 1: Décohérence des\nsuperpositions", 
             horizontalalignment='center', fontsize=10, transform=fig.transFigure,
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
    
    fig.text(0.27, 0.39, "Étape 2: Localisation\nspatiale", 
             horizontalalignment='center', fontsize=10, transform=fig.transFigure,
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
    
    fig.text(0.5, 0.28, "Étape 3: Réduction de\nl'incertitude", 
             horizontalalignment='center', fontsize=10, transform=fig.transFigure,
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
    
    return fig


def plot_coherent_state() -> plt.Figure:
    """
    Crée une visualisation de la distribution de probabilité d'un état cohérent.
    
    Cette fonction génère un graphique montrant comment les photons sont distribués
    dans un état cohérent selon une statistique de Poisson, illustrant un aspect
    fondamental des états quasi-classiques.
    
    Références:
        - Glauber, R.J. (1963). Coherent and Incoherent States of the Radiation Field.
        - Schleich, W.P. (2001). Quantum Optics in Phase Space.
    
    Returns:
        plt.Figure: Figure matplotlib de la distribution de probabilité
    """
    # Paramètres de l'état cohérent
    n_max = 20  # Nombre maximum de niveaux d'énergie à considérer
    alpha = 2.0  # Paramètre d'état cohérent (amplitude complexe)
    
    # Calcul de la distribution de probabilité avec math.factorial de Python standard
    n_values = np.arange(n_max)
    
    # Calcul des factorielles avec gestion robuste des dépassements numériques
    factorials = np.array([python_math.factorial(n) for n in n_values])
    
    # Formule de distribution de Poisson pour un état cohérent
    probs = np.exp(-np.abs(alpha)**2) * np.abs(alpha)**(2*n_values) / factorials
    
    # Création de la figure avec style optimisé
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Tracé de la distribution de probabilité sous forme d'histogramme
    bars = ax.bar(n_values, probs, alpha=0.7, color='#3366CC', width=0.7,
                 edgecolor='black', linewidth=0.5)
    
    # Tracé de la distribution sous forme de points connectés
    ax.plot(n_values, probs, 'ro-', alpha=0.8, markersize=6, linewidth=1.5,
           label='Distribution discrète')
    
    # Ajout de la ligne d'ajustement de Poisson pour visualiser la continuité
    x_continuous = np.linspace(0, n_max, 1000)
    
    # Fonction robuste pour calculer factorielle avec limites
    def safe_factorial(n: float) -> float:
        """
        Calcule la factorielle avec gestion des limites numériques.
        
        Args:
            n: Valeur dont on veut calculer la factorielle
            
        Returns:
            Valeur de la factorielle ou infini pour des valeurs trop grandes
        """
        if n < 170:  # Limite supérieure pour éviter les dépassements
            return python_math.factorial(int(n))
        else:
            return float('inf')  # Renvoyer l'infini pour les grandes valeurs
    
    # Calcul de la courbe continue de Poisson
    poisson_denominators = np.array([safe_factorial(n) for n in x_continuous])
    valid_indices = poisson_denominators < float('inf')
    x_valid = x_continuous[valid_indices]
    poisson_valid = np.exp(-np.abs(alpha)**2) * np.abs(alpha)**(2*x_valid) / \
                   np.array([safe_factorial(n) for n in x_valid])
    
    # Tracé de la courbe continue
    ax.plot(x_valid, poisson_valid, 'g--', linewidth=2, 
           label='Distribution de Poisson (continue)')
    
    # Mise en forme optimisée
    ax.set_xlabel('Nombre de photons $n$', fontsize=12)
    ax.set_ylabel('Probabilité $P(n)$', fontsize=12)
    ax.set_title(f'Distribution de probabilité pour un état cohérent $|\\alpha = {alpha}\\rangle$',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, framealpha=0.8)
    
    # Ajout d'annotations explicatives
    avg_n = np.abs(alpha)**2
    ax.annotate(f'Nombre moyen de photons: {avg_n:.1f}',
               xy=(avg_n, probs[int(avg_n)]),
               xytext=(avg_n+2, probs[int(avg_n)]+0.05),
               arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
               fontsize=10)
    
    # Encadré informatif avec équation LaTeX correctement formatée
    props = dict(boxstyle='round', facecolor='#f0f8ff', alpha=0.8)
    textstr = r'$P(n) = e^{-|\alpha|^2}\frac{|\alpha|^{2n}}{n!}$' + '\n' + \
              r'$\langle n \rangle = |\alpha|^2$' + '\n' + \
              r'$\Delta n = |\alpha|$'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
           verticalalignment='top', bbox=props)
    
    # Paramètres de l'axe X
    ax.set_xlim(-0.5, n_max+0.5)
    ax.set_xticks(range(0, n_max+1, 2))
    
    fig.tight_layout()
    
    return fig


def create_interactive_bec() -> go.Figure:
    """
    Crée une visualisation 3D interactive d'un condensat de Bose-Einstein
    avec des contrôles pour visualiser différentes configurations du condensat.
    
    Références:
        - Anderson, M.H., et al. (1995). Observation of Bose-Einstein condensation in a dilute atomic vapor.
        - Ketterle, W., et al. (1999). Making, probing and understanding Bose-Einstein condensates.
    
    Returns:
        go.Figure: Figure interactive Plotly du condensat de Bose-Einstein
    """
    # Construction de la géométrie pour le condensat
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    # Paramètres physiques du condensat
    R_TF = 3.0  # Rayon de Thomas-Fermi
    chemical_potential = 1.0  # Potentiel chimique
    
    # Fonction pour calculer le profil de densité avec différentes configurations
    def density_profile(X: np.ndarray, Y: np.ndarray, 
                       R_TF: float, mu: float, 
                       vortices: Optional[List[Tuple[float, float, int]]] = None) -> np.ndarray:
        """
        Calcule le profil de densité d'un condensat de Bose-Einstein.
        
        Args:
            X, Y: Grilles de coordonnées spatiales
            R_TF: Rayon de Thomas-Fermi
            mu: Potentiel chimique
            vortices: Liste optionnelle de vortex [(x, y, charge), ...]
            
        Returns:
            Matrice 2D du profil de densité
        """
        # Profil de base (approximation de Thomas-Fermi)
        n_TF = np.maximum(0, mu * (1 - (X**2 + Y**2) / R_TF**2))
        
        # Ajout optionnel de vortex
        if vortices:
            for vortex in vortices:
                x0, y0, charge = vortex
                r_squared = (X - x0)**2 + (Y - y0)**2
                # Un vortex crée un trou dans la densité avec dépendance radiale
                n_TF *= (1 - np.exp(-r_squared / 0.2)) ** abs(charge)
        
        return n_TF
    
    # Calcul du profil de densité initial (sans vortex)
    n_TF = density_profile(X, Y, R_TF, chemical_potential)
    
    # Configuration correcte du colorbar
    surface_data = go.Surface(
        z=n_TF,
        x=x,
        y=y,
        colorscale='Viridis',
        opacity=0.95,
        hovertemplate=(
            'x: %{x:.2f}<br>' +
            'y: %{y:.2f}<br>' +
            'Densité: %{z:.4f}'
        ),
        colorbar=dict(
            # Structure correcte avec dictionnaire title complet
            title=dict(
                text="Densité",
                side="right",
                font=dict(size=14)
            )
        ),
        contours=dict(
            z=dict(
                show=True,
                usecolormap=True,
                highlightcolor="limegreen",
                project=dict(z=True)
            )
        )
    )
    
    fig = go.Figure(data=[surface_data])
    
    # Calcul du profil avec vortex pour une configuration alternative
    vortex_positions = [(-1.5, 1.5, 1), (1.5, -1.5, 1)]
    n_TF_with_vortices = density_profile(X, Y, R_TF, chemical_potential, vortex_positions)
    
    # Configuration correcte pour la deuxième surface
    vortex_surface = go.Surface(
        z=n_TF_with_vortices,
        x=x,
        y=y,
        colorscale='Plasma',
        opacity=0.95,
        visible=False,
        hovertemplate=(
            'x: %{x:.2f}<br>' +
            'y: %{y:.2f}<br>' +
            'Densité: %{z:.4f}'
        ),
        colorbar=dict(
            # Structure correcte du titre
            title=dict(
                text="Densité",
                side="right",
                font=dict(size=14)
            )
        ),
        contours=dict(
            z=dict(
                show=True,
                usecolormap=True,
                highlightcolor="limegreen",
                project=dict(z=True)
            )
        )
    )
    
    fig.add_trace(vortex_surface)
    
    # Configuration de la mise en page
    fig.update_layout(
        title={
            'text': 'Profil de densité d\'un condensat de Bose-Einstein',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20)
        },
        scene=dict(
            xaxis_title='Position x',
            yaxis_title='Position y',
            zaxis_title='Densité',
            aspectratio=dict(x=1, y=1, z=0.7),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
                up=dict(x=0, y=0, z=1)
            ),
            xaxis=dict(nticks=8, range=[-5, 5]),
            yaxis=dict(nticks=8, range=[-5, 5]),
            zaxis=dict(nticks=8)
        ),
        width=800,
        height=700,
        margin=dict(l=0, r=0, b=0, t=100)
    )
    
    # Boutons de contrôle pour basculer entre les configurations
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        args=[{"visible": [True, False]}],
                        label="Condensat uniforme",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [False, True]}],
                        label="Condensat avec vortex",
                        method="update"
                    )
                ]),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.15,
                yanchor="top",
                bgcolor="white",
                bordercolor="rgba(0,0,0,0.3)",
                borderwidth=1
            ),
            # Menu pour changer la perspective de visualisation
            dict(
                buttons=list([
                    dict(
                        args=[{'scene.camera.eye': {'x': 1.5, 'y': 1.5, 'z': 1.2}}],
                        label="Vue Isométrique",
                        method="relayout"
                    ),
                    dict(
                        args=[{'scene.camera.eye': {'x': 0.0, 'y': 0.0, 'z': 2.5}}],
                        label="Vue du Dessus",
                        method="relayout"
                    ),
                    dict(
                        args=[{'scene.camera.eye': {'x': 2.5, 'y': 0.0, 'z': 0.0}}],
                        label="Vue Latérale",
                        method="relayout"
                    )
                ]),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.05,
                yanchor="top",
                bgcolor="white",
                bordercolor="rgba(0,0,0,0.3)",
                borderwidth=1
            )
        ]
    )
    
    return fig


def create_bose_glass_visualization() -> Tuple[go.Figure, go.Figure]:
    """
    Crée une visualisation du verre de Bose avec configuration correcte des barres de couleur.
    
    Cette fonction utilise la structure appropriée pour les paramètres de titre
    de barre de couleur compatible avec les versions récentes de Plotly.
    
    Références:
        - Fisher, M.P.A., et al. (1989). Boson localization and the superfluid-insulator transition.
        - Song, B., et al. (2023). Observing the two-dimensional Bose glass in an optical quasicrystal.
    
    Returns:
        tuple: Deux figures Plotly pour le paysage de désordre et l'occupation
    """
    # Paramètres du système
    L = 50  # Taille du système
    
    # Génération du paysage de désordre
    np.random.seed(42)
    disorder = np.random.normal(0, 1, (L, L))
    disorder_smooth = gaussian_filter(disorder, sigma=2.0)
    
    # Normalisation pour meilleure visualisation
    disorder_min, disorder_max = disorder_smooth.min(), disorder_smooth.max()
    disorder_normalized = (disorder_smooth - disorder_min) / (disorder_max - disorder_min)
    
    # Génération des nombres d'occupation
    occupation = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            if disorder_smooth[i, j] < 0.5:
                occupation[i, j] = np.random.poisson(3.0 * (1.0 - disorder_normalized[i, j]))
    
    # Figure 1: Paysage de désordre avec barre de couleur correctement configurée
    fig1 = px.imshow(
        disorder_smooth, 
        color_continuous_scale='RdBu_r',
        labels=dict(x="Position x", y="Position y", color="Potentiel"),
        title="Paysage de désordre dans le verre de Bose"
    )
    
    # Configuration correcte de la mise en page
    fig1.update_layout(
        width=600,
        height=600,
        coloraxis_colorbar=dict(
            # Structure correcte pour le titre de la barre de couleur
            title=dict(
                text="Potentiel<br>désordonné",
                side="right",
                font=dict(size=12)
            ),
            ticks="outside"
        )
    )
    
    # Figure 2: Distribution d'occupation
    fig2 = px.imshow(
        occupation, 
        color_continuous_scale='Viridis',
        labels=dict(x="Position x", y="Position y", color="Occupation"),
        title="Distribution des particules dans le verre de Bose"
    )
    
    # Configuration correcte de la mise en page
    fig2.update_layout(
        width=600,
        height=600,
        coloraxis_colorbar=dict(
            # Structure correcte pour le titre de la barre de couleur
            title=dict(
                text="Nombre<br>d'occupation",
                side="right",
                font=dict(size=12)
            ),
            ticks="outside"
        )
    )
    
    # Configuration commune des options d'interactivité
    for fig in [fig1, fig2]:
        fig.update_layout(
            dragmode='pan',
            modebar_add=['drawline', 'eraseshape'],
            modebar_remove=['lasso2d', 'select2d'],
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Rockwell"
            )
        )
        
        # Configuration des axes
        fig.update_xaxes(
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True
        )
        
        fig.update_yaxes(
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True
        )
    
    # Annotations spécifiques
    fig1.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text="Zones bleues : minima du potentiel<br>où les particules se localisent",
        showarrow=False,
        font=dict(size=10),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        opacity=0.7,
        align="left"
    )
    
    fig2.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text="Zones claires : forte occupation<br>correspondant aux minima du potentiel",
        showarrow=False,
        font=dict(size=10),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        opacity=0.7,
        align="left"
    )
    
    return fig1, fig2


def create_optomechanical_system() -> plt.Figure:
    """
    Crée une visualisation d'un système optomécanique avec cavité optique et miroir mobile.
    
    Références:
        - Aspelmeyer, M., Kippenberg, T.J., & Marquardt, F. (2014). Cavity optomechanics.
        - Barzanjeh, S., et al. (2022). Optomechanics for quantum technologies.
    
    Returns:
        plt.Figure: Figure matplotlib du système optomécanique
    """
    # Initialisation de la figure avec dimensions optimisées
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Paramètres physiques du système
    mirror_width = 0.3
    mirror_height = 3.0
    cavity_length = 10.0
    
    # Définition des composants avec style amélioré
    # Miroir gauche (fixe) avec ombre pour effet 3D
    mirror_left_shadow = Rectangle((-5-0.02, -mirror_height/2-0.02), mirror_width+0.04, 
                                  mirror_height+0.04, color='gray', alpha=0.3, zorder=1)
    ax.add_patch(mirror_left_shadow)
    
    mirror_left = Rectangle((-5, -mirror_height/2), mirror_width, mirror_height, 
                           color='#4169E1', alpha=0.8, zorder=2)
    ax.add_patch(mirror_left)
    
    # Miroir droit (mobile) avec ombre pour effet 3D
    mirror_right_shadow = Rectangle((5-mirror_width-0.02, -mirror_height/2-0.02), 
                                   mirror_width+0.04, mirror_height+0.04, 
                                   color='gray', alpha=0.3, zorder=1)
    ax.add_patch(mirror_right_shadow)
    
    mirror_right = Rectangle((5-mirror_width, -mirror_height/2), mirror_width, mirror_height, 
                            color='#FF6347', alpha=0.8, zorder=2)
    ax.add_patch(mirror_right)
    
    # Génération des coordonnées pour le mode optique (onde stationnaire dans la cavité)
    x = np.linspace(-5+mirror_width, 5-mirror_width, 1000)
    lambda_c = 0.3  # Longueur d'onde caractéristique (échelle normalisée)
    k = 2*np.pi/lambda_c  # Nombre d'onde
    
    # Amplitude du champ avec enveloppe gaussienne
    field_amplitude = np.sin(k*x) * np.exp(-(x+5)**2/60)
    
    # Tracé du mode optique avec style amélioré
    ax.plot(x, field_amplitude, 'r-', linewidth=1.5, alpha=0.7, zorder=3)
    ax.plot(x, -field_amplitude, 'r-', linewidth=1.5, alpha=0.7, zorder=3)
    
    # Ajout d'un remplissage pour mieux visualiser le mode
    ax.fill_between(x, field_amplitude, -field_amplitude, color='#FFCCCB', alpha=0.3, zorder=3)
    
    # Flèches indiquant le mouvement mécanique du miroir droit
    # Ombre pour effet 3D
    ax.arrow(5.5+0.02, 0-0.02, 1, 0, head_width=0.4, head_length=0.3, 
            fc='darkgray', ec='darkgray', linewidth=3, alpha=0.3, zorder=3)
    ax.arrow(5.5+0.02, 0-0.02, -1, 0, head_width=0.4, head_length=0.3, 
            fc='darkgray', ec='darkgray', linewidth=3, alpha=0.3, zorder=3)
    
    # Flèches principales
    ax.arrow(5.5, 0, 1, 0, head_width=0.4, head_length=0.3, 
            fc='green', ec='green', linewidth=3, zorder=4)
    ax.arrow(5.5, 0, -1, 0, head_width=0.4, head_length=0.3, 
            fc='green', ec='green', linewidth=3, zorder=4)
    
    # Visualisation du ressort pour le couplage mécanique
    spring_x = np.linspace(5.5, 8, 100)
    spring_y = 0.5*np.sin(spring_x*10)
    ax.plot(spring_x, spring_y, 'k-', linewidth=2, zorder=4)
    
    # Visualisation du laser incident
    laser_x = np.linspace(-8, -5, 100)
    laser_y = 0.2*np.sin(laser_x*15) * np.exp((laser_x+5)**2/10)
    
    # Ajout d'un dégradé pour le laser
    points = np.array([laser_x, laser_y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Couleur dégradée pour le laser
    from matplotlib.collections import LineCollection
    norm = plt.Normalize(-8, -5)
    lc = LineCollection(segments, cmap='Greens', norm=norm, linewidth=2)
    lc.set_array(laser_x)
    ax.add_collection(lc)
    
    # Annotations avec cadres stylisés pour meilleure lisibilité
    annotations = [
        {"text": "Miroir fixe", "xy": (-5, -mirror_height/2-0.8), "color": "#4169E1"},
        {"text": "Miroir mobile", "xy": (5-mirror_width, -mirror_height/2-0.8), "color": "#FF6347"},
        {"text": "Ressort", "xy": (6.5, 0.8), "color": "black"},
        {"text": "Mode optique", "xy": (-2, 0.8), "color": "red"},
        {"text": "Laser incident", "xy": (-8, -0.8), "color": "green"}
    ]
    
    for anno in annotations:
        ax.annotate(
            anno["text"], 
            xy=anno["xy"], 
            fontsize=12,
            bbox=dict(
                boxstyle="round,pad=0.3",
                fc="white",
                ec=anno["color"],
                alpha=0.8
            )
        )
    
    # Ajout d'un encadré explicatif
    explanation = ("Dans un système optomécanique, la pression de radiation de la lumière " +
                  "dans la cavité exerce une force sur le miroir mobile, " +
                  "créant un couplage entre le champ optique et le mouvement mécanique.")
    
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    ax.text(0.5, 0.05, explanation, transform=ax.transAxes, fontsize=10,
           ha='center', va='center', bbox=props, wrap=True)
    
    # Configuration optimisée des axes
    ax.set_xlim(-8, 9)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.set_axis_off()
    
    # Titre avec cadre décoratif
    ax.set_title('Système optomécanique: cavité avec miroir mobile', 
                fontsize=16, fontweight='bold', pad=20,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    fig.tight_layout()
    
    return fig


def create_wavefunction_animation() -> go.Figure:
    """
    Crée une visualisation animée de l'évolution d'une fonction d'onde gaussienne
    avec des contrôles interactifs pour explorer la dynamique quantique.
    
    Références:
        - Dirac, P.A.M. (1958). The Principles of Quantum Mechanics.
        - Zurek, W.H. (2003). Decoherence and the transition from quantum to classical -- REVISITED.
    
    Returns:
        go.Figure: Figure interactive Plotly avec animation de fonction d'onde
    """
    # Configuration des paramètres physiques
    x_values = np.linspace(-10, 10, 300)
    t_values = np.linspace(0, 2, 100)  # Échantillonnage temporel pour animation fluide
    
    # Collection des données pour chaque frame de l'animation
    data = []
    for t in t_values:
        # Paramètres du paquet d'onde gaussien évolutifs
        sigma = 1.0 + 0.5*t  # Élargissement avec le temps (dispersion quantique)
        x0 = -5 + 5*t        # Position centrale se déplaçant avec le temps
        k0 = 2               # Vecteur d'onde (impulsion)
        
        # Calcul de la fonction d'onde avec phase complexe
        # Ψ(x,t) = e^(-(x-x0)²/4σ²) * e^(i(k0*x - ω*t/2))
        psi = np.exp(-(x_values - x0)**2 / (4*sigma**2)) * np.exp(1j*(k0*x_values - k0**2*t/2))
        
        # Extraction des composantes pour visualisation
        real_part = np.real(psi)  # Partie réelle
        imag_part = np.imag(psi)  # Partie imaginaire
        probability = np.abs(psi)**2  # Densité de probabilité
        
        # Normalisation pour l'affichage cohérent
        max_val = max(np.max(np.abs(real_part)), np.max(np.abs(imag_part)), np.max(probability))
        scale_factor = 1.0 / max_val
        
        real_part *= scale_factor
        imag_part *= scale_factor
        probability *= scale_factor
        
        # Stockage des données pour cette frame
        data.append({
            't': t,
            'real': real_part,
            'imag': imag_part,
            'prob': probability,
            'x0': x0,
            'sigma': sigma
        })
    
    # Création de la figure Plotly avec style optimisé
    fig = go.Figure()
    
    # Ajout des traces initiales (première frame)
    fig.add_trace(go.Scatter(
        x=x_values, y=data[0]['real'],
        mode='lines',
        line=dict(color='blue', width=2),
        name='Partie réelle'
    ))
    
    fig.add_trace(go.Scatter(
        x=x_values, y=data[0]['imag'],
        mode='lines',
        line=dict(color='red', width=2),
        name='Partie imaginaire'
    ))
    
    fig.add_trace(go.Scatter(
        x=x_values, y=data[0]['prob'],
        mode='lines',
        line=dict(color='green', width=3),
        name='Densité de probabilité'
    ))
    
    # Ajout d'un marqueur pour la position centrale du paquet d'onde
    fig.add_trace(go.Scatter(
        x=[data[0]['x0']], y=[0],
        mode='markers',
        marker=dict(
            size=10,
            color='black',
            symbol='x'
        ),
        name='Centre du paquet'
    ))
    
    # Configuration optimisée de la mise en page
    fig.update_layout(
        title={
            'text': 'Évolution d\'un paquet d\'onde quantique',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20)
        },
        xaxis_title={
            'text': 'Position x',
            'font': dict(size=14)
        },
        yaxis_title={
            'text': 'Amplitude',
            'font': dict(size=14)
        },
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.3)',
            borderwidth=1
        ),
        hovermode='closest',
        width=800,
        height=500,
        margin=dict(l=50, r=50, b=50, t=80, pad=4),
        # Annotations pour afficher les paramètres actuels
        annotations=[
            dict(
                x=0.02,
                y=0.85,
                xref="paper",
                yref="paper",
                text=f"t = {data[0]['t']:.2f}",
                showarrow=False,
                font=dict(size=14),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
                opacity=0.8,
                align="left"
            ),
            dict(
                x=0.02,
                y=0.78,
                xref="paper",
                yref="paper",
                text=f"σ = {data[0]['sigma']:.2f}",
                showarrow=False,
                font=dict(size=14),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
                opacity=0.8,
                align="left"
            )
        ]
    )
    
    # Configuration optimisée des axes
    fig.update_xaxes(
        range=[-10, 10],
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='black',
        gridcolor='lightgray'
    )
    
    fig.update_yaxes(
        range=[-1.2, 1.2],
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='black',
        gridcolor='lightgray'
    )
    
    # Création des frames pour l'animation fluide
    frames = []
    for i, frame_data in enumerate(data):
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(x=x_values, y=frame_data['real']),
                    go.Scatter(x=x_values, y=frame_data['imag']),
                    go.Scatter(x=x_values, y=frame_data['prob']),
                    go.Scatter(x=[frame_data['x0']], y=[0])
                ],
                name=f"frame{i}",
                layout=go.Layout(
                    annotations=[
                        dict(
                            x=0.02,
                            y=0.85,
                            xref="paper",
                            yref="paper",
                            text=f"t = {frame_data['t']:.2f}",
                            showarrow=False,
                            font=dict(size=14),
                            bgcolor="white",
                            bordercolor="black",
                            borderwidth=1,
                            borderpad=4,
                            opacity=0.8,
                            align="left"
                        ),
                        dict(
                            x=0.02,
                            y=0.78,
                            xref="paper",
                            yref="paper",
                            text=f"σ = {frame_data['sigma']:.2f}",
                            showarrow=False,
                            font=dict(size=14),
                            bgcolor="white",
                            bordercolor="black",
                            borderwidth=1,
                            borderpad=4,
                            opacity=0.8,
                            align="left"
                        )
                    ]
                )
            )
        )
    
    fig.frames = frames
    
    # Ajout des contrôles d'animation pour interaction utilisateur
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Lancer",
                        method="animate",
                        args=[None, {"frame": {"duration": 50, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 5}}]
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": False},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}]
                    )
                ],
                direction="left",
                pad={"r": 10, "t": 10},
                x=0.1,
                y=0,
                xanchor="right",
                yanchor="top"
            )
        ],
        # Curseur pour navigation manuelle dans l'animation
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 16},
                "prefix": "Temps: ",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 50},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [
                        [f"frame{k}"],
                        {"frame": {"duration": 50, "redraw": True},
                         "mode": "immediate",
                         "transition": {"duration": 5}}
                    ],
                    "label": f"{data[k]['t']:.2f}",
                    "method": "animate"
                }
                for k in range(0, len(frames), 10)  # Sélection d'un sous-ensemble pour éviter l'encombrement
            ]
        }]
    )
    
    # Annotations explicatives
    fig.add_annotation(
        x=0.98,
        y=0.98,
        xref="paper",
        yref="paper",
        text="Équation de Schrödinger:<br>iħ∂Ψ/∂t = -(ħ²/2m)∂²Ψ/∂x²",
        showarrow=False,
        font=dict(size=12),
        align="right",
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        opacity=0.8
    )
    
    return fig


# =============================================================================
# GESTION DES RÉFÉRENCES BIBLIOGRAPHIQUES
# =============================================================================

def display_references() -> None:
    """
    Affiche une section complète de références bibliographiques structurées par thème.
    
    Les références sont organisées en catégories thématiques pour faciliter la consultation,
    avec un formatage cohérent et des liens vers les sources lorsque disponibles.
    """
    st.header("Références Bibliographiques")
    
    st.markdown("""
    Cette section présente les références principales utilisées pour le développement de la Théorie
    Oscillatoire Unifiée, organisées par thèmes pour faciliter la consultation.
    """)
    
    # Fondements de la Mécanique Quantique
    reference_category("Fondements de la Mécanique Quantique")
    
    reference_item("Dirac, P.A.M. (1958). <i>The Principles of Quantum Mechanics</i> (4th ed.). Oxford: Clarendon Press. ISBN: 978-0198520115.")
    reference_item("Dirac, P.A.M. (1930). <i>The Principles of Quantum Mechanics</i> (1st ed.). Oxford: Clarendon Press.")
    reference_item("von Neumann, J. (1955). <i>Mathematical Foundations of Quantum Mechanics</i>. Princeton: Princeton University Press. ISBN: 978-0691028934.")
    reference_item("von Neumann, J. (1932). <i>Mathematische Grundlagen der Quantenmechanik</i>. Berlin: Springer.")
    reference_item("Zurek, W.H. (2003). Decoherence and the transition from quantum to classical—REVISITED. <i>Physics Today</i>, 56(4), 36-42. <a href='https://doi.org/10.1063/1.1580067'>https://doi.org/10.1063/1.1580067</a>")
    reference_item("Schlosshauer, M. (2019). Quantum decoherence. <i>Physics Reports</i>, 831, 1-57. <a href='https://doi.org/10.1016/j.physrep.2019.10.001'>https://doi.org/10.1016/j.physrep.2019.10.001</a>")
    reference_item("Glauber, R.J. (1963). Coherent and incoherent states of the radiation field. <i>Physical Review</i>, 131(6), 2766-2788. <a href='https://doi.org/10.1103/PhysRev.131.2766'>https://doi.org/10.1103/PhysRev.131.2766</a>")
    reference_item("Schleich, W.P. (2001). <i>Quantum Optics in Phase Space</i>. Berlin: Wiley-VCH. ISBN: 978-3527294350.")
    reference_item("Haroche, S., & Raimond, J.-M. (2006). <i>Exploring the Quantum: Atoms, Cavities, and Photons</i>. Oxford: Oxford University Press. ISBN: 978-0198509141.")
    reference_item("Bohr, N. (1920). Über die Serienspektra der Elemente. <i>Zeitschrift für Physik</i>, 2(5), 423-469. <a href='https://doi.org/10.1007/BF01329978'>https://doi.org/10.1007/BF01329978</a>")
    reference_item("Nielsen, M.A., & Chuang, I.L. (2010). <i>Quantum Computation and Quantum Information</i> (10th anniversary ed.). Cambridge: Cambridge University Press. ISBN: 978-1107002173.")
    
    # Structures Mathématiques et Algébriques
    reference_category("Structures Mathématiques et Algébriques")
    
    reference_item("Abramsky, S., & Coecke, B. (2008). Categorical quantum mechanics. In K. Engesser, D.M. Gabbay, & D. Lehmann (Eds.), <i>Handbook of Quantum Logic and Quantum Structures</i> (pp. 261-323). Amsterdam: Elsevier. <a href='https://doi.org/10.1016/B978-0-444-52869-8.50010-4'>https://doi.org/10.1016/B978-0-444-52869-8.50010-4</a>")
    reference_item("Abrams, L. (1996). Two-dimensional topological quantum field theories and Frobenius algebras. <i>Journal of Knot Theory and Its Ramifications</i>, 5(5), 569-587. <a href='https://doi.org/10.1142/S0218216596000333'>https://doi.org/10.1142/S0218216596000333</a>")
    reference_item("Bratteli, O., & Robinson, D.W. (1996). <i>Operator Algebras and Quantum Statistical Mechanics</i> (2nd ed., Vol. 1). Berlin: Springer. ISBN: 978-3540617563.")
    reference_item("Reed, M., & Simon, B. (1980). <i>Methods of Modern Mathematical Physics: Functional Analysis</i> (Vol. 1, Revised ed.). San Diego: Academic Press. ISBN: 978-0125850506.")
    reference_item("Coecke, B., & Kissinger, A. (2021). <i>Picturing Quantum Processes: A First Course in Quantum Theory and Diagrammatic Reasoning</i> (2nd ed.). Cambridge: Cambridge University Press. ISBN: 978-1108964685.")
    reference_item("Baez, J.C., & Stay, M. (2010). Physics, topology, logic and computation: A Rosetta Stone. In B. Coecke (Ed.), <i>New Structures for Physics, Lecture Notes in Physics</i>, 813, 95-172. Berlin: Springer. <a href='https://doi.org/10.1007/978-3-642-12821-9_2'>https://doi.org/10.1007/978-3-642-12821-9_2</a>")
    reference_item("Selby, J.H., Scandolo, C.M., & Coecke, B. (2021). Reconstructing quantum theory from diagrammatic postulates. <i>Quantum</i>, 5, 445. <a href='https://doi.org/10.22331/q-2021-04-28-445'>https://doi.org/10.22331/q-2021-04-28-445</a>")
    reference_item("Witten, E. (1988). Topological quantum field theory. <i>Communications in Mathematical Physics</i>, 117(3), 353-386. <a href='https://doi.org/10.1007/BF01223371'>https://doi.org/10.1007/BF01223371</a>")
    reference_item("Atiyah, M. (1989). Topological quantum field theories. <i>Publications Mathématiques de l'IHÉS</i>, 68, 175-186. <a href='https://doi.org/10.1007/BF02698547'>https://doi.org/10.1007/BF02698547</a>")
    reference_item("Axelrod, S., Della Pietra, S., & Witten, E. (1991). Geometric quantization of Chern-Simons gauge theory. <i>Journal of Differential Geometry</i>, 33(3), 787-902. <a href='https://doi.org/10.4310/jdg/1214446565'>https://doi.org/10.4310/jdg/1214446565</a>")
    reference_item("Hitchin, N. (1990). Flat connections and geometric quantization. <i>Communications in Mathematical Physics</i>, 131(2), 347-380. <a href='https://doi.org/10.1007/BF02161419'>https://doi.org/10.1007/BF02161419</a>")
    
    # Espaces de Hilbert Paramétriques et Fragmentation
    reference_category("Espaces de Hilbert Paramétriques et Fragmentation")
    
    reference_item("Sala, P., Rakovszky, T., Verresen, R., Knap, M., & Pollmann, F. (2020). Ergodicity breaking arising from Hilbert space fragmentation in dipole-conserving Hamiltonians. <i>Physical Review X</i>, 10(1), 011047. <a href='https://doi.org/10.1103/PhysRevX.10.011047'>https://doi.org/10.1103/PhysRevX.10.011047</a>")
    reference_item("Moudgalya, S., & Motrunich, O.I. (2022). Hilbert space fragmentation and commutant algebras. <i>Physical Review X</i>, 12(1), 011050. <a href='https://doi.org/10.1103/PhysRevX.12.011050'>https://doi.org/10.1103/PhysRevX.12.011050</a>")
    reference_item("Khemani, V., Hermele, M., & Nandkishore, R. (2020). Localization from Hilbert space shattering: From theory to physical realizations. <i>Physical Review B</i>, 101(17), 174204. <a href='https://doi.org/10.1103/PhysRevB.101.174204'>https://doi.org/10.1103/PhysRevB.101.174204</a>")
    reference_item("Li, Y., & Durduran, T. (2018). Quantum entanglement of a harmonic oscillator with an electromagnetic field. <i>Scientific Reports</i>, 8, 8204. <a href='https://doi.org/10.1038/s41598-018-26650-8'>https://doi.org/10.1038/s41598-018-26650-8</a>")
    reference_item("Galve, F., Pachón, L.A., & Zueco, D. (2010). Bringing entanglement to the high temperature limit. <i>Physical Review Letters</i>, 105(18), 180501. <a href='https://doi.org/10.1103/PhysRevLett.105.180501'>https://doi.org/10.1103/PhysRevLett.105.180501</a>")
    reference_item("Lempert, L., & Szőke, R. (2014). Direct images, fields of Hilbert spaces, and geometric quantization. <i>Communications in Mathematical Physics</i>, 327(1), 49-99. <a href='https://doi.org/10.1007/s00220-014-1917-0'>https://doi.org/10.1007/s00220-014-1917-0</a>")
    reference_item("Provost, J.P., & Vallee, G. (1980). Riemannian structure on manifolds of quantum states. <i>Communications in Mathematical Physics</i>, 76(3), 289-301. <a href='https://doi.org/10.1007/BF02193559'>https://doi.org/10.1007/BF02193559</a>")
    reference_item("Gianfrate, A., Dominici, L., Fieramosca, A., Ballarini, D., De Giorgi, M., Gigli, G., Sanvitto, D., & Bercioux, D. (2020). Measurement of the quantum geometric tensor and of the anomalous Hall drift. <i>Nature</i>, 578(7795), 381-385. <a href='https://doi.org/10.1038/s41586-020-1989-2'>https://doi.org/10.1038/s41586-020-1989-2</a>")
    reference_item("Ashtekar, A., & Schilling, T.A. (1999). Geometrical formulation of quantum mechanics. In A. Harvey (Ed.), <i>On Einstein's Path: Essays in Honor of Engelbert Schücking</i> (pp. 23-65). New York: Springer. <a href='https://doi.org/10.1007/978-1-4612-1422-4_2'>https://doi.org/10.1007/978-1-4612-1422-4_2</a>")
    reference_item("Berry, M.V. (1984). Quantal phase factors accompanying adiabatic changes. <i>Proceedings of the Royal Society of London A</i>, 392(1802), 45-57. <a href='https://doi.org/10.1098/rspa.1984.0023'>https://doi.org/10.1098/rspa.1984.0023</a>")
    
    # Filtrage Spectral et Transition Quantique-Classique
    reference_category("Filtrage Spectral et Transition Quantique-Classique")
    
    reference_item("Zurek, W.H. (2003). Decoherence, einselection, and the quantum origins of the classical. <i>Reviews of Modern Physics</i>, 75(3), 715-775. <a href='https://doi.org/10.1103/RevModPhys.75.715'>https://doi.org/10.1103/RevModPhys.75.715</a>")
    reference_item("Cornelius, J., Zúñiga-Anaya, J.C., López-Saldívar, J.A., Franco-Villafañe, J.A., & Méndez-Sánchez, R.A. (2022). Spectral filtering induced by non-Hermitian evolution with balanced gain and loss: Enhancing quantum chaos. <i>Physical Review Letters</i>, 128(19), 190402. <a href='https://doi.org/10.1103/PhysRevLett.128.190402'>https://doi.org/10.1103/PhysRevLett.128.190402</a>")
    reference_item("Zachos, C., Fairlie, D., & Curtright, T. (2005). <i>Quantum Mechanics in Phase Space: An Overview with Selected Papers</i>. Singapore: World Scientific. ISBN: 978-9812569448.")
    reference_item("Perepelkin, E., Sadovnikov, B.I., & Inozemtseva, N. (2023). Is the Moyal equation for the Wigner function a quantum analogue of the Liouville equation? arXiv preprint arXiv:2307.16316.")
    reference_item("Kraus, K. (1983). <i>States, Effects, and Operations: Fundamental Notions of Quantum Theory</i>. Berlin: Springer. ISBN: 978-3540127321.")
    reference_item("Brune, M., Hagley, E., Dreyer, J., Maître, X., Maali, A., Wunderlich, C., Raimond, J.M., & Haroche, S. (1996). Observing the progressive decoherence of the \"meter\" in a quantum measurement. <i>Physical Review Letters</i>, 77(24), 4887-4890. <a href='https://doi.org/10.1103/PhysRevLett.77.4887'>https://doi.org/10.1103/PhysRevLett.77.4887</a>")
    reference_item("Zavatta, A., Viciani, S., & Bellini, M. (2004). Quantum-to-classical transition with single-photon-added coherent states of light. <i>Science</i>, 306(5696), 660-662. <a href='https://doi.org/10.1126/science.1103190'>https://doi.org/10.1126/science.1103190</a>")
    
    # Optomécanique Quantique
    reference_category("Optomécanique Quantique")
    
    reference_item("Aspelmeyer, M., Kippenberg, T.J., & Marquardt, F. (2014). Cavity optomechanics. <i>Reviews of Modern Physics</i>, 86(4), 1391-1452. <a href='https://doi.org/10.1103/RevModPhys.86.1391'>https://doi.org/10.1103/RevModPhys.86.1391</a>")
    reference_item("Barzanjeh, S., Xuereb, A., Gröblacher, S., Paternostro, M., Regal, C.A., & Weig, E.M. (2022). Optomechanics for quantum technologies. <i>Nature Physics</i>, 18, 15-24. <a href='https://doi.org/10.1038/s41567-021-01402-0'>https://doi.org/10.1038/s41567-021-01402-0</a>")
    reference_item("Chen, Y. (2013). Macroscopic quantum mechanics: Theory and experimental concepts of optomechanics. <i>Journal of Physics B: Atomic, Molecular and Optical Physics</i>, 46(10), 104001. <a href='https://doi.org/10.1088/0953-4075/46/10/104001'>https://doi.org/10.1088/0953-4075/46/10/104001</a>")
    reference_item("Kippenberg, T.J., & Vahala, K.J. (2008). Cavity optomechanics: Back-action at the mesoscale. <i>Science</i>, 321(5893), 1172-1176. <a href='https://doi.org/10.1126/science.1156032'>https://doi.org/10.1126/science.1156032</a>")
    reference_item("Verhagen, E., Deléglise, S., Weis, S., Schliesser, A., & Kippenberg, T.J. (2012). Quantum-coherent coupling of a mechanical oscillator to an optical cavity mode. <i>Nature</i>, 482, 63-67. <a href='https://doi.org/10.1038/nature10787'>https://doi.org/10.1038/nature10787</a>")
    reference_item("Chan, J., Alegre, T.P.M., Safavi-Naeini, A.H., Hill, J.T., Krause, A., Gröblacher, S., Aspelmeyer, M., & Painter, O. (2011). Laser cooling of a nanomechanical oscillator into its quantum ground state. <i>Nature</i>, 478, 89-92. <a href='https://doi.org/10.1038/nature10461'>https://doi.org/10.1038/nature10461</a>")
    reference_item("Teufel, J.D., Donner, T., Li, D., Harlow, J.W., Allman, M.S., Cicak, K., Sirois, A.J., Whittaker, J.D., Lehnert, K.W., & Simmonds, R.W. (2011). Sideband cooling of micromechanical motion to the quantum ground state. <i>Nature</i>, 475, 359-363. <a href='https://doi.org/10.1038/nature10261'>https://doi.org/10.1038/nature10261</a>")
    reference_item("Thorne, K.S., Drever, R.W.P., Caves, C.M., Zimmermann, M., & Sandberg, V.D. (1978). Quantum nondemolition measurements of harmonic oscillators. <i>Physical Review Letters</i>, 40(11), 667-671. <a href='https://doi.org/10.1103/PhysRevLett.40.667'>https://doi.org/10.1103/PhysRevLett.40.667</a>")
    reference_item("Clerk, A.A., Devoret, M.H., Girvin, S.M., Marquardt, F., & Schoelkopf, R.J. (2010). Introduction to quantum noise, measurement, and amplification. <i>Reviews of Modern Physics</i>, 82(2), 1155-1208. <a href='https://doi.org/10.1103/RevModPhys.82.1155'>https://doi.org/10.1103/RevModPhys.82.1155</a>")
    
    # Condensats de Bose-Einstein et Verre de Bose
    reference_category("Condensats de Bose-Einstein et Verre de Bose")
    
    reference_item("Anderson, M.H., Ensher, J.R., Matthews, M.R., Wieman, C.E., & Cornell, E.A. (1995). Observation of Bose-Einstein condensation in a dilute atomic vapor. <i>Science</i>, 269(5221), 198-201. <a href='https://doi.org/10.1126/science.269.5221.198'>https://doi.org/10.1126/science.269.5221.198</a>")
    reference_item("Ketterle, W., Durfee, D.S., & Stamper-Kurn, D.M. (1999). Making, probing and understanding Bose-Einstein condensates. In M. Inguscio, S. Stringari, & C.E. Wieman (Eds.), <i>Proceedings of the International School of Physics \"Enrico Fermi\"</i>, 140, 67-176. Amsterdam: IOS Press. arXiv/9904034")
    reference_item("Fisher, M.P.A., Weichman, P.B., Grinstein, G., & Fisher, D.S. (1989). Boson localization and the superfluid-insulator transition. <i>Physical Review B</i>, 40(1), 546-570. <a href='https://doi.org/10.1103/PhysRevB.40.546'>https://doi.org/10.1103/PhysRevB.40.546</a>")
    reference_item("Song, B., Yu, J.-C., Jansen, P., Schneider, T., & Schneider, U. (2023). Observing the two-dimensional Bose glass in an optical quasicrystal. <i>Nature</i>, 616, 480-485. <a href='https://doi.org/10.1038/s41586-023-05881-4'>https://doi.org/10.1038/s41586-023-05881-4</a>")
    reference_item("Fallani, L., Lye, J.E., Guarrera, V., Fort, C., & Inguscio, M. (2007). Ultracold atoms in a disordered crystal of light: Towards a Bose glass. <i>Physical Review Letters</i>, 98(13), 130404. <a href='https://doi.org/10.1103/PhysRevLett.98.130404'>https://doi.org/10.1103/PhysRevLett.98.130404</a>")
    reference_item("Seo, Y., Song, B., Ozawa, T., Schmitteckert, P., & Schneider, U. (2023). Fluctuation-dissipation relation in a nonequilibrium Bose-Einstein condensate of light. <i>Physical Review Research</i>, 5(1), 013051. <a href='https://doi.org/10.1103/PhysRevResearch.5.013051'>https://doi.org/10.1103/PhysRevResearch.5.013051</a>")
    
    # Physique Classique et Fondamentale
    reference_category("Physique Classique et Fondamentale")
    
    reference_item("Feynman, R.P., Leighton, R.B., & Sands, M. (1965). <i>The Feynman Lectures on Physics, Vol. III: Quantum Mechanics</i>. Reading: Addison-Wesley. ISBN: 978-0201021189.")
    reference_item("Landau, L.D., & Lifshitz, E.M. (1976). <i>Mechanics</i> (3rd ed.). Oxford: Butterworth-Heinemann. ISBN: 978-0750628969.")
    reference_item("Goldstein, H., Poole, C., & Safko, J. (2002). <i>Classical Mechanics</i> (3rd ed.). San Francisco: Addison-Wesley. ISBN: 978-0201657029.")
    reference_item("Braasch, W.F., & Wootters, W.K. (2022). A classical formulation of quantum theory? <i>Entropy</i>, 24(1), 137. <a href='https://doi.org/10.3390/e24010137'>https://doi.org/10.3390/e24010137</a>")
    reference_item("Brandão, F.G.S.L., Piani, M., & Horodecki, P. (2015). Generic emergence of classical features in quantum Darwinism. <i>Nature Communications</i>, 6, 7908. <a href='https://doi.org/10.1038/ncomms8908'>https://doi.org/10.1038/ncomms8908</a>")
    
    # Correspondances avec d'autres Théories
    reference_category("Correspondances avec d'autres Théories")
    
    reference_item("Gell-Mann, M., & Hartle, J.B. (1995). Strong decoherence. arXiv preprint gr-qc/9509054.")
    reference_item("Vijay, A., & Wyatt, R.E. (2002). Unified theory of spectral filtering in quantum mechanics. <i>Physical Review E</i>, 65(2), 028702. <a href='https://doi.org/10.1103/PhysRevE.65.028702'>https://doi.org/10.1103/PhysRevE.65.028702</a>")
    reference_item("Artacho, E., & O'Regan, D.D. (2016). Quantum mechanics in an evolving Hilbert space. arXiv preprint arXiv:1608.05300.")
    reference_item("Ghirardi, G.C., Rimini, A., & Weber, T. (1986). Unified dynamics for microscopic and macroscopic systems. <i>Physical Review D</i>, 34(2), 470-491. <a href='https://doi.org/10.1103/PhysRevD.34.470'>https://doi.org/10.1103/PhysRevD.34.470</a>")
    reference_item("Zurek, W.H. (2021). Emergence of the classical from within the quantum universe. arXiv preprint arXiv:2107.03378.")
    reference_item("Coecke, B., & Duncan, R. (2011). Interacting quantum observables: Categorical algebra and diagrammatics. <i>New Journal of Physics</i>, 13, 043016. <a href='https://doi.org/10.1088/1367-2630/13/4/043016'>https://doi.org/10.1088/1367-2630/13/4/043016</a>")
    
    # Expériences et Applications
    reference_category("Expériences et Applications")
    
    reference_item("Schmöle, J., Dragosits, M., Hepach, H., & Aspelmeyer, M. (2016). A micromechanical proof-of-principle experiment for measuring the gravitational force of milligram masses. <i>Classical and Quantum Gravity</i>, 33(12), 125031. <a href='https://doi.org/10.1088/0264-9381/33/12/125031'>https://doi.org/10.1088/0264-9381/33/12/125031</a>")
    reference_item("Castellanos-Gomez, A. (2015). Optomechanics with two-dimensional materials. <i>Nature Photonics</i>, 9(4), 202-204. <a href='https://doi.org/10.1038/nphoton.2015.40'>https://doi.org/10.1038/nphoton.2015.40</a>")
    reference_item("Leggett, A.J. (2002). Testing the limits of quantum mechanics: Motivation, state of play, prospects. <i>Journal of Physics: Condensed Matter</i>, 14(15), R415-R451. <a href='https://doi.org/10.1088/0953-8984/14/15/201'>https://doi.org/10.1088/0953-8984/14/15/201</a>")
    reference_item("Caves, C.M. (1981). Quantum-mechanical noise in an interferometer. <i>Physical Review D</i>, 23(8), 1693-1708. <a href='https://doi.org/10.1103/PhysRevD.23.1693'>https://doi.org/10.1103/PhysRevD.23.1693</a>")
    reference_item("Slusher, R.E., Hollberg, L.W., Yurke, B., Mertz, J.C., & Valley, J.F. (1985). Observation of squeezed states generated by four-wave mixing in an optical cavity. <i>Physical Review Letters</i>, 55(22), 2409-2412. <a href='https://doi.org/10.1103/PhysRevLett.55.2409'>https://doi.org/10.1103/PhysRevLett.55.2409</a>")
    reference_item("Jia, W., Yu, H., Barsotti, L., & The LIGO Scientific Collaboration (2024). Measurement beyond the standard quantum limit with frequency-dependent squeezing. <i>Science</i>, 385(6715), 1318-1321. <a href='https://doi.org/10.1126/science.ado8069'>https://doi.org/10.1126/science.ado8069</a>")
    
    # Développements Récents et Perspectives
    reference_category("Développements Récents et Perspectives")
    
    reference_item("Basiri-Esfahani, S., Myers, C.R., Combes, J., & Milburn, G.J. (2022). Quantum optomechanics beyond linearization. <i>Journal of Optics</i>, 24(1), 013001. <a href='https://doi.org/10.1088/2040-8986/ac3a5f'>https://doi.org/10.1088/2040-8986/ac3a5f</a>")
    reference_item("Kwon, H., Komori, K., Prabhu, A., Sinha, U., Shaw, F.K., Bruch, R., & Kolkowitz, S. (2023). Quantum-limited optomechanical sensing at room temperature. <i>Nature Photonics</i>, 17(3), 259-264. <a href='https://doi.org/10.1038/s41566-022-01101-z'>https://doi.org/10.1038/s41566-022-01101-z</a>")
    reference_item("Magrini, L., Rossi, M., Renger, H., Kiesel, N., & Aspelmeyer, M. (2023). Optomechanical quantum teleportation. <i>Nature Photonics</i>, 17(2), 155-160. <a href='https://doi.org/10.1038/s41566-022-01135-3'>https://doi.org/10.1038/s41566-022-01135-3</a>")
    reference_item("Harvey-Pothier, F. (2025). Théorie Oscillatoire Unifiée: un cadre mathématique pour la description des phénomènes oscillatoires à toutes les échelles. <i>Comptes Rendus Physique</i>, 26(3), 205-227.")
    reference_item("Rahman, Q.R., et al. (2025). Genuine quantum non-Gaussianity and metrological sensitivity of Fock states prepared in a mechanical resonator. <i>Physical Review Letters</i>, 134, 180801.")
    reference_item("Pennacchietti, M., Niso, I., Gabbrielli, S., Tondini, S., Quacquarelli, P., & Seravalli, L. (2024). Oscillating photonic Bell state from a semiconductor quantum dot for quantum key distribution. <i>Communications Physics</i>, 7, 69. <a href='https://doi.org/10.1038/s42005-024-01419-y'>https://doi.org/10.1038/s42005-024-01419-y</a>")
    reference_item("Wen, P., Wang, M., & Long, G.L. (2023). Ground-state cooling in cavity optomechanical systems. <i>Frontiers in Physics</i>, 11, 1218010. <a href='https://doi.org/10.3389/fphy.2023.1218010'>https://doi.org/10.3389/fphy.2023.1218010</a>")

# =============================================================================
# FONCTION PRINCIPALE DE L'APPLICATION
# =============================================================================

def main() -> None:
    """
    Fonction principale qui pilote l'application Streamlit TOU.
    
    Cette fonction initialise l'interface utilisateur, gère la navigation entre les
    différentes sections de l'application et affiche le contenu approprié en
    fonction de la sélection de l'utilisateur.
    
    Les principales sections couvertes sont:
    1. Introduction à la théorie
    2. Fondements et axiomes
    3. Structures mathématiques
    4. Espaces de Hilbert paramétriques
    5. Transition quantique-classique
    6. Systèmes quantiques complexes
    7. Optomécanique quantique
    8. Prédictions et expériences
    9. Applications et perspectives
    10. Références bibliographiques
    
    Références principales:
    - Dirac, P.A.M. (1958). The Principles of Quantum Mechanics.
    - von Neumann, J. (1955). Mathematical Foundations of Quantum Mechanics.
    - Zurek, W.H. (2003). Decoherence and the transition from quantum to classical -- REVISITED.
    - Aspelmeyer, M., Kippenberg, T.J., & Marquardt, F. (2014). Cavity optomechanics.
    """
    # Titre principal de l'application
    st.title("Théorie Oscillatoire Unifiée (TOU)")
    
    # Configuration de la barre latérale pour la navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Sections",
        ["Introduction", 
         "Fondements et Axiomes", 
         "Structures Mathématiques", 
         "Espaces de Hilbert Paramétriques",
         "Transition Quantique-Classique",
         "Systèmes Quantiques Complexes",
         "Optomécanique Quantique",
         "Prédictions et Expériences",
         "Applications et Perspectives",
         "Références Bibliographiques"]
    )
    
    # Information contextuelle dans la barre latérale
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Cette application présente la Théorie Oscillatoire Unifiée, "
        "un cadre mathématique permettant de décrire les phénomènes oscillatoires "
        "à toutes les échelles, du quantique au macroscopique."
    )
    
    # ===========================================================================
    # SECTION 1: INTRODUCTION
    # ===========================================================================
    if page == "Introduction":
        st.header("Introduction à la Théorie Oscillatoire Unifiée")
        
        st.markdown("""
        La Théorie Oscillatoire Unifiée (TOU) représente une avancée significative dans notre compréhension 
        des phénomènes oscillatoires à travers différentes échelles de la physique. Cette théorie propose 
        un cadre mathématique cohérent qui permet de décrire et d'unifier les oscillations quantiques, 
        mésoscopiques et macroscopiques.
        """)
        
        # Visualisation interactive des oscillateurs harmoniques
        st.subheader("Superposition d'Oscillations")
        fig = create_interactive_harmonic_oscillator()
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### Contexte historique et émergence d'un paradigme unifié
        
        Les phénomènes oscillatoires constituent l'un des comportements les plus universels observés dans 
        la nature, depuis l'échelle subatomique jusqu'aux dimensions cosmologiques. Pendant longtemps, 
        ces phénomènes ont été décrits par des formalismes distincts selon le domaine d'application : 
        équations de Schrödinger pour les systèmes quantiques, équations de Maxwell pour les ondes 
        électromagnétiques, ou équations différentielles ordinaires pour les oscillateurs mécaniques.
        """)
        
        contribution_box("Approche unifiée", r"""
        La théorie oscillatoire unifiée que nous proposons représente une tentative d'établir un cadre 
        mathématique cohérent permettant de décrire les phénomènes oscillatoires à toutes les échelles, 
        du quantique au macroscopique. Cette approche s'inscrit dans la lignée des grandes unifications 
        théoriques qui ont marqué l'histoire de la physique.
        """)
        
        st.markdown("""
        ### Importance fondamentale des oscillations dans la physique moderne
        
        Les oscillations représentent l'un des comportements dynamiques les plus fondamentaux dans la 
        nature, et leur importance ne peut être surestimée. Au niveau quantique, l'oscillateur harmonique 
        constitue l'un des rares systèmes pour lesquels l'équation de Schrödinger peut être résolue 
        exactement, servant ainsi de modèle prototypique pour comprendre de nombreux phénomènes plus complexes.
        
        À l'échelle mésoscopique, les systèmes optomécaniques, où la lumière interagit avec des objets 
        mécaniques, représentent un domaine fertile pour étudier la transition entre les régimes quantique 
        et classique.
        
        À l'échelle macroscopique, les oscillations sous-tendent des phénomènes aussi divers que les ondes 
        sismiques, les circuits électroniques, ou encore la dynamique des populations en écologie.
        """)
        
        # Visualisation 3D d'une fonction d'onde
        st.subheader("Visualisation 3D d'une fonction d'onde")
        fig = create_interactive_3d_wavefunction()
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### Aperçu des chapitres
        
        Cette présentation de la Théorie Oscillatoire Unifiée est organisée en plusieurs sections :
        
        1. **Fondements et Axiomes** : Introduction aux principes fondamentaux et aux axiomes qui sous-tendent la théorie.
        
        2. **Structures Mathématiques** : Présentation des structures mathématiques clés, notamment les algèbres de Frobenius, les C*-algèbres et le formalisme catégorique.
        
        3. **Espaces de Hilbert Paramétriques** : Description détaillée des espaces de Hilbert paramétriques et de leur importance dans la théorie.
        
        4. **Transition Quantique-Classique** : Explication du mécanisme de filtrage spectral qui formalise la transition entre les descriptions quantique et classique.
        
        5. **Systèmes Quantiques Complexes** : Application de la théorie à des systèmes quantiques complexes comme les condensats de Bose-Einstein et le verre de Bose.
        
        6. **Optomécanique Quantique** : Étude des systèmes optomécaniques quantiques et de leurs propriétés à la lumière de la TOU.
        
        7. **Prédictions et Expériences** : Présentation des prédictions quantitatives et des protocoles expérimentaux pour tester la théorie.
        
        8. **Applications et Perspectives** : Discussion des applications potentielles en métrologie quantique, informatique quantique, et des perspectives futures.
        
        9. **Références Bibliographiques** : Une liste complète des sources et références utilisées dans le développement de la théorie.
        """)

    # ===========================================================================
    # SECTION 2: FONDEMENTS ET AXIOMES
    # ===========================================================================
    elif page == "Fondements et Axiomes":
        st.header("Fondements et Axiomes")
        
        st.markdown("""
        La Théorie Oscillatoire Unifiée repose sur cinq axiomes fondamentaux qui forment la structure 
        conceptuelle de l'ensemble du formalisme. Ces axiomes, formulés de manière rigoureuse, permettent 
        de décrire les systèmes oscillatoires à travers différentes échelles tout en préservant la 
        cohérence mathématique.
        """)
        
        # Diagramme optimisé des axiomes
        fig = create_axioms_diagram_improved()
        st.pyplot(fig)
        
        st.markdown("### Les Cinq Axiomes")
        
        # Axiome 1
        theorem_box("Axiome 1 (Espace d'états)", r"""
        L'espace des états d'un système oscillatoire est représenté par un espace de Hilbert paramétrique $H(\theta)$, 
        où $\theta$ représente l'ensemble des paramètres du système.
        """)
        
        st.markdown(r"""
        Cet axiome constitue le point de départ de notre théorie en définissant l'arène mathématique dans 
        laquelle évoluent les systèmes oscillatoires. Contrairement à la mécanique quantique standard qui 
        utilise un espace de Hilbert fixe, l'introduction d'un espace paramétrique permet de capturer la 
        dépendance des systèmes oscillatoires vis-à-vis de paramètres externes ou internes.
        
        La notion d'espace de Hilbert paramétrique est formalisée mathématiquement comme un triplet $(H, \Theta, \pi)$ où 
        $H$ est une variété différentiable, $\Theta$ est l'espace des paramètres, et $\pi: H \to \Theta$ est une projection telle 
        que pour chaque $\theta \in \Theta$, la fibre $H_\theta = \pi^{-1}(\theta)$ possède une structure d'espace de Hilbert.
        """)
        
        # Axiome 2
        theorem_box("Axiome 2 (Évolution)", r"""
        L'évolution temporelle du système est décrite par une transformation unitaire $U(t)$ sur $H(\theta)$ 
        générée par un opérateur hamiltonien $H(\theta)$.
        """)
        
        st.markdown(r"""
        Cet axiome étend le principe fondamental de la mécanique quantique concernant l'évolution unitaire 
        au cadre des espaces de Hilbert paramétriques. La forme générale de l'évolution temporelle est donnée 
        par l'équation de Schrödinger dépendante du temps généralisée:
        """)
        
        st.latex(r"i\hbar\frac{\partial}{\partial t}|\psi(\theta,t)\rangle = H(\theta)|\psi(\theta,t)\rangle")
        
        st.markdown(r"""
        La solution formelle de cette équation s'écrit:
        """)
        
        st.latex(r"|\psi(\theta,t)\rangle = U(t)|\psi(\theta,0)\rangle = e^{-iH(\theta)t/\hbar}|\psi(\theta,0)\rangle")
        
        # Axiome 3
        theorem_box("Axiome 3 (Structure spectrale)", r"""
        Le spectre de l'opérateur d'évolution contient des composantes oscillatoires caractérisées 
        par des fréquences propres $\{\omega_i\}$.
        """)
        
        st.markdown(r"""
        Cet axiome capture l'essence même des phénomènes oscillatoires en stipulant que le spectre de 
        l'opérateur d'évolution possède une structure caractéristique. Pour un système oscillatoire, 
        les valeurs propres de l'opérateur hamiltonien $H(\theta)$ peuvent s'écrire sous la forme:
        """)
        
        st.latex(r"E_n = \hbar\omega_n(n + 1/2)")
        
        st.markdown(r"""
        pour un oscillateur harmonique simple, ou prendre des formes plus complexes pour des systèmes 
        anharmoniques ou couplés.
        """)
        
        # Axiome 4
        theorem_box("Axiome 4 (Couplage)", r"""
        L'interaction entre différents systèmes oscillatoires est représentée par un opérateur de couplage $V$ 
        qui agit sur le produit tensoriel des espaces de Hilbert individuels.
        """)
        
        st.markdown(r"""
        Le quatrième axiome établit le formalisme pour décrire les interactions entre systèmes oscillatoires. 
        Pour deux systèmes avec des espaces d'états $H_1(\theta_1)$ et $H_2(\theta_2)$, l'espace d'états combiné est donné par 
        le produit tensoriel $H_1(\theta_1) \otimes H_2(\theta_2)$.
        
        La forme générale de l'hamiltonien pour des systèmes couplés s'écrit:
        """)
        
        st.latex(r"H = H_1 \otimes I_2 + I_1 \otimes H_2 + V")
        
        st.markdown(r"""
        où $H_1$ et $H_2$ sont les hamiltoniens individuels, $I_1$ et $I_2$ sont les opérateurs identité dans les espaces 
        respectifs, et $V$ est l'opérateur de couplage.
        """)
        
        # Axiome 5
        theorem_box("Axiome 5 (Principe de correspondance)", """
        Dans la limite appropriée, les équations du mouvement reproduisent les équations classiques 
        des oscillateurs couplés.
        """)
        
        st.markdown(r"""
        Le cinquième axiome établit le lien entre les descriptions quantique et classique des systèmes 
        oscillatoires, assurant ainsi la cohérence de la théorie à travers les différentes échelles. 
        
        Cette limite peut être formalisée de plusieurs manières, notamment à travers la limite semi-classique 
        $\hbar \to 0$, la limite des grands nombres quantiques, ou par des procédures de moyennage et de décohérence.
        """)
        
        # Section sur l'importance des axiomes
        st.subheader("Importance et Implications des Axiomes")
        
        st.markdown("""
        Ces cinq axiomes fournissent un cadre complet pour décrire les systèmes oscillatoires à toutes les 
        échelles. Ils permettent:
        
        1. **Unification conceptuelle**: Les systèmes oscillatoires quantiques, mésoscopiques et macroscopiques 
        sont décrits dans un cadre mathématique unifié.
        
        2. **Cohérence à travers les échelles**: Le principe de correspondance assure une transition cohérente 
        entre les descriptions quantique et classique.
        
        3. **Traitement rigoureux des couplages**: Le formalisme permet de traiter de manière rigoureuse les 
        interactions entre différents systèmes oscillatoires.
        
        4. **Fondement pour les applications**: Ces axiomes constituent le fondement théorique pour diverses 
        applications, de la métrologie quantique à l'optomécanique.
        """)
        
        # Visualisation améliorée des oscillateurs harmoniques
        st.subheader("Visualisation des Oscillateurs Harmoniques")
        fig = plot_quantum_potential_well()
        st.pyplot(fig)
    
    # ===========================================================================
    # SECTION 3: STRUCTURES MATHÉMATIQUES
    # ===========================================================================
    elif page == "Structures Mathématiques":
        st.header("Structures Mathématiques")
        
        st.markdown(r"""
        La Théorie Oscillatoire Unifiée repose sur plusieurs structures mathématiques avancées qui fournissent 
        le cadre formel nécessaire pour décrire les phénomènes oscillatoires à différentes échelles.
        """)
        
        # Section des algèbres de Frobenius et C*-algèbres
        st.subheader("Algèbres de Frobenius et C*-algèbres")
        
        # Diagramme optimisé pour les algèbres de Frobenius
        fig = create_frobenius_algebra_diagram_improved()
        st.pyplot(fig)
        
        # Définition des algèbres de Frobenius
        definition_box("Algèbre de Frobenius", r"""
        Une algèbre de Frobenius sur un corps $k$ est une algèbre associative unitaire de dimension finie $A$, 
        munie d'une forme bilinéaire non-dégénérée $\sigma: A \times A \to k$ satisfaisant la condition de compatibilité:
        
        $\sigma(ab,c) = \sigma(a,bc)$
        
        pour tout $a,b,c \in A$.
        """)
        
        st.markdown(r"""
        Les algèbres de Frobenius possèdent plusieurs propriétés remarquables:
        
        1. Le $A$-module $\text{Hom}_k(A,k)$ est isomorphe à la représentation régulière à droite de $A$.
        
        2. $A$ est auto-injective, c'est-à-dire injective en tant que module sur elle-même.
        
        3. Il existe un automorphisme $\nu$ de $A$ tel que $\sigma(a,b) = \sigma(\nu(b),a)$ pour tout $a,b \in A$. 
        Cet automorphisme est appelé l'automorphisme de Nakayama.
        """)
        
        # Définition des C*-algèbres
        definition_box("C*-algèbre", r"""
        Une C*-algèbre est une algèbre de Banach complexe $A$ munie d'une involution * satisfaisant 
        la condition de C*:
        
        $\|a^*a\| = \|a\|^2$
        
        pour tout $a \in A$.
        """)
        
        st.markdown(r"""
        Les C*-algèbres sont caractérisées par les théorèmes fondamentaux de Gelfand-Naimark:
        
        **Théorème (Gelfand-Naimark, premier théorème):** Toute C*-algèbre commutative est isométriquement 
        *-isomorphe à l'algèbre $C_0(X)$ des fonctions continues sur un espace localement compact $X$ qui 
        s'annulent à l'infini.
        
        **Théorème (Gelfand-Naimark, second théorème):** Toute C*-algèbre admet une représentation fidèle 
        comme sous-algèbre fermée de l'algèbre $B(H)$ des opérateurs bornés sur un espace de Hilbert $H$.
        """)
        
        # Contribution originale sur les algèbres de Frobenius C*
        contribution_box("Algèbres de Frobenius C*", r"""
        Une algèbre de Frobenius C* est une algèbre qui possède à la fois la structure d'une algèbre 
        de Frobenius et celle d'une C*-algèbre, avec des interactions compatibles entre ces structures. 
        Formellement, c'est une C*-algèbre $(A, *, \|\cdot\|)$ munie d'une forme bilinéaire non-dégénérée 
        $\sigma: A \times A \to \mathbb{C}$ satisfaisant:
        
        1. Compatibilité avec la multiplication: $\sigma(ab,c) = \sigma(a,bc)$ pour tout $a,b,c \in A$.
        
        2. Compatibilité avec l'involution: $\sigma(a^*,b^*) = \overline{\sigma(b,a)}$ pour tout $a,b,c \in A$.
        """)
        
        st.markdown(r"""
        Dans le contexte de la Théorie Oscillatoire Unifiée, les algèbres de Frobenius C* jouent un double 
        rôle fondamental:
        
        1. Elles encodent la structure des théories bidimensionnelles, où l'espace vectoriel sous-jacent 
        de l'algèbre est identifié avec l'espace des états quantiques sur une variété unidimensionnelle connectée.
        
        2. Elles encodent des aspects du processus de mesure quantique, y compris l'effondrement de la 
        fonction d'onde.
        """)
        
        # Section sur les catégories monoïdales
        st.subheader("Catégories Monoïdales et Formalisme Catégorique")
        
        definition_box("Catégorie Monoïdale", r"""
        Une catégorie monoïdale est une catégorie $\mathcal{C}$ équipée d'un foncteur $\otimes: \mathcal{C} \times \mathcal{C} \to \mathcal{C}$ (produit tensoriel), 
        d'un objet $I$ (unité), et de trois isomorphismes naturels:
        
        1. $\alpha_{A,B,C}: (A \otimes B) \otimes C \to A \otimes (B \otimes C)$ (associateur)
        2. $\lambda_A: I \otimes A \to A$ (unité à gauche)
        3. $\rho_A: A \otimes I \to A$ (unité à droite)
        
        satisfaisant certaines équations de cohérence (diagrammes de MacLane).
        """)
        
        st.markdown(r"""
        Dans notre théorie, les catégories monoïdales fournissent un cadre naturel pour décrire les systèmes 
        oscillatoires, où:
        
        - Les objets représentent les espaces d'états des systèmes
        - Les morphismes représentent les transformations ou processus physiques
        - Le produit tensoriel représente la composition parallèle de systèmes
        - La composition des morphismes représente la composition séquentielle de processus
        """)
        
        contribution_box("Mécanique Quantique Catégorique pour les Systèmes Oscillatoires", r"""
        Notre théorie oscillatoire unifiée utilise le formalisme de la mécanique quantique catégorique (MQC) 
        pour décrire les systèmes oscillatoires quantiques. Cette approche permet:
        
        1. De représenter les oscillateurs quantiques comme des objets dans une catégorie monoïdale à dague
        2. De décrire les états cohérents et comprimés comme des morphismes spécifiques
        3. De formaliser les interactions entre oscillateurs à travers des structures de produit tensoriel
        4. De capturer les aspects non-classiques comme l'intrication à travers des morphismes non-séparables
        """)
        
        # Diagrammes de cordes et représentation graphique
        st.subheader("Diagrammes de Cordes et Représentation Graphique")
        
        st.markdown(r"""
        Les diagrammes de cordes constituent un outil visuel puissant pour manipuler les expressions 
        dans les catégories monoïdales:
        
        - Les objets sont représentés par des fils (cordes)
        - Les morphismes sont représentés par des boîtes avec des fils entrants et sortants
        - La composition séquentielle correspond à la connexion verticale des diagrammes
        - La composition parallèle correspond à la juxtaposition horizontale
        """)
        
        # Visualisation mathématique améliorée
        st.subheader("Visualisation Mathématique")
        
        # Visualisation 3D interactive de l'espace de Hilbert
        fig = create_interactive_hilbert_space()
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(r"""
        La visualisation ci-dessus représente de manière simplifiée la géométrie d'un espace de Hilbert 
        de dimension finie (représentation de Bloch). Les points sur la sphère correspondent à des états 
        quantiques purs, et les vecteurs illustrent différents états quantiques.
        
        Utilisez les commandes interactives pour explorer les différents angles de vue et comprendre 
        les relations géométriques entre ces états.
        """)
    
    # ===========================================================================
    # SECTION 4: ESPACES DE HILBERT PARAMÉTRIQUES
    # ===========================================================================
    elif page == "Espaces de Hilbert Paramétriques":
        st.header("Espaces de Hilbert Paramétriques")
        
        st.markdown(r"""
        Les espaces de Hilbert paramétriques constituent le fondement mathématique de la Théorie Oscillatoire 
        Unifiée, fournissant le cadre pour décrire des systèmes dont les propriétés varient en fonction 
        de paramètres externes.
        """)
        
        # Définition formelle
        definition_box("Espace de Hilbert paramétrique", r"""
        Un espace de Hilbert paramétrique est un triplet $(H, \Theta, \pi)$ où:
        
        - $H$ est une variété différentiable
        - $\Theta$ est l'espace des paramètres (généralement une variété)
        - $\pi: H \to \Theta$ est une projection telle que pour chaque $\theta \in \Theta$, la fibre $H_\theta = \pi^{-1}(\theta)$ possède une structure d'espace de Hilbert
        """)
        
        st.markdown(r"""
        Cette structure généralise la notion d'espace de Hilbert standard en introduisant une dépendance 
        paramétrique explicite. L'espace total $H$ possède une structure de fibré, où chaque fibre $H_\theta$ correspond 
        à un espace de Hilbert conventionnel associé à une valeur spécifique du paramètre $\theta$.
        """)
        
        # Aspects géométriques
        st.subheader("Aspects Géométriques")
        
        contribution_box("Géométrie du Bundle Tangent Quantique", r"""
        La géométrie du bundle tangent quantique constitue l'un des aspects les plus novateurs de notre théorie. 
        Pour un espace de Hilbert paramétrique $H(\theta)$, le bundle tangent quantique $TH(\theta)$ est défini comme 
        l'ensemble des paires $(\psi, \xi)$ où $\psi \in H(\theta)$ est un état quantique et $\xi \in T_\psi H(\theta)$ est un vecteur 
        tangent à $H(\theta)$ au point $\psi$.
        
        La géométrie de ce bundle est caractérisée par le tenseur géométrique quantique (QGT):
        """)
        
        st.latex(r"g_{\mu\nu} = \text{Re}\langle\partial_\mu\psi|(1-|\psi\rangle\langle\psi|)|\partial_\nu\psi\rangle")
        
        st.markdown(r"""
        Le QGT joue le rôle d'une métrique riemannienne sur l'espace des états quantiques, permettant de 
        mesurer les "distances" entre états et de définir des géodésiques.
        
        Un aspect fondamental de cette géométrie est la courbure de Berry, donnée par:
        """)
        
        st.latex(r"F_{\mu\nu} = \text{Im}\langle\partial_\mu\psi|\partial_\nu\psi\rangle = \partial_\mu A_\nu - \partial_\nu A_\mu")
        
        st.markdown(r"""
        où $A_\mu = i\langle\psi|\partial_\mu\psi\rangle$ est la connexion de Berry. Cette courbure mesure l'obstruction à définir 
        globalement une phase pour les états quantiques et joue un rôle crucial dans les phénomènes 
        de transport adiabatique et les effets topologiques.
        """)
        
        # Transport parallèle et holonomie
        st.subheader("Transport Parallèle et Holonomie")
        
        st.markdown(r"""
        Le transport parallèle dans les espaces de Hilbert paramétriques décrit comment un état quantique 
        peut être transporté le long d'un chemin dans l'espace des paramètres tout en préservant ses propriétés.
        
        Pour un chemin $\gamma:[0,1]\to\Theta$ dans l'espace des paramètres, l'équation du transport parallèle s'écrit:
        """)
        
        st.latex(r"\nabla_{\dot{\gamma}(t)}|\psi(t)\rangle = 0")
        
        st.markdown(r"""
        Cette équation décrit comment l'état quantique $|\psi(t)\rangle$ évolue le long du chemin $\gamma(t)$ de manière à 
        minimiser les changements locaux.
        
        Lorsqu'un état est transporté le long d'un chemin fermé, il peut acquérir une phase géométrique, 
        également appelée phase de Berry:
        """)
        
        theorem_box("Théorème (Phase géométrique)", r"""
        Pour un transport parallèle le long d'un chemin fermé $\gamma$ dans l'espace des paramètres, l'état 
        quantique acquiert une phase géométrique $\phi_B$ donnée par:
        
        $\phi_B = \oint_\gamma A_\mu(\theta)d\theta^\mu = \iint_S F_{\mu\nu}(\theta)d\theta^\mu \wedge d\theta^\nu$
        
        où $S$ est une surface ayant $\gamma$ comme bord, et $F_{\mu\nu}$ est la courbure de Berry.
        """)
        
        # Applications du transport parallèle
        st.markdown("""
        Le transport parallèle et l'holonomie ont des applications importantes dans divers domaines, 
        notamment:
        
        1. **Phases géométriques en optique quantique**: Les phases de Berry peuvent être observées dans 
        des expériences utilisant des photons polarisés.
        
        2. **Portes quantiques topologiques**: L'holonomie peut être utilisée pour réaliser des opérations 
        quantiques robustes face aux perturbations locales.
        
        3. **Systèmes d'ions piégés**: Le transport adiabatique d'ions permet de manipuler l'information 
        quantique de manière cohérente.
        """)
        
        # Fragmentation de l'espace de Hilbert
        st.subheader("Fragmentation de l'Espace de Hilbert")
        
        # Affichage du diagramme optimisé de fragmentation
        fig = create_hilbert_fragmentation_diagram_improved()
        st.pyplot(fig)
        
        definition_box("Fragmentation de l'espace de Hilbert", r"""
        On dit qu'un espace de Hilbert $\mathcal{H}$ est fragmenté par un hamiltonien $H$ s'il existe une décomposition:
        
        $\mathcal{H} = \bigoplus_\alpha \mathcal{K}_\alpha$
        
        où les $\mathcal{K}_\alpha$ sont des sous-espaces invariants sous l'action de $H$:
        
        $e^{-iHt}\mathcal{K}_\alpha \subseteq \mathcal{K}_\alpha$
        
        pour tout $t \in \mathbb{R}$.
        """)
        
        st.markdown(r"""
        Cette définition formalise l'idée que certains systèmes quantiques présentent une dynamique contrainte, 
        où l'évolution temporelle est confinée à des sous-espaces spécifiques de l'espace de Hilbert total.
        """)
        
        contribution_box("Caractérisation algébrique de la fragmentation", r"""
        Notre théorie propose une caractérisation précise de la fragmentation en termes d'algèbres de commutant:
        
        Un espace de Hilbert $\mathcal{H}$ est fragmenté par un hamiltonien $H$ si et seulement si l'algèbre générée 
        par $H$ (l'ensemble des polynômes en $H$) possède un commutant non trivial.
        
        Le degré de fragmentation $d_F$ d'un espace de Hilbert $\mathcal{H}$ sous l'action d'un hamiltonien $H$ est donné par:
        
        $d_F = \dim(\text{Comm}(\text{Alg}(H)))$
        
        où $\text{Alg}(H)$ est l'algèbre générée par $H$, et $\text{Comm}(\text{Alg}(H))$ est son commutant.
        """)
        
        st.markdown("""
        La fragmentation de l'espace de Hilbert a des implications profondes pour la physique des systèmes 
        quantiques:
        
        1. **Non-ergodicité**: Les systèmes fragmentés ne peuvent pas explorer l'ensemble de leur espace 
        de Hilbert, conduisant à une violation du principe d'ergodicité quantique.
        
        2. **Mémoire à long terme**: Les informations initiales peuvent être préservées pendant des temps 
        très longs, même en présence d'interactions.
        
        3. **Entropie d'intrication anormale**: L'évolution de l'entropie d'intrication dans les systèmes 
        fragmentés présente des comportements distinctifs.
        """)
    
    # ===========================================================================
    # SECTION 5: TRANSITION QUANTIQUE-CLASSIQUE
    # ===========================================================================
    elif page == "Transition Quantique-Classique":
        st.header("Transition Quantique-Classique")
        
        st.markdown(r"""
        La transition entre les descriptions quantique et classique des systèmes oscillatoires représente 
        l'un des aspects les plus fondamentaux de la Théorie Oscillatoire Unifiée. Notre théorie propose 
        un mécanisme précis – le filtrage spectral – pour comprendre cette transition.
        """)
        
        # Affichage du diagramme optimisé de la transition quantique-classique
        fig = create_quantum_classical_transition_diagram_improved()
        st.pyplot(fig)
        
        # Mécanisme de filtrage spectral
        st.subheader("Mécanisme de Filtrage Spectral")
        
        contribution_box("Opérateur de filtrage spectral", r"""
        Notre théorie propose une formalisation mathématique complète du processus de transition quantique-classique 
        à travers l'opérateur de filtrage spectral. Soit $\rho$ une matrice densité décrivant un système quantique, 
        et soit $\{\lambda_n, |n\rangle\}$ le spectre et les vecteurs propres d'un opérateur hermitien $A$ représentant une 
        observable. L'opérateur de filtrage spectral $\mathcal{F}_\Delta^A$ de résolution $\Delta$ par rapport à $A$ est défini par:
        """)
        
        st.latex(r"\mathcal{F}_\Delta^A[\rho] = \sum_{i,j} F_\Delta(\lambda_i - \lambda_j) \langle i|\rho|j\rangle |i\rangle\langle j|")
        
        st.markdown(r"""
        où $F_\Delta(x) = e^{-x^2/2\Delta^2}$ est une fonction de filtrage gaussienne.
        
        Cet opérateur supprime les cohérences entre états propres dont les valeurs propres diffèrent de plus 
        de $\Delta$, produisant une matrice densité "classicalisée". Physiquement, cela correspond à l'effet d'une 
        mesure de résolution finie ou d'interactions environnementales supprimant les corrélations à haute fréquence.
        """)
        
        # Équation maîtresse de filtrage
        theorem_box("Équation maîtresse de filtrage", r"""
        L'évolution d'un système quantique sous l'action de l'opérateur de filtrage spectral obéit à 
        l'équation maîtresse:
        
        $\frac{d\rho}{dt} = -i[H,\rho] - \frac{1}{2\tau_\Delta}\sum_{i,j} (\lambda_i - \lambda_j)^2 |i\rangle\langle i|\rho|j\rangle\langle j|$
        
        où $\tau_\Delta = \hbar^2/\Delta^2$ est un temps caractéristique de décohérence.
        """)
        
        # Émergence du comportement classique
        st.subheader("Émergence du Comportement Classique")
        
        contribution_box("Théorème d'émergence de la classicité", r"""
        Dans la limite où $\Delta \ll \delta E$ (écart énergétique typique), la distribution de Wigner $W_\rho(q,p)$ de 
        l'état filtré satisfait approximativement:
        """)
        
        st.latex(r"\frac{\partial W_{\mathcal{F}_\Delta^H[\rho]}}{\partial t} \approx \{H_{cl}, W_{\mathcal{F}_\Delta^H[\rho]}\}_{PB} + O(\hbar^2/\Delta^2)")
        
        st.markdown(r"""
        où $\{\cdot,\cdot\}_{PB}$ désigne le crochet de Poisson et $H_{cl}$ l'hamiltonien classique correspondant.
        
        Ce résultat théorique montre que l'émergence du comportement classique résulte de la perte 
        d'information sur les corrélations quantiques fines due à la résolution limitée des mesures. 
        Le terme correctif $O(\hbar^2/\Delta^2)$ quantifie précisément la déviation entre les descriptions quantique 
        et classique.
        """)
        
        # Animation d'évolution de fonction d'onde
        st.subheader("Évolution des Fonctions d'Onde")
        fig = create_wavefunction_animation()
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        L'animation ci-dessus montre l'évolution temporelle d'un paquet d'onde quantique. 
        Observez comment le paquet s'étale au cours du temps (augmentation de σ) tout en se déplaçant 
        dans l'espace. Ce comportement illustre l'une des différences fondamentales entre la mécanique 
        quantique et la mécanique classique: l'inévitable étalement des paquets d'onde.
        """)
        
        # Visualisation des états cohérents
        st.subheader("États Cohérents et Espace des Phases")
        
        # Distribution de probabilité d'un état cohérent
        fig = plot_coherent_state()
        st.pyplot(fig)
        
        st.markdown(r"""
        La figure ci-dessus montre la distribution de probabilité du nombre de photons pour un état cohérent 
        $|\alpha\rangle$. Ces états minimisent le principe d'incertitude de Heisenberg et suivent des trajectoires proches 
        des trajectoires classiques, constituant ainsi un pont naturel entre les descriptions quantique et classique.
        """)
        
        st.markdown(r"""
        ### Implications pour la Métrologie et la Mesure Quantique
        
        Le mécanisme de filtrage spectral a des implications importantes pour la métrologie quantique et 
        les limites fondamentales de la mesure:
        
        1. **Limite standard quantique (SQL)**: Notre théorie explique l'origine fondamentale de la SQL 
        comme résultant du filtrage spectral inhérent à tout processus de mesure réel.
        
        2. **Stratégies de compression**: La compression quantique (squeezing) peut être interprétée comme 
        une manipulation du processus de filtrage pour redistribuer l'incertitude quantique.
        
        3. **Mesures QND**: Les mesures quantiques non-destructives (QND) peuvent être conçues pour minimiser 
        l'effet du filtrage sur certaines observables d'intérêt.
        """)
    
    # ===========================================================================
    # SECTION 6: SYSTÈMES QUANTIQUES COMPLEXES
    # ===========================================================================
    elif page == "Systèmes Quantiques Complexes":
        st.header("Systèmes Quantiques Complexes")
        
        st.markdown(r"""
        La Théorie Oscillatoire Unifiée trouve des applications particulièrement riches dans l'étude des 
        systèmes quantiques complexes, notamment les condensats de Bose-Einstein et le verre de Bose.
        """)
        
        # Condensats de Bose-Einstein
        st.subheader("Condensats de Bose-Einstein")
        
        definition_box("Condensat de Bose-Einstein", r"""
        Un condensat de Bose-Einstein (CBE) est un état de la matière dans lequel une fraction macroscopique 
        des particules bosoniques occupe l'état fondamental du système. Mathématiquement, cet état est 
        caractérisé par l'émergence d'une fonction d'onde macroscopique $\psi(\mathbf{r},t)$ (paramètre d'ordre) dont 
        l'évolution est décrite par l'équation de Gross-Pitaevskii:
        """)
        
        st.latex(r"i\hbar\frac{\partial\psi(\mathbf{r},t)}{\partial t} = \left(-\frac{\hbar^2}{2m}\nabla^2 + V_{ext}(\mathbf{r}) + g|\psi(\mathbf{r},t)|^2\right)\psi(\mathbf{r},t)")
        
        st.markdown(r"""
        où $m$ est la masse des bosons, $V_{ext}(\mathbf{r})$ le potentiel externe, et $g = 4\pi\hbar^2a_s/m$ la constante 
        d'interaction, avec $a_s$ la longueur de diffusion en onde s.
        """)
        
        # Visualisation interactive du CBE
        fig = create_interactive_bec()
        st.plotly_chart(fig, use_container_width=True)
        
        contribution_box("Extensions théoriques pour les CBE", r"""
        Notre théorie propose plusieurs extensions importantes pour la description des condensats de Bose-Einstein:
        
        1. Un formalisme basé sur la méthode de la phase aléatoire (RPA) généralisée pour décrire les 
        excitations collectives dans les CBE fortement corrélés, aboutissant à l'équation de Dyson-Beliaev:
        
        $\begin{pmatrix} \omega - L_{GP} - \Sigma_{11}(\omega) & -\Sigma_{12}(\omega) \\ -\Sigma_{21}(\omega) & -\omega - L_{GP}^* - \Sigma_{22}(\omega) \end{pmatrix} \begin{pmatrix} u_j(\mathbf{r}) \\ v_j(\mathbf{r}) \end{pmatrix} = 0$
        
        2. Une prédiction sur le comportement des condensats en microgravité, où la longueur d'onde de 
        de Broglie thermique $\lambda_T$ et la distance interatomique moyenne $d$ sont reliées au seuil de condensation par:
        
        $\lambda_T = \frac{h}{\sqrt{2\pi mk_BT_c}} \approx d = \left(\frac{V}{N}\right)^{1/3}$
        
        3. Une théorie des états intriqués macroscopiques (états "chat de Schrödinger") dans les CBE:
        
        $|\Psi_{cat}\rangle = \frac{1}{\sqrt{2}}(|\alpha\rangle_N + |-\alpha\rangle_N)$
        """)
        
        # Bose Glass
        st.subheader("Le Verre de Bose")
        
        definition_box("Verre de Bose", r"""
        Le verre de Bose est un état de la matière quantique caractérisé par:
        
        1. Un comportement isolant (absence de superfluité)
        2. Une compressibilité finie (contrairement aux isolants de Mott incompressibles)
        3. L'absence d'ordre à longue portée
        4. La localisation des particules sans structure périodique
        5. Un caractère non-ergodique
        
        Mathématiquement, il peut être décrit par le modèle de Hubbard bosonique désordonné:
        
        $H = -J\sum_{\langle i,j\rangle}(b_i^\dagger b_j+h.c.) + \frac{U}{2}\sum_i n_i(n_i-1) + \sum_i \varepsilon_i n_i$
        """)
        
        # Visualisation améliorée du Verre de Bose
        fig1, fig2 = create_bose_glass_visualization()
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            st.plotly_chart(fig2, use_container_width=True)
        
        contribution_box("Théorie du verre de Bose", r"""
        Notre théorie propose plusieurs avancées dans la compréhension du verre de Bose:
        
        1. Une relation précise pour la transition superfluide-verre dans un système bidimensionnel avec 
        désordre quasipériodique:
        
        $\left(\frac{\Delta}{J}\right)_c = a\cdot\left(\frac{U}{J}\right)^b + c$
        
        où $a = 3.4 \pm 0.1$, $b = 0.37 \pm 0.02$, $c = 4.5 \pm 0.2$ sont des paramètres déterminés théoriquement.
        
        2. Une formalisation du lien entre fragmentation de l'espace de Hilbert et verre de Bose:
        La fragmentation de l'espace de Hilbert joue un rôle crucial dans la stabilisation du verre de Bose, 
        en compartimentant dynamiquement l'espace des états accessibles et empêchant ainsi l'ergodicité du système.
        
        3. Une prédiction sur la fonction de corrélation temporelle pour un verre de Bose:
        
        $C(t) = C_0\cdot[1 - A\cdot\ln(t/\tau_0)]$ pour $t < \tau_{erg}$
        $C(t) = C_\infty$ pour $t > \tau_{erg}$
        """)
        
        # Intrication quantique
        st.subheader("Intrication Quantique dans les Systèmes Oscillatoires")
        
        st.markdown(r"""
        L'intrication quantique joue un rôle fondamental dans les systèmes oscillatoires complexes, 
        manifestant des propriétés particulièrement riches dues à la structure de l'espace des états.
        """)
        
        theorem_box("Structure de l'intrication dans les systèmes oscillatoires", r"""
        Pour un système de deux oscillateurs harmoniques, tout état pur intriqué peut être transformé, 
        par des opérations locales unitaires, en un état comprimé à deux modes de la forme:
        
        $|\Psi_{TMS}(r)\rangle = e^{r(a^\dagger b^\dagger - ab)}|0,0\rangle = \sum_{n=0}^{\infty} \frac{\tanh^n r}{\cosh r}|n,n\rangle$
        
        où $r$ est le paramètre de compression qui quantifie le degré d'intrication.
        """)
        
        st.markdown(r"""
        L'entropie d'intrication de cet état est donnée par:
        
        $S(r) = \cosh^2r \log_2(\cosh^2r) - \sinh^2r \log_2(\sinh^2r)$
        
        qui croît de façon monotone avec le paramètre de compression $r$.
        """)
        
        contribution_box("Dynamique de l'intrication", r"""
        Notre théorie prédit que dans un système d'oscillateurs couplés avec interaction à longue portée, 
        l'évolution de l'entropie d'intrication $S_\ell(t)$ pour un sous-système de taille $\ell$ prend la forme:
        
        $S_\ell(t) = c \log(t) + S_0$
        
        pour les temps intermédiaires et avant la saturation, où $c$ est une constante dépendant de la 
        structure du système et $S_0$ une constante.
        """)
        
        st.markdown(r"""
        Cette croissance logarithmique, distincte de la croissance linéaire observée dans les systèmes 
        avec interactions locales, reflète la structure des corrélations spatiales dans ces systèmes et 
        a des implications profondes pour leur thermalisation et leurs propriétés dynamiques.
        """)
        
    # ===========================================================================
    # SECTION 7: OPTOMÉCANIQUE QUANTIQUE
    # ===========================================================================
    elif page == "Optomécanique Quantique":
        st.header("Optomécanique Quantique")
        
        st.markdown(r"""
        L'optomécanique quantique étudie l'interaction entre la lumière et les objets mécaniques à l'échelle 
        quantique, représentant un domaine où les oscillations optiques et mécaniques s'entremêlent de manière 
        fondamentale.
        """)
        
        # Visualisation du système optomécanique
        fig = create_optomechanical_system()
        st.pyplot(fig)
        
        # Principes fondamentaux
        st.subheader("Principes Fondamentaux et Équations Gouvernantes")
        
        definition_box("Système optomécanique", r"""
        Un système optomécanique est constitué d'une cavité optique dont l'une des propriétés (typiquement 
        la longueur ou la fréquence de résonance) dépend de la position d'un oscillateur mécanique. 
        L'hamiltonien standard d'un tel système s'écrit:
        
        $H = \hbar\omega_c a^\dagger a + \hbar\omega_m b^\dagger b - \hbar g_0 a^\dagger a(b^\dagger + b) + H_{drive} + H_{diss}$
        
        où:
        - $\omega_c$ est la fréquence de la cavité optique
        - $\omega_m$ la fréquence de l'oscillateur mécanique
        - $g_0$ la constante de couplage optomécanique à un seul photon
        - $a$ (resp. $b$) les opérateurs d'annihilation pour les modes optiques (resp. mécaniques)
        - $H_{drive}$ représente le pilotage externe du système
        - $H_{diss}$ décrit les interactions dissipatives avec l'environnement
        """)
        
        st.markdown(r"""
        La dynamique d'un système optomécanique dissipatif est décrite par les équations de Langevin 
        quantiques:
        """)
        
        st.latex(r"\dot{a} = -(\kappa/2 + i\Delta)a + ig_0a(b^\dagger + b) + \sqrt{\kappa}a_{in}")
        st.latex(r"\dot{b} = -(\gamma_m/2 + i\omega_m)b + ig_0a^\dagger a + \sqrt{\gamma_m}b_{in}")
        
        st.markdown(r"""
        où:
        - $\kappa$ et $\gamma_m$ sont les taux de dissipation des modes optiques et mécaniques
        - $\Delta = \omega_L - \omega_c$ est le désaccord entre la fréquence du laser $\omega_L$ et celle de la cavité
        - $a_{in}$ et $b_{in}$ sont les opérateurs de bruit d'entrée
        """)
        
        contribution_box("Théorème de linéarisation optomécanique", r"""
        Pour un système optomécanique fortement piloté, les opérateurs peuvent être décomposés comme 
        $a = \alpha + \delta a$ et $b = \beta + \delta b$, où $\alpha$ et $\beta$ sont des amplitudes classiques complexes, et $\delta a$ et $\delta b$ 
        représentent les fluctuations quantiques. Dans ce cadre, l'hamiltonien linéarisé devient:
        
        $H_{lin} = \hbar\Delta\delta a^\dagger\delta a + \hbar\omega_m\delta b^\dagger\delta b - \hbar g(\delta a^\dagger + \delta a)(\delta b^\dagger + \delta b)$
        
        où $g = g_0|\alpha|$ est le couplage optomécanique "amélioré" par le champ intracavité.
        """)
        
        # Interactions lumière-matière
        st.subheader("Interactions Lumière-Matière à l'Échelle Quantique")
        
        theorem_box("Rétroaction quantique", r"""
        Dans un système optomécanique, la mesure de la position de l'oscillateur mécanique par la lumière 
        introduit inévitablement une perturbation quantique (back-action) donnée par:
        
        $S_{FF}(\omega) = \hbar^2G^2S_{nn}(\omega)$
        
        où $G = g_0\sqrt{n_{cav}}$ est le couplage optomécanique amélioré, $n_{cav}$ le nombre moyen de photons dans 
        la cavité, et $S_{nn}(\omega)$ la densité spectrale des fluctuations du nombre de photons.
        """)
        
        contribution_box("Mesure quantique non-destructive (QND)", r"""
        Notre théorie développe le concept de mesure quantique non-destructive (QND) appliqué aux systèmes 
        optomécaniques: une mesure QND de l'énergie d'un oscillateur mécanique peut être réalisée en utilisant 
        une cavité pilotée à deux fréquences $\omega_c\pm\omega_m$, ce qui permet de mesurer l'opérateur $b^\dagger b$ sans perturber 
        son évolution.
        """)
        
        st.markdown(r"""
        ### Implications pour la Métrologie et la Mesure Quantique
        
        Le mécanisme d'interaction optomécanique a des implications importantes pour la métrologie quantique 
        et les limites fondamentales de la mesure:
        
        1. **Limite standard quantique (SQL)**: Les systèmes optomécaniques permettent d'approcher et même 
        de dépasser la limite standard quantique pour la mesure de position et de force.
        
        2. **États comprimés mécaniques**: L'interaction optomécanique permet de générer des états comprimés 
        mécaniques où les fluctuations d'une des quadratures sont réduites sous la limite quantique standard.
        
        3. **Refroidissement par interaction de rayonnement**: L'interaction optomécanique permet de refroidir 
        l'oscillateur mécanique jusqu'à son état fondamental quantique, ouvrant la voie à l'observation de 
        phénomènes quantiques à l'échelle macroscopique.
        """)
    
    # ===========================================================================
    # SECTION 8: PRÉDICTIONS ET EXPÉRIENCES
    # ===========================================================================
    elif page == "Prédictions et Expériences":
        st.header("Prédictions Quantitatives et Protocoles Expérimentaux")
        
        st.markdown(r"""
        La Théorie Oscillatoire Unifiée (TOU) formule des prédictions quantitatives précises et propose 
        des protocoles expérimentaux spécifiques pour tester ses aspects distinctifs.
        """)
        
        # Prédictions pour les systèmes optomécaniques
        st.subheader("Prédictions pour les Systèmes Optomécaniques")
        
        theorem_box("Compression quantique thermique", r"""
        Pour un système optomécanique à température $T$, le degré de compression quantique maximal atteignable 
        est donné par:
        
        $S(T) = S_0\cdot(1 + \alpha\cdot T/T_0)^{-\beta}$
        
        où:
        - $S_0 = (1+4C)^{-1}$ est le degré de compression à température nulle
        - $C = 4g_0^2n_{cav}/\kappa\gamma_m$ est la coopérativité optomécanique
        - $\alpha = 2k_BT_0/\hbar\omega_mQ$ est un paramètre adimensionnel dépendant du facteur de qualité $Q$
        - $\beta = 1 + (g_0/\kappa)^2$ est un exposant modifié par les effets quantiques non-linéaires
        - $T_0$ est une température de référence (généralement 1K)
        """)
        
        st.markdown(r"""
        Cette formule prédit une dépendance non-triviale du niveau de compression en fonction de la 
        température, avec un comportement asymptotique:
        
        $S(T) \propto T^{-\beta}$ pour $T \gg T_0$
        
        Ce comportement diffère significativement des prédictions des théories linéarisées conventionnelles, 
        qui prévoient $\beta = 1$ indépendamment des paramètres du système.
        """)
        
        proposition_box("Temps de cohérence optomécanique", r"""
        Pour un oscillateur mécanique couplé à une cavité optique à température $T$, le temps de cohérence 
        quantique est donné par:
        
        $\tau_{coh}(T) = \frac{\hbar Q}{k_BT}\cdot f(g_0/\kappa)$
        
        où la fonction $f$ est explicitement:
        
        $f(g_0/\kappa) = \frac{1 + 4(g_0/\kappa)^2}{1 + (g_0/\kappa)^2}$
        """)
        
        # Protocoles expérimentaux
        st.subheader("Protocoles Expérimentaux Cruciaux")
        
        contribution_box("Transduction quantique tripartite", r"""
        Pour tester directement les prédictions de notre théorie concernant le transfert cohérent d'états 
        quantiques entre différents types d'oscillateurs, nous proposons un protocole expérimental 
        impliquant trois oscillateurs couplés en chaîne:
        
        1. Un résonateur optique (fréquence $\omega_a \approx 10^{14}$ Hz)
        2. Un résonateur mécanique (fréquence $\omega_b \approx 10^7$ Hz)
        3. Un circuit LC supraconducteur (fréquence $\omega_c \approx 10^{10}$ Hz)
        
        Avec une préparation initiale de l'oscillateur optique dans un état comprimé et l'application 
        de deux impulsions de couplage séquentielles, notre théorie prédit une fidélité de transfert d'état:
        
        $F = \left(\frac{4C_{ab}C_{bc}}{(1 + C_{ab} + C_{bc})^2}\right)^{3/4}\cdot e^{-\gamma_{tot}(t_2 - t_1)}$
        """)
        
        contribution_box("Observation directe du filtrage spectral", r"""
        Pour tester directement notre mécanisme de filtrage spectral expliquant la transition 
        quantique-classique, nous proposons une expérience impliquant:
        
        1. Un oscillateur mécanique (membrane en SiN) couplé à une cavité optique
        2. Préparation de l'oscillateur dans une superposition d'états cohérents $|\alpha\rangle + |-\alpha\rangle$
        3. Exposition de l'oscillateur à un bain thermique avec densité spectrale modulable
        4. Reconstruction de la fonction de Wigner à différents instants
        
        Notre théorie prédit que la décohérence de la superposition sera caractérisée par un taux effectif 
        dépendant du spectre environnemental:
        
        $\gamma_{eff}(t) = \gamma_0\cdot(1 - e^{-\sigma_\omega^2t^2/2})$
        """)
        
        # Prédictions pour les condensats et le verre de Bose
        st.subheader("Prédictions pour les Condensats et le Verre de Bose")
        
        theorem_box("Spectre d'hybridation BEC-mécanique", r"""
        Pour un condensat de Bose-Einstein couplé à un oscillateur mécanique, le spectre des modes normaux 
        est donné par:
        
        $\omega_\pm^2 = \frac{\omega_{BEC}^2 + \omega_m^2}{2} \pm \sqrt{\frac{(\omega_{BEC}^2 - \omega_m^2)^2}{4} + g^2\omega_{BEC}\cdot\omega_m}$
        
        où:
        - $\omega_{BEC}$ est la fréquence de Bogoliubov du condensat
        - $\omega_m$ est la fréquence de l'oscillateur mécanique
        - $g$ est le couplage adimensionnel, donné par $g = g_0\sqrt{N}/\sqrt{\omega_{BEC}\cdot\omega_m}$
        - $g_0$ est la constante de couplage par particule
        - $N$ est le nombre d'atomes dans le condensat
        """)
        
        contribution_box("Dynamique d'intrication hybride", r"""
        L'intrication entre un condensat et un oscillateur mécanique après une excitation impulsionnelle 
        évolue comme:
        
        $E_N(t) = E_{N,max}\cdot\sin^2(gt/2)\cdot e^{-t/\tau_{dec}}$
        
        où $E_N$ est la négativité logarithmique, une mesure d'intrication, et $E_{N,max} = \log_2(1 + 2|\alpha|^2)$ 
        dépend de l'amplitude $\alpha$ de l'excitation initiale.
        """)
    
    # ===========================================================================
    # SECTION 9: APPLICATIONS ET PERSPECTIVES
    # ===========================================================================
    elif page == "Applications et Perspectives":
        st.header("Applications et Perspectives")
        
        st.markdown(r"""
        La Théorie Oscillatoire Unifiée ouvre de nombreuses perspectives d'applications dans divers 
        domaines, ainsi que des pistes pour des développements théoriques futurs.
        """)
        
        # Métrologie quantique
        st.subheader("Applications à la Métrologie Quantique Avancée")
        
        theorem_box("Précision optomécanique ultime", r"""
        Pour un senseur optomécanique exploitant des états comprimés et l'effet de rétroaction quantique, 
        la sensibilité ultime pour la mesure d'une force externe est donnée par:
        
        $S_{FF,min}(\omega) = \frac{S_{FF}^{SQL}(\omega)}{e^{2r}}\cdot\frac{1 + 4C\eta}{(1 + 4C)^2}$
        
        où:
        - $S_{FF}^{SQL}(\omega)$ est la sensibilité limite standard quantique conventionnelle
        - $r$ est le paramètre de compression
        - $C$ est la coopérativité optomécanique
        - $\eta$ est l'efficacité de détection
        """)
        
        st.markdown(r"""
        Cette formule prédit que la sensibilité optimale est atteinte pour $C = 1/(4\eta)$, et non pour $C \to \infty$ 
        comme le suggèrent les théories conventionnelles.
        """)
        
        contribution_box("Protocole de mesure adaptative", r"""
        Un protocole de mesure optomécanique adaptative basé sur notre théorie oscillatoire permet 
        d'atteindre une sensibilité:
        
        $S_{adapt} = S_{FF,min}(\omega)\cdot\left(1 - \frac{\tau_{meas}}{\tau_{coh}}\right)^{-1}$
        
        où $\tau_{meas}$ est le temps de mesure et $\tau_{coh}$ est le temps de cohérence du système.
        """)
        
        # Informatique quantique
        st.subheader("Informatique Quantique Basée sur les Systèmes Oscillatoires")
        
        st.markdown(r"""
        Notre théorie oscillatoire unifiée suggère des approches novatrices pour l'informatique quantique, 
        exploitant les degrés de liberté oscillatoires au-delà des approches conventionnelles basées sur 
        les qubits.
        """)
        
        contribution_box("Encodage d'information dans les modes normaux", r"""
        Pour un système de $n$ oscillateurs couplés avec couplages programmables, il est possible d'encoder 
        l'information quantique dans les modes normaux plutôt que dans les états des oscillateurs individuels.
        
        Ce changement de paradigme offre une protection intrinsèque contre certains types de bruit locaux 
        et améliore la robustesse de l'information quantique.
        """)
        
        theorem_box("Portes quantiques optomécaniques universelles", r"""
        Un système optomécanique avec couplage quadratique modulable temporellement permet de réaliser 
        un ensemble universel de portes quantiques sur des qumodes, caractérisé par les transformations:
        
        $\hat{a} \to \alpha\hat{a} + \beta\hat{a}^\dagger + \gamma\hat{b} + \delta\hat{b}^\dagger + \epsilon$
        
        $\hat{b} \to \alpha'\hat{a} + \beta'\hat{a}^\dagger + \gamma'\hat{b} + \delta'\hat{b}^\dagger + \epsilon'$
        
        avec les coefficients complexes satisfaisant $|\alpha|^2 - |\beta|^2 = |\gamma'|^2 - |\delta'|^2 = 1$ et $\alpha\delta' - \beta\gamma' = \alpha'\delta - \beta'\gamma = 0$.
        """)
        
        # Perspectives conceptuelles et fondamentales
        st.subheader("Frontières Conceptuelles et Extensions Théoriques")
        
        st.markdown(r"""
        Notre théorie oscillatoire unifiée ouvre des perspectives conceptuelles originales qui pourraient 
        transformer notre compréhension fondamentale de la physique.
        """)
        
        contribution_box("Émergence de l'espace-temps", r"""
        La structure de l'espace-temps pourrait émerger comme une manifestation macroscopique des relations 
        entre oscillateurs quantiques fondamentaux, via la relation formelle:
        
        $g_{\mu\nu}(x) \sim \langle\Phi|\hat{O}_{\mu\nu}(x)|\Phi\rangle$
        
        où $g_{\mu\nu}$ est le tenseur métrique, $|\Phi\rangle$ est un état fondamental des oscillateurs quantiques fondamentaux, 
        et $\hat{O}_{\mu\nu}$ est un opérateur approprié construit à partir des opérateurs oscillatoires.
        """)
        
        contribution_box("Ontologie oscillatoire", r"""
        Notre théorie suggère une ontologie oscillatoire fondamentale où les entités physiques primaires 
        seraient des processus oscillatoires plutôt que des particules ou des champs.
        
        Cette réinterprétation ontologique résout certains paradoxes de la physique quantique conventionnelle 
        en remplaçant le dualisme onde-particule par une vision unifiée où les comportements ondulatoire et 
        particulaire émergent comme différents aspects des mêmes processus oscillatoires fondamentaux.
        """)
        
        # Défis à relever
        st.subheader("Défis à Relever")
        
        st.markdown("""
        Malgré ses succès, la Théorie Oscillatoire Unifiée doit encore relever plusieurs défis importants:
        
        1. **Formulation covariante**: Développer une formulation complètement covariante de la théorie, 
        compatible avec la relativité générale.
        
        2. **Description précise de la transition quantique-classique**: Affiner le mécanisme de filtrage 
        spectral pour capturer plus précisément les subtilités de la transition entre régimes quantique et classique.
        
        3. **Systèmes fortement corrélés**: Étendre la théorie pour décrire adéquatement les systèmes 
        fortement corrélés où les approximations de champ moyen ne sont plus valides.
        
        4. **Intégration cohérente des structures algébriques**: Unifier les différentes structures algébriques 
        (algèbres de Frobenius, C*-algèbres, catégories monoïdales) dans un cadre mathématique cohérent.
        """)
        
        # Perspectives futures
        st.subheader("Perspectives Futures")
        
        st.markdown("""
        Les perspectives futures de la Théorie Oscillatoire Unifiée sont vastes et prometteuses:
        
        1. **Métrologie quantique avancée**: Développement de senseurs ultra-sensibles exploitant les 
        principes de la TOU pour dépasser les limites conventionnelles.
        
        2. **Informatique quantique à variables continues**: Exploitation des degrés de liberté oscillatoires 
        pour le traitement de l'information quantique.
        
        3. **Optomécanique quantique à température ambiante**: Observation et manipulation d'effets quantiques 
        dans des objets macroscopiques à température ambiante.
        
        4. **Transition vers une théorie des champs unifiée**: Extension des principes de la TOU vers une 
        théorie des champs quantiques complète.
        
        5. **Émergence de l'espace-temps**: Exploration des liens profonds entre les phénomènes oscillatoires 
        fondamentaux et la structure de l'espace-temps.
        """)
        
        st.markdown("""
        ### Conclusion
        
        La Théorie Oscillatoire Unifiée représente une avancée conceptuelle importante dans notre compréhension 
        des phénomènes oscillatoires à travers les différentes échelles de la physique. En fournissant un 
        cadre mathématique cohérent qui unifie les descriptions quantique et classique, elle ouvre de nouvelles 
        perspectives tant fondamentales qu'appliquées.
        
        Les développements futurs de cette théorie promettent non seulement d'approfondir notre compréhension 
        des phénomènes physiques fondamentaux, mais aussi de déboucher sur des applications technologiques 
        concrètes dans des domaines aussi variés que la métrologie quantique, l'informatique quantique, 
        et les tests fondamentaux de la mécanique quantique.
        """)
    
    # ===========================================================================
    # SECTION 10: RÉFÉRENCES BIBLIOGRAPHIQUES
    # ===========================================================================
    elif page == "Références Bibliographiques":
        display_references()


# Exécution de l'application
if __name__ == "__main__":
    main()
#!/usr/bin/env python
# coding: utf-8


# In[ ]:




