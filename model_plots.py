import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from shiny import Inputs, Outputs, Session, module, render, ui, module

from shiny import ui

def plot_indicator(indicator, description):
    return (ui.h1("{0}".format(indicator)),
            ui.p(description))

def plot_percent_indicator(indicator, description):
    return (ui.h1("""%.1f%%""" % (indicator * 100)),
            ui.p(description))

def plot_float_indicator(indicator, description):
    return (ui.h1("""%.1f""" % (indicator)),
            ui.p(description))

def plot_idea_rank(df):
    #fig = go.Bar(x=df.vote, y=df.idea)
    fig = px.histogram(df, y='idea', x='vote')
    fig.update_layout(bargap=0.1)
    fig.layout.height = 500
    return fig

def trim_labels(label, max_length=50):
    """Trim label to max_length characters."""
    if len(label) > max_length:
        return label[:max_length] + '...'
    return label