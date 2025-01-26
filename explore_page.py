import pickle
import streamlit as st
import pickle
import numpy as np
import datetime
import pandas as pd
from datetime import datetime

def load_model():
    with open('kmeans_centers.pkl', 'rb') as file:
        centers = pickle.load(file)
    return centers

def normalize_dataframe(df):
    # Extract time and date components before normalization
    df['Time'] = df['timestamp'].map(lambda t: t.time())
    df['Date'] = df['timestamp'].map(lambda t: t.date())

    # Select only numerical columns for normalization
    numeric_cols = df.select_dtypes(include=['number']).columns
    df_norm = (df[numeric_cols] - df[numeric_cols].mean()) / (df[numeric_cols].max() - df[numeric_cols].min())

    # Add back the time and date columns to the normalized dataframe
    df_norm['Time'] = df['Time']
    df_norm['Date'] = df['Date']
    
    return df_norm

import streamlit as st
import pandas as pd

