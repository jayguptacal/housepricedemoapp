import streamlit as st
import pandas as pd
import numpy as np

# Load data visualization packages
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import plotly.express as px

# load data
@st.cache
def load_data(data):
	df = pd.read_csv(data)
	return df

def run_eda_app():
	st.subheader("Running EDA Application")
	df = load_data("data/train.csv")
	df_encoded = load_data("data/df_scaled.csv")
	
	submenu = st.sidebar.selectbox("Submenu", ["Descriptive_EDA", "EDA_Plots"])
	if submenu == "Descriptive_EDA":
		with st.beta_expander("Original Data before EDA"):
			st.dataframe(df)

		with st.beta_expander("Data after Cleaning, Imputing, Feature Selection and Encoding"):
			st.dataframe(df_encoded)

		with st.beta_expander("Data Types after Pre-Processing Steps"):
			st.dataframe(df_encoded.dtypes)

		with st.beta_expander("Descriptive Data Summary before ML"):
			st.dataframe(df_encoded.describe().T)

		
		
	if submenu == "EDA_Plots":
		st.subheader("Plots")

		# For Lot Frontage Distribution Plot
		with st.beta_expander("Lot Frontage Distribution Plot"):
			fig = plt.figure()
			sns.distplot(df_encoded['LotFrontage'])
			st.pyplot(fig)
			
		with st.beta_expander("Overall Quality Distribution Plot"):
			fig = plt.figure()
			sns.distplot(df_encoded['OverallQual'])
			st.pyplot(fig)

		with st.beta_expander("Overall Basement Quality Distribution Plot"):
			fig = plt.figure()
			sns.distplot(df_encoded['BsmtQual'])
			st.pyplot(fig)

		with st.beta_expander("Overall Kitchen Quality Distribution Plot"):
			fig = plt.figure()
			sns.distplot(df_encoded['KitchenQual'])
			st.pyplot(fig)

		with st.beta_expander("First Floor Sq Ft Distribution Plot"):
			fig = plt.figure()
			sns.distplot(df_encoded['FirstFlrSF'])
			st.pyplot(fig)

		with st.beta_expander("Ground Floor Living Area Distribution Plot"):
			fig = plt.figure()
			sns.distplot(df_encoded['GrLivArea'])
			st.pyplot(fig)

		with st.beta_expander("Car Garage Capacity Distribution Plot"):
			fig = plt.figure()
			sns.distplot(df_encoded['GarageCars'])
			st.pyplot(fig)
		

		# Box Plot for House Sales Price Outlier Detection
		with st.beta_expander("Sale Price Outlier Detection with Box Plot"):
			fig = plt.figure()
			sns.boxplot(df['SalePrice'])
			st.pyplot(fig)

		# Correlation Plot
		with st.beta_expander("Heatmap of the Correlation Plot"):
			corr_matrix = df_encoded.corr()
			fig = plt.figure(figsize=(24, 20))
			sns.heatmap(corr_matrix, annot=True)
			st.pyplot(fig)

		# Correlation Matrix
		with st.beta_expander("Correlation Matrix"):
			p1 = px.imshow(corr_matrix)
			st.plotly_chart(p1)