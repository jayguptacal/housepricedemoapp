import streamlit as st 
import streamlit.components.v1 as stc
# Import other apps here
from eda_app import run_eda_app
from ml_app import run_ml_app
from about_app import run_about_app
# Display image
from PIL import Image





html_temp = """
		<div style="background-color: #248c4d;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">House Sales Price Demonstration App </h1>
		<h4 style="color:white;text-align:center;">Find Your House Prices! </h4>
		</div>
		"""

desc_temp = """
			### Initial House Sales Price Prediction App
				The dataset is compiled by Dean De Cock from "Journal of Statistics Education" 
				for the purpose of machine learning projects.

			#### Datasource:  
				 This dataset has been provided by Dean De Cock at Kaggle for the data science 
				 community to do machine learning projects and participate in their competitions.

				- http://jse.amstat.org/v19n3/decock.pdf 
				- https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

			#### App Content 
				- EDA Section: Exploratory Data Analysis of Data
				- ML Section:  ML Predictor App

			#### Attribute Information:
				- **SalePrice** - This is the target variable that you're trying to predict.
				- **MSSubClass**: The building class
				- **MSZoning**: The general zoning classification
				- **LotFrontage**: Linear feet of street connected to property
				- **LotArea**: Lot size in square feet
				- **Neighborhood**: Physical locations within Ames city limits
				- **OverallQual**: Overall material and finish quality
				- **OverallCond**: Overall condition rating
				- **YearBuilt**: Original construction date
				- **YearRemodAdd**: Remodel date
				- **RoofStyle**: Type of roof
				- **MasVnrType**: Masonry veneer type
				- **ExterQual**: Exterior material quality
				- **BsmtQual**: Height of the basement
				- **BsmtExposure**: Walkout or garden level basement walls
				- **HeatingQC**: Heating quality and condition
				- **CentralAir**: Central air conditioning
				- **1stFlrSF**: First Floor square feet
				- **GrLivArea**: Above grade (ground) living area square feet
				- **BsmtFullBath**: Basement full bathrooms
				- **BsmtHalfBath**: Basement half bathrooms
				- **KitchenQual**: Kitchen quality
				- **Fireplaces**: Number of fireplaces
				- **FireplaceQu**: Fireplace quality
				- **GarageType**: Garage location
				- **GarageFinish**: Interior finish of the garage
				- **GarageCars**: Size of garage in car capacity
				- **GarageQual**: Garage quality
				- **PavedDrive**: Paved driveway.

			"""

def main():
	stc.html(html_temp)
	img = Image.open("images/houses.jpg")
	st.image(img, use_column_width=True)
	# Sidemenu
	menu = ["Home", "EDA", "ML", "About"]
	choice = st.sidebar.selectbox("Menu", menu)

	if choice == "Home":
		st.subheader("Home")
		st.markdown(desc_temp, unsafe_allow_html=True)

	elif choice == "EDA":
		run_eda_app()

	elif choice == "ML":
		run_ml_app()

	else:
		st.subheader("About")
		run_about_app()


if __name__ == '__main__':
	main()