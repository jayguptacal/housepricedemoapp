import streamlit as st
import numpy as np
import joblib
import os

attrib_info = """
#### Attribute Information:
- **SalePrice** - the property's sale price in dollars. This is the target variable that you're trying to predict.
- **MSSubClass**: The building class
- **MSZoning**: The general zoning classification
- **LotFrontage**: Linear feet of street connected to property - to be checked
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
- **FirstFlrSF**: First Floor square feet
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
['MSSubClass', 'MSZoning', 'Neighborhood', 'OverallQual', 'YearRemodAdd', 'RoofStyle', 'MasVnrType', 'ExterQual', 'BsmtQual', 'BsmtExposure', 'HeatingQC', 
'CentralAir', 'FirstFlrSF', 'GrLivArea', 'BsmtFullBath','KitchenQual', 'Fireplaces', 'FireplaceQu', 'GarageType','GarageFinish', 'GarageCars','PavedDrive', 'LotFrontage']

MSSubClass_map = {"Class20":0.00,"Class30":0.7958806,"Class40":0.66809527,"Class45":0.55646406,"Class50":0.48799196,
"Class60":0.18010331,"Class70":0.61577729,"Class75":1.00,"Class80":0.40700665, "Class85":0.92366593,
"Class90":0.64270611,"Class120":0.97598391,"Class160":0.36020662,"Class180":0.58710996, "Class190":0.30788865}
MSZoning_map = {"Zone1":0.00,"Zone2":0.25,"Zone3":0.50,"Zone4":0.75,"Zone5":1.0}
Neighborhood_map = {"CollgCr":0.86363636, "Veenker":0.36363636, "Crawfor":0.95454545, "NoRidge":0.45454545, "Mitchel":0.68181818, 
"Somerst":0.13636364, "NWAmes":0.22727273, "OldTown":0.54545455, "BrkSide":0.72727273, "Sawyer":0.40909091, 
"NridgHt":0.59090909, "NAmes":0.81818182, "SawyerW":0.18181818, "IDOTRR":0.00, "MeadowV":0.27272727, 
"Edwards":1.00, "Timber":0.09090909, "Gilbert":0.63636364,"StoneBr":0.04545454, "ClearCr":0.90909091, 
"Blmngtn":0.31818182, "BrDale":0.50, "SWISU":0.77272727}
OverallQual_map = {"Below_Avg":0.00,"Avg":0.30103,"Above_Avg":0.47712126,"Below_Good":0.60205999,"Good":0.69897,
"Above_Good":0.77815125,"Below_High":0.84509804,"High":0.90308999,"Above_High":0.95424251,"Excellent":1.00}
YearRemodAdd_map = {"1950":0.00,"1951":0.04918033,"1952":0.09836066,"1953":0.16393443,"1954":0.73770492,"1955":0.01639344,
"1956":0.21311475,"1957":0.47540984,"1958":0.13114754,"1959":0.93442623,"1960":0.26229508,
"1961":0.63934426, "1962":0.85245902, "1963":0.60655738, "1964":0.18032787, "1965":0.08196721,
"1966":0.78688525, "1967":0.49180328, "1968":0.32786885, "1969":0.90163934, "1970":0.19672131,
"1971":0.40983607, "1972":0.24590164, "1973":0.98360656, "1974":0.6557377 , "1975":0.1147541 ,
"1976":0.37704918, "1977":0.81967213, "1978":0.7704918 , "1979":0.8852459 , "1980":0.52459016,
"1981":0.67213115, "1982":0.70491803, "1983":0.96721311, "1984":0.54098361, "1985":0.34426229,
"1986":0.03278689, "1987":0.68852459, "1988":0.06557377, "1989":0.57377049, "1990":0.14754098,
"1991":0.55737705, "1992":0.59016393, "1993":0.36065574, "1994":0.42622951, "1995":0.2295082 ,
"1996":0.44262295, "1997":0.39344262, "1998":0.91803279, "1999":0.95081967, "2000":0.86885246,
"2001":0.31147541, "2002":0.50819672, "2003":0.83606557, "2004":0.29508197, "2005":0.45901639,
"2006":0.75409836, "2007":0.62295082, "2008":0.80327869, "2009":0.27868853, "2010":0.72131148,
"2011":1.00}      
RoofStyle_map = {"Flat":0.00,"Gable":0.50,"Hip":1.00}
MasVnrType_map = {"No_Masonry":0.00,"Cenent":0.33333333,"BrickFace":0.66666667,"Stone":1.00}
ExterQual_map = {"Avg":0.00,"Good":0.33333333,"High":0.66666667,"Excellent":1.00}
BsmtQual_map = {"Low_Grade":0.00,"Avg_Grade":0.25,"Good_Grade":0.50,"High_Grade":0.75,"Excellent_Grade":1.00}
BsmtExposure_map = {"No_Bsmt":0.00,"Min":0.25,"Good":0.50,"High":0.75,"Excellent":1.00}
HeatingQC_map = {"No_Heating":0.00,"Min":0.25,"Good":0.50,"High":0.75,"Excellent":1.00}
CentralAir_map = {"No_AC":0,"Yes":1}
#FirstFlrSF = Min-300 and Max - 5000 Sq Ft - FirstFlrSF_user = st.number_input("Provide 1st Floor Area", 300, 5000)
#GrLivArea = Min-300 and Max - 6000 Sq Ft - GrLivArea_user = st.number_input("Provide Ground Floor Living Area", 300, 6000)
BsmtFullBath_map = {"No_Bath":0.00,"1Bath":0.33333333,"1.5Baths":0.66666667,"2Baths":1.00}
KitchenQual_map = {"Avg":0.00,"Good":0.33333333,"High":0.66666667,"Excellent":1.00}
Fireplaces_map = {"No_Fireplace":0.00,"1FP":0.33333333,"2FP":0.66666667,"3FP":1.00}
FireplaceQu_map = {"No_Fireplace":0.00,"Avg_Grade":0.20,"Good_Grade":0.40,"High_Grade":0.60,"Top_Grade":0.80,"Excellent_Grade":1.00}
GarageType_map = {"No_Garage":0.00,"Avg_Grade":0.20,"Good_Grade":0.40,"High_Grade":0.60,"Top_Grade":0.80,"Excellent_Grade":1.00}
GarageFinish_map = {"No_Garage":0.00,"Unfinished":0.33333333,"Finished":0.66666667,"Remodeled":1.00}
GarageCars_map = {"No_Garage":0.00,"1Car":0.25,"2Cars":0.50,"3Cars":0.75,"4Cars":1.00}
PavedDrive_map = {"No_Paving":0.00,"Partly_Paved":0.50,"Fully_Paved":1.00}
#LotFrontage = Min-20 and Max - 320 Ft - LotFrontage_user = st.number_input("Provide Lot Front Width", 20, 400)


def get_value(val,my_dict):
       for key,value in my_dict.items():
              if val == key:
                     return value 

# Load ML Models
@st.cache
def load_model(model_file):
       loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
       return loaded_model

def run_ml_app():
       st.title("Machine Learning Modeling")
       # User Values to be Collected here
       with st.beta_expander("Click on the Plus Sign to Select the Options on Community Features:"):
              MSSubClass = st.selectbox("Select One applicable Class for Your Community:",["Class20","Class30","Class40","Class45","Class50",
              "Class60","Class70","Class75","Class80","Class85","Class90","Class120","Class160","Class180", "Class190"])
              MSZoning = st.selectbox("Select Your Housing Zone:",["Zone1","Zone2","Zone3","Zone4","Zone5"]) 
              Neighborhood = st.selectbox("Select Your Neighborhood:",["CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel", "Somerst",
              "NWAmes", "OldTown", "BrkSide", "Sawyer", "NridgHt", "NAmes","SawyerW", "IDOTRR", "MeadowV", "Edwards", "Timber", "Gilbert",
              "StoneBr", "ClearCr", "Blmngtn", "BrDale", "SWISU"]) 

       with st.beta_expander("Click on the Plus Sign to Provide Values for the House:"):
              #LotFrontage_user = st.number_input("Provide Lot Front Size in feet", 20.00, 320.00)
              #LotFrontage_log = np.log(LotFrontage_user)
              #LotFrontage = float(LotFrontage_log)

              GrLivArea_user = st.number_input("Provide Ground Floor Area in Sq Ft", 300.00, 6000.00)
              GrLivArea_log = np.log(GrLivArea_user)
              GrLivArea = float(GrLivArea_log)

              FirstFlrSF_user = st.number_input("Provide 1st Floor Area in Sq Ft", 300.00, 5000.00)
              FirstFlrSF_log = np.log(FirstFlrSF_user)
              FirstFlrSF = float(FirstFlrSF_log)
              

       with st.beta_expander("Click on the Plus Sign to Select House Feature Options:"):

              OverallQual = st.selectbox("Select the Overall Quality of the House:",["Below_Avg","Avg","Above_Avg","Below_Good","Good",
              "Above_Good","Below_High","High","Above_High","Excellent"])
              YearRemodAdd = st.selectbox("Select the Year when the Remodeling was done (range is from 1950 to 2011):",["1950","1951","1952","1953","1954","1955",
                            "1956","1957","1958","1959","1960","1961","1962","1963","1964","1965","1966","1967","1968","1969","1970",
                            "1971","1972", "1973", "1974", "1975", "1976", "1977", "1978", "1979", "1980","1981", "1982", "1983", "1984", "1985",
                            "1986", "1987", "1988", "1989", "1990", "1991", "1992", "1993", "1994", "1995","1996", "1997", "1998", "1999", "2000",
                            "2001", "2002", "2003", "2004", "2005","2006", "2007", "2008", "2009", "2010","2011"])
              
              
              BsmtQual = st.selectbox("Select Basement Quality", ["Low_Grade","Avg_Grade","Good_Grade","High_Grade","Excellent_Grade"])
              BsmtExposure = st.selectbox("Select Basement Exposure if applicable", ["No_Bsmt","Min","Good","High","Excellent"])
              BsmtFullBath = st.selectbox("Select Basement Bathrooms if applicable", ["No_Bath","1Bath","1.5Baths","2Baths"])
              HeatingQC = st.selectbox("Select House Heating Quality if applicable", ["No_Heating","Min","Good","High","Excellent"])
              CentralAir = st.selectbox("Select Central Air Conditioning if applicable", ["No_AC","Yes"])
              KitchenQual = st.selectbox("Select Kichen Quality", ["Avg","Good","High","Excellent"])
              Fireplaces = st.selectbox("Select Number of Fireplaces if applicable", ["No_Fireplace","1FP","2FP","3FP"])
              FireplaceQu = st.selectbox("Select Fireplaces Quality if applicable", ["No_Fireplace","Avg_Grade","Good_Grade","High_Grade","Top_Grade","Excellent_Grade"])


       with st.beta_expander("Click on the Plus Sign to Select House Exeterior Options:"):
              RoofStyle = st.selectbox("Select Your Roof Style", ["Gable","Flat","Hip"])
              MasVnrType = st.selectbox("Select Masonry Veneer Type if applicable", ["No_Masonry","Cenent","BrickFace","Stone"])
              ExterQual = st.selectbox("Select Quality of the House Exterior", ["Avg","Good","High","Excellent"])
              GarageType = st.selectbox("Select Garage Quality if applicable", ["No_Garage","Avg_Grade","Good_Grade","High_Grade","Top_Grade","Excellent_Grade"])
              GarageFinish = st.selectbox("Select Garage Finish if applicable", ["No_Garage","Unfinished","Finished","Remodeled"])
              GarageCars = st.selectbox("Select Garage Capacity if applicable", ["No_Garage","1Car","2Cars","3Cars","4Cars"])
              PavedDrive = st.selectbox("Select Type of Paving if applicable", ["No_Paving","Partly_Paved","Fully_Paved"])

       with st.beta_expander("Your Selected Options"):
              result = {'MSSubClass':MSSubClass, 'MSZoning':MSZoning, 'Neighborhood':Neighborhood, 'GrLivArea':GrLivArea,
              'FirstFlrSF':FirstFlrSF, 'OverallQual':OverallQual, 'YearRemodAdd':YearRemodAdd, 'BsmtQual':BsmtQual, 'BsmtExposure':BsmtExposure,'BsmtFullBath':BsmtFullBath,
              'HeatingQC':HeatingQC, 'CentralAir':CentralAir, 'KitchenQual':KitchenQual, 'Fireplaces':Fireplaces, 'FireplaceQu':FireplaceQu,'RoofStyle':RoofStyle, 
              'MasVnrType':MasVnrType, 'ExterQual':ExterQual, 'GarageType':GarageType, 'GarageFinish':GarageFinish, 'GarageCars':GarageCars, 'PavedDrive':PavedDrive}
               
               
              
              st.write(result)
              encoded_result = []
              for i in result.values():
                     if type(i) == float:
                            encoded_result.append(i)

                     elif type(i) == int:
                            encoded_result.append(i)

                     elif i in ["Class20","Class30","Class40","Class45","Class50",
                               "Class60","Class70","Class75","Class80","Class85","Class90","Class120","Class160","Class180", "Class190"]:
                            res = get_value(i,MSSubClass_map)
                            encoded_result.append(res)

                     elif i in ["Zone1","Zone2","Zone3","Zone4","Zone5"]:
                            res = get_value(i,MSZoning_map)
                            encoded_result.append(res)

                     elif i in ["CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel", "Somerst",
                                   "NWAmes", "OldTown", "BrkSide", "Sawyer", "NridgHt", "NAmes","SawyerW", "IDOTRR", "MeadowV", "Edwards", "Timber", "Gilbert",
                                   "StoneBr", "ClearCr", "Blmngtn", "BrDale", "SWISU"]:
                            res = get_value(i,Neighborhood_map)
                            encoded_result.append(res)

                     elif i in ["Below_Avg","Avg","Above_Avg","Below_Good","Good",
                                   "Above_Good","Below_High","High","Above_High","Excellent"]:
                            res = get_value(i,OverallQual_map)
                            encoded_result.append(res)

                     elif i in ["1950","1951","1952","1953","1954","1955",
                            "1956","1957","1958","1959","1960","1961","1962","1963","1964","1965","1966","1967","1968","1969","1970",
                            "1971","1972", "1973", "1974", "1975", "1976", "1977", "1978", "1979", "1980","1981", "1982", "1983", "1984", "1985",
                            "1986", "1987", "1988", "1989", "1990", "1991", "1992", "1993", "1994", "1995","1996", "1997", "1998", "1999", "2000",
                            "2001", "2002", "2003", "2004", "2005","2006", "2007", "2008", "2009", "2010","2011"]:
                            res = get_value(i,YearRemodAdd_map)
                            encoded_result.append(res)
                     
                     elif i in ["Low_Grade","Avg_Grade","Good_Grade","High_Grade","Excellent_Grade"]:
                            res = get_value(i,BsmtQual_map)
                            encoded_result.append(res)

                     elif i in ["No_Bsmt","Min","Good","High","Excellent"]:
                            res = get_value(i,BsmtExposure_map)
                            encoded_result.append(res)

                     elif i in ["No_Bath","1.0Bath","1.5Baths","2Baths"]:
                            res = get_value(i,BsmtFullBath_map)
                            encoded_result.append(res)

                     elif i in ["No_Heating","Min","Good","High","Excellent"]:
                            res = get_value(i,HeatingQC_map)
                            encoded_result.append(res)

                     elif i in ["No_AC","Yes"]:
                            res = get_value(i,CentralAir_map)
                            encoded_result.append(res)

                     elif i in ["Avg","Good","High","Excellent"]:
                            res = get_value(i,KitchenQual_map)
                            encoded_result.append(res)

                     elif i in ["No_Fireplace","1FP","2FP","3FP"]:
                            res = get_value(i,Fireplaces_map)
                            encoded_result.append(res)

                     elif i in ["No_Fireplace","Avg_Grade","Good_Grade","High_Grade","Top_Grade","Excellent_Grade"]:
                            res = get_value(i,FireplaceQu_map)
                            encoded_result.append(res)

                     elif i in ["Gable","Flat","Hip"]:
                            res = get_value(i,RoofStyle_map)
                            encoded_result.append(res)

                     elif i in ["No_Masonry","Cenent","BrickFace","Stone"]:
                            res = get_value(i,MasVnrType_map)
                            encoded_result.append(res)

                     elif i in ["Avg","Good","High","Excellent"]:
                            res = get_value(i,ExterQual_map)
                            encoded_result.append(res)

                     elif i in ["No_Garage","Avg_Grade","Good_Grade","High_Grade","Top_Grade","Excellent_Grade"]:
                            res = get_value(i,GarageType_map)
                            encoded_result.append(res)

                     elif i in ["No_Garage","Unfinished","Finished","Remodeled"]:
                            res = get_value(i,GarageFinish_map)
                            encoded_result.append(res)

                     elif i in ["No_Garage","1Car","2Cars","3Cars","4Cars"]:
                            res = get_value(i,GarageCars_map)
                            encoded_result.append(res)

                     elif i in ["No_Paving","Partly_Paved","Fully_Paved"]:
                            res = get_value(i,PavedDrive_map)
                            encoded_result.append(res)

                     else:
                            st.write(encoded_result)

                     

       with st.beta_expander("Predicted House Prices"):
              single_sample = np.array(encoded_result).reshape(1, -1)
              st.write(single_sample)
              # Loading model
              model = load_model("models/lasso_regression.pkl")
              prediction = model.predict(single_sample)
              pred_val = np.exp(prediction)
              
              st.subheader("Predicted House Price in USD is shown below:")
              st.success(pred_val)

              

