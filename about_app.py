import streamlit as st 
import streamlit.components.v1 as stc
from PIL import Image



def run_about_app():
	st.write("This is an educational project to demonstrate the development of an application following the below Data Science Methodology:")
	img = Image.open("images/MLmethodology.jpg")
	st.image(img, use_column_width=True)



