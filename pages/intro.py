import streamlit as st
import pandas as pd
import numpy as np


def show_intro():
    st.title("Hi there!")
    col1, spacing_col, col2 = st.columns([5, 1, 5]) 


    with col1:
        st.write("")
        st.write('''
                Welcome to miniSoul. This is a demo of my developed in-house app SOUL - Simulation and Optimization Unified Lab.

                Media spend planning is an art of tuning and balancing. 
                Equipped with the latest media mix model (MMM) results and greedy optimization algorithm, 
                this is a space we can improvise and rehearse, over and over again until we find the perfect harmony across media channels. 
                Take a seat and let's get started! 
                ''')

        
    # The middle column creates the white space
    with spacing_col:
        st.write("")

    with col2:
        st.write("")
        st.image("static_files/images/soul-joe-mr-mittens_piano2.png", width= 500)