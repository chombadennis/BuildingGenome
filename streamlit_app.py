import streamlit as st
from reorder_n_viz import display_cluster_info

st.title('MACHINE LEARNING PROJECT ON THE BUIILDNG GENOME PROJECT')

st.header('SECTION A: Clustering of the Daily Energy Consumption load profiles for the Building Data genome Project')

st.write('Here, we use the clustering algorithm to be able to group buildings with similar energy consumption patterns and immportantly, identify days or groups which have similar total energy consumption patterns. Through clustering, we provide important information to develop solutions on specific energy saving strategies. We will be able to detect anomalies such as unsual spikes or drop in energy use, that could potetilly indicate issues. Additionallly, clustering will improve demand forecasting that we are later going to do through regression anlysis. Clustering can adapt to changes in energy patterns over time, ensuring continuous optimizations as building usage or occupancy patterns evolve.')

st.subheader('Objective of Using Clustering Algorithm:')
st.write('We aim to use clustering to enhance energy management, leading to greater efficiency, cost savings and more informed decision making.')

from predict_page import show_predict_page 

page = st.sidebar.selectbox("Explore and Predcit Clusters", ("Predict", "Explore"))

if page == "Predict":
    show_predict_page()
elif page == "Explore":
    if 'df_pivot_w_clusters' in st.session_state:
        display_cluster_info(st.session_state.df_pivot_w_clusters)
    else:
        st.write("Please run the prediction first to generate cluster information.")

show_predict_page()


