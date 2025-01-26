import streamlit as st
from predict_page import show_predict_page
from explore_page import show_explore_page

st.title('MACHINE LEARNING PROJECT OF THE BUIILDNG GENOME PROJECT')

st.header('SECTION A: Clustering of the Daily Energy Consumption load profiles for the Building Data genome Project')

st.write('Here, we use the clustering algorithm to be able to group buildings with similar energy consumption patterns and immportantly, identify days or groups which have similar total energy consumption patterns. Through clustering, we provide important information to develop solutions on specific energy saving strategies. We will be able to detect anomalies such as unsual spikes or drop in energy use, that could potetilly indicate issues. Additionallly, clustering will improve demand forecasting that we are later going to do through regression anlysis. Clustering can adapt to changes in energy patterns over time, ensuring continuous optimizations as building usage or occupancy patterns evolve.')

st.subheader('Objective of Using Clustering Algorithm:')
st.write('We aim to use clustering to enhance energy management, leading to greater efficiency, cost savings and more informed decision making.')


# Sidebar for navigation
page = st.sidebar.selectbox("Explore and Predict Clusters", ("Predict", "Explore"))

if page == "Predict":
    show_predict_page()
elif page == "Explore":
    st.write("Click the button below to explore cluster information.")
    if st.button('Show Cluster Information'):
        show_explore_page()