import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.title("Clustering Algorithm Experiment (K-Means)")

st.write("Upload dataset to perform clustering")

file = st.file_uploader("Upload CSV file", type=["csv"])

if file is not None:

    data = pd.read_csv(file)

    st.subheader("Dataset Preview")
    st.write(data)

    X = data[['X','Y']]

    k = st.slider("Select Number of Clusters", 2, 5, 3)

    model = KMeans(n_clusters=k)

    model.fit(X)

    labels = model.predict(X)

    st.subheader("Cluster Labels")
    st.write(labels)

    plt.scatter(X['X'], X['Y'], c=labels)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Cluster Visualization")

    st.pyplot(plt)
