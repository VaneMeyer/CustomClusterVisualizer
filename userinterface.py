import streamlit as st
import pandas as pd
import os
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, k_means
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

# Title and Welcome text
st.title('Create a Profile based on K-Means Cluster Analysis')
st.write('Choose a .csv file you want to analyse:')
# Data upload
dataset = st.file_uploader(label='')

# After data upload
if dataset:
    data = pd.read_csv(dataset)
    features = np.array(data.columns)
    selected_features_array = []
    input_vector = []

    st.sidebar.header('Data Pre-processing')

    if st.sidebar.checkbox('Show original data'):
        st.subheader('Original Data')
        st.write(data)

    handle_missing_values = st.sidebar.selectbox(label='Handle Missing Values',
                                                 options=['Ignore', 'Fill with mean',
                                                          ])
    if handle_missing_values == 'Fill with mean':
        data = data.fillna(data.mean())

        if st.sidebar.checkbox('Replace 0 with mean'):
            data = data.replace(0, data.mean())

    selected_features = st.sidebar.multiselect(label='Select Features', options=features)
    selected_data = data[selected_features]

    if selected_features:
        data = selected_data
        features = selected_features
        if st.sidebar.checkbox('Show subset data'):
            st.subheader('Selected Data')
            st.write(data)

    if st.sidebar.checkbox('Scale data'):
        try:
            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(data)
            data_scaled = pd.DataFrame(data_scaled, columns=features)

            if st.sidebar.checkbox('Show scaled data'):
                st.subheader('Scaled Data')
                st.write(data_scaled)

            st.sidebar.header('Your Data')
            for f in features:
                input_vector.append(st.sidebar.number_input(label=f, key=f))

            k = st.sidebar.number_input(label='Choose k', min_value=2, max_value=9)

            if st.sidebar.button('Analyse'):
                kmeans = KMeans(n_clusters=k, random_state=100)
                kmeans = kmeans.fit(data_scaled)
                centers = kmeans.cluster_centers_

                centers_descaled = scaler.inverse_transform(centers)

                # Distances between input-vector and centers
                dist = []
                for center in centers_descaled:
                    dist.append(np.linalg.norm(input_vector - center))
                min_dist = min(dist)

                assigned_Cluster = dist.index(min_dist)
                st.subheader('Your Cluster')
                # Clustername
                clustername = 'Cluster ' + str(assigned_Cluster)
                st.write('You belong to ', clustername)

                # Radarchart
                if len(features) > 2:
                    fig_cluster = go.Figure()

                    fig_cluster.add_trace(go.Scatterpolar(
                        r=centers[assigned_Cluster],
                        theta=features,
                        fill='toself',
                        marker=dict(color='darkblue'),
                        name=clustername
                    ))

                    # Scale input vector

                    input_vector_scaled = []
                    for i in range(0, len(features)):
                        diff = max(data[features[i]]) - min(data[features[i]])
                        input_vector_scaled.append((input_vector[i] - min(data[features[i]])) / diff)

                    fig_cluster.add_trace(go.Scatterpolar(
                        r=input_vector_scaled,
                        theta=features,
                        fill='toself',
                        name='Your Data'
                    ))

                    fig_cluster.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,

                            )),
                        showlegend=True

                    )

                    st.write(fig_cluster)
                else:
                    fig_2d = go.Figure()

                    fig_2d.add_trace(go.Bar(
                        x=features,
                        y=centers[assigned_Cluster],
                        name=clustername

                    ))
                    # Scale input vector

                    input_vector_scaled = []
                    for i in range(0, len(features)):
                        diff = max(data[features[i]]) - min(data[features[i]])
                        input_vector_scaled.append((input_vector[i] - min(data[features[i]])) / diff)

                    fig_2d.add_trace(go.Bar(
                        x=features,
                        y=input_vector_scaled,
                        name='Your Data'

                    ))

                    st.write(fig_2d)

                st.subheader('Your Features')
                col1, col2 = st.columns(2, gap="large")
                for i in range(0, len(features)):
                    if i % 2 == 0:
                        with col1:
                            gauge = go.Figure(go.Indicator(
                                mode="gauge+number+delta",
                                value=input_vector[i],
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': features[i], 'font': {'size': 24}},
                                delta={'reference': centers_descaled[assigned_Cluster][i],
                                       'increasing': {'color': "RebeccaPurple"}},
                                gauge={
                                    'axis': {'range': [min(data[features[i]]), max(data[features[i]])],
                                             'tickwidth': 1, 'tickcolor': "darkblue"},
                                    'bar': {'color': "salmon"},
                                    'bgcolor': "white",
                                    'borderwidth': 2,
                                    'bordercolor': "gray",
                                    'threshold': {
                                        'line': {'color': "darkblue", 'width': 4},
                                        'thickness': 0.75,
                                        'value': centers_descaled[assigned_Cluster][i]}}))

                            gauge.update_layout(paper_bgcolor="white", font={'color': "black", 'family': "Arial"},
                                                height=250, width=300)

                            st.write(gauge)

                    if i % 2 == 1:
                        with col2:
                            gauge = go.Figure(go.Indicator(
                                mode="gauge+number+delta",
                                value=input_vector[i],
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': features[i], 'font': {'size': 24}},
                                delta={'reference': centers_descaled[assigned_Cluster][i],
                                       'increasing': {'color': "RebeccaPurple"}},
                                gauge={
                                    'axis': {'range': [min(data[features[i]]), max(data[features[i]])],
                                             'tickwidth': 1, 'tickcolor': "darkblue"},
                                    'bar': {'color': "salmon"},
                                    'bgcolor': "white",
                                    'borderwidth': 2,
                                    'bordercolor': "gray",
                                    'threshold': {
                                        'line': {'color': "darkblue", 'width': 4},
                                        'thickness': 0.75,
                                        'value': centers_descaled[assigned_Cluster][i]}}))

                            gauge.update_layout(paper_bgcolor="white", font={'color': "black", 'family': "Arial"},
                                                height=250, width=300)

                            st.write(gauge)

        except:
            st.write(
                'Some preprocessing has to be done to scale the data. Make sure that you have only numerical data and '
                'no missing values.')

    else:
        st.sidebar.header('Your Data')
        for f in features:
            input_vector.append(st.sidebar.number_input(label=f, key=f))

        k = st.sidebar.number_input(label='Choose k', min_value=2, max_value=9)

        if st.sidebar.button('Analyse'):
            try:
                kmeans = KMeans(n_clusters=k, random_state=100)
                kmeans = kmeans.fit(data)
                centers = kmeans.cluster_centers_

                st.write('Data is not scaled.')

                # Distances between input-vector and centers (descaled)
                dist = []
                for center in centers:
                    dist.append(np.linalg.norm(input_vector - center))
                min_dist = min(dist)

                assigned_Cluster = dist.index(min_dist)
                st.subheader('Your Cluster')
                # Clustername
                clustername = 'Cluster ' + str(assigned_Cluster)
                st.write('You belong to ', clustername)

                # Radarchart
                if len(features) > 2:
                    fig_cluster = go.Figure()

                    fig_cluster.add_trace(go.Scatterpolar(
                        r=centers[assigned_Cluster],
                        theta=features,
                        fill='toself',
                        marker=dict(color='darkblue'),
                        name=clustername

                    ))

                    fig_cluster.add_trace(go.Scatterpolar(
                        r=input_vector,
                        theta=features,
                        fill='toself',
                        name='Your Data'
                    ))

                    fig_cluster.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,

                            )),
                        showlegend=True

                    )

                    st.write(fig_cluster)
                else:
                    fig_2d = go.Figure()

                    fig_2d.add_trace(go.Bar(
                        x=features,
                        y=centers[assigned_Cluster],
                        name=clustername

                    ))

                    fig_2d.add_trace(go.Bar(
                        x=features,
                        y=input_vector,
                        name='Your Data'

                    ))

                    st.write(fig_2d)

                #### Gauge plots ####
                st.subheader('Your Features')
                col1, col2 = st.columns(2, gap="large")
                for i in range(0, len(features)):
                    if i % 2 == 0:
                        with col1:
                            gauge = go.Figure(go.Indicator(
                                mode="gauge+number+delta",
                                value=input_vector[i],
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': features[i], 'font': {'size': 24}},
                                delta={'reference': centers[assigned_Cluster][i],
                                       'increasing': {'color': "RebeccaPurple"}},
                                gauge={
                                    'axis': {'range': [min(data[features[i]]), max(data[features[i]])], 'tickwidth': 1,
                                             'tickcolor': "darkblue"},
                                    'bar': {'color': "salmon"},
                                    'bgcolor': "white",
                                    'borderwidth': 2,
                                    'bordercolor': "gray",
                                    'threshold': {
                                        'line': {'color': "darkblue", 'width': 4},
                                        'thickness': 0.75,
                                        'value': centers[assigned_Cluster][i]}}))

                            gauge.update_layout(paper_bgcolor="white", font={'color': "black", 'family': "Arial"},
                                                height=250, width=300)

                            st.write(gauge)
                    if i % 2 == 1:
                        with col2:
                            gauge = go.Figure(go.Indicator(
                                mode="gauge+number+delta",
                                value=input_vector[i],
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': features[i], 'font': {'size': 24}},
                                delta={'reference': centers[assigned_Cluster][i],
                                       'increasing': {'color': "RebeccaPurple"}},
                                gauge={
                                    'axis': {'range': [min(data[features[i]]), max(data[features[i]])], 'tickwidth': 1,
                                             'tickcolor': "darkblue"},
                                    'bar': {'color': "salmon"},
                                    'bgcolor': "white",
                                    'borderwidth': 2,
                                    'bordercolor': "gray",
                                    'threshold': {
                                        'line': {'color': "darkblue", 'width': 4},
                                        'thickness': 0.75,
                                        'value': centers[assigned_Cluster][i]}}))

                            gauge.update_layout(paper_bgcolor="white", font={'color': "black", 'family': "Arial"},
                                                height=250, width=300)

                            st.write(gauge)

            except:
                st.write(
                    'Some preprocessing has to be done to the data. Make sure that you have only numerical data and '
                    'no missing values.')

else:
    st.write('Welcome!')
    st.write('To start analyzing your data, drag and drop any .csv file into the box above. In the next steps, '
             'you can make further adjustments to the data, such as replacing missing values with mean values, '
             'selecting desired features, and scaling the data if the data has not already been pre-processed '
             'accordingly. Individual values can then be entered for the features. The number of groups for the '
             'k-means '
             'clustering can also be determined by the user. The clusters are calculated and an assignment to one of '
             'the '
             'clusters is made based on the entered feature values. In order to see to what extent the entered values '
             'differ from the mean values of the cluster, the differences are shown in detail for each feature.')
