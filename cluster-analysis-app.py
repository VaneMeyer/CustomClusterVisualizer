import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift, AffinityPropagation, Birch
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# Error Handling
st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.title("Cluster Analysis App")
    st.write("Upload your CSV file for cluster analysis.")

    # File upload
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.write(data.head())

        #  Feature selection
        st.sidebar.title("Feature Selection")
        #all_features = st.sidebar.checkbox("Select All Features")
        #if all_features:
         #   features = st.sidebar.multiselect("Select Features", data.columns, default=data.columns)
        #else:
            #features = st.sidebar.multiselect("Select Features", data.columns)
        features = st.sidebar.multiselect("Select Features", data.columns)

        if features:
            data = data[features]
            st.write("Selected Features Data Preview:")
            st.write(data.head())

            #  Preprocessing Pipeline Selection
            st.sidebar.title("Preprocessing Pipeline")
            preprocessing_option = st.sidebar.selectbox(
                "Choose a preprocessing pipeline",
                ["Manual", "Autoencoder (Deep Learning)"]
            )

            if preprocessing_option == "Manual":
                # Manual Preprocessing Options
                st.sidebar.subheader("Manual Preprocessing Options")
                
                # Determine feature types
                numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
                non_numeric_features = data.select_dtypes(exclude=[np.number]).columns.tolist()
                
                if non_numeric_features:
                    st.sidebar.error(f"The following features are non-numeric and cannot be used with selected preprocessing methods: {non_numeric_features}")

                # Handling missing values
                missing_values_method = st.sidebar.selectbox(
                    "Handling Missing Values",
                    ["None", "Mean Imputation", "Median Imputation", "Most Frequent Imputation"]
                )
                if missing_values_method != "None":
                    imputer_strategy = {
                        "Mean Imputation": "mean",
                        "Median Imputation": "median",
                        "Most Frequent Imputation": "most_frequent"
                    }
                    imputer = SimpleImputer(strategy=imputer_strategy[missing_values_method])
                    data[numeric_features] = pd.DataFrame(imputer.fit_transform(data[numeric_features]), columns=numeric_features)

                    st.write("Data Preview after imputing missing values:")
                    st.write(data.head())

                # Outlier Detection 
                #outlier_detection_method = st.sidebar.selectbox(
                 #   "Outlier Detection",
                   # ["None", "Z-Score", "IQR"]
                #)
                

                # Scaling
                scaling_method = st.sidebar.selectbox(
                    "Scaling",
                    ["None", "Standard Scaling", "Min-Max Scaling"]
                )
                if scaling_method != "None":
                    scaler = StandardScaler() if scaling_method == "Standard Scaling" else MinMaxScaler()
                    data[numeric_features] = pd.DataFrame(scaler.fit_transform(data[numeric_features]), columns=numeric_features)
                    st.write("Data Preview after scaling:")
                    st.write(data.head())

                # Dimensionality Reduction
                dim_reduction_method = st.sidebar.selectbox(
                    "Dimensionality Reduction",
                    ["None", "PCA"]
                )
                if dim_reduction_method == "PCA":
                    n_components = st.sidebar.slider("Number of PCA Components", 1, len(numeric_features), 2)
                    pca = PCA(n_components=n_components)
                    data = pd.DataFrame(pca.fit_transform(data[numeric_features]), columns=[f"PC{i+1}" for i in range(n_components)])
                    st.write("Data Preview after dimensionality reduction:")
                    st.write(data.head())
            
            elif preprocessing_option == "Autoencoder (Deep Learning)":
                #####################################
                
                numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
                categorical_features = data.select_dtypes(include=['object']).columns

                st.write(f"Numerical features: {numerical_features.tolist()}")
                st.write(f"Categorical features: {categorical_features.tolist()}")

                
                numerical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ])

                
                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ])

                
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numerical_transformer, numerical_features),
                        ('cat', categorical_transformer, categorical_features)
                    ])

               
                df_processed = preprocessor.fit_transform(data)
                data = df_processed
                #####################################
                st.sidebar.subheader("Autoencoder Preprocessing Options")
                # Autoencoder parameters
                encoding_dim = st.sidebar.slider("Encoding Dimension", 1, len(features)//2, 3)
                #encoding_dim = st.sidebar.slider("Encoding Dimension", 1, 128, 3)
                epochs = st.sidebar.number_input("Epochs", 1, 100, 10)
                batch_size = st.sidebar.number_input("Batch Size", 1, 128, 32)
                
                # Autoencoder model
                input_dim = data.shape[1]
                input_layer = Input(shape=(input_dim,))
                encoded = Dense(encoding_dim, activation='relu')(input_layer)
                decoded = Dense(input_dim, activation='sigmoid')(encoded)
                autoencoder = Model(input_layer, decoded)
                encoder = Model(input_layer, encoded)
                autoencoder.compile(optimizer=Adam(), loss='mse')
                autoencoder.fit(data, data, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0)
                data = pd.DataFrame(encoder.predict(data), columns=[f"Enc{i+1}" for i in range(encoding_dim)])
                st.write("Data Preview after autoencoding:")
                st.write(data.head())
            
            # User Input Section
            st.sidebar.title("Enter Your Data")
            user_input = {}
            for feature in features:
                user_input[feature] = st.sidebar.number_input(f"Enter {feature}", value=0.0)
            user_input_df = pd.DataFrame([user_input])

            # Preprocess user input
            if preprocessing_option == "Manual":
                if scaling_method != "None":
                    user_input_df = pd.DataFrame(scaler.transform(user_input_df), columns=user_input_df.columns)
                if dim_reduction_method == "PCA":
                    user_input_df = pd.DataFrame(pca.transform(user_input_df), columns=[f"PC{i+1}" for i in range(n_components)])
            elif preprocessing_option == "Autoencoder (Deep Learning)":
                user_input_df = pd.DataFrame(encoder.predict(user_input_df), columns=[f"Enc{i+1}" for i in range(encoding_dim)])
            st.write("Preprocessed user input data:")
            st.write(user_input_df.head())

            # Cluster Algorithm Selection
            st.sidebar.title("Clustering Algorithm")
            clustering_algorithm = st.sidebar.selectbox(
                "Choose a clustering algorithm",
                ["K-Means", "Hierarchical Clustering", "DBSCAN", "Mean-Shift", "Affinity Propagation", "BIRCH"]
            )

            # Algorithm Parameters
            cluster_params = {}
            if clustering_algorithm == "K-Means":
                cluster_params['n_clusters'] = st.sidebar.number_input("Number of Clusters (k)", 2, 10, 3)
            elif clustering_algorithm == "Hierarchical Clustering":
                cluster_params['n_clusters'] = st.sidebar.number_input("Number of Clusters", 2, 10, 3)
                cluster_params['linkage'] = st.sidebar.selectbox("Linkage Method", ["ward", "complete", "average", "single"])
            elif clustering_algorithm == "DBSCAN":
                cluster_params['eps'] = st.sidebar.slider("Epsilon", 0.1, 10.0, 0.5)
                cluster_params['min_samples'] = st.sidebar.slider("Minimum Samples", 1, 20, 5)
            elif clustering_algorithm == "Mean-Shift":
                cluster_params['bandwidth'] = st.sidebar.slider("Bandwidth", 0.1, 10.0, 1.0)
            elif clustering_algorithm == "Affinity Propagation":
                cluster_params['damping'] = st.sidebar.slider("Damping", 0.5, 1.0, 0.9)
            elif clustering_algorithm == "BIRCH":
                cluster_params['n_clusters'] = st.sidebar.number_input("Number of Clusters", 2, 10, 3)
                cluster_params['threshold'] = st.sidebar.slider("Threshold", 0.1, 1.0, 0.5)

            if st.sidebar.button("Start Analysis"):
                # Determine optimal number of clusters 
                if 'n_clusters' in cluster_params and clustering_algorithm in ["K-Means", "Hierarchical Clustering", "BIRCH"]:
                    silhouette_avg = []
                    for k in range(2, 11):
                        model = KMeans(n_clusters=k) if clustering_algorithm == "K-Means" else AgglomerativeClustering(n_clusters=k)
                        cluster_labels = model.fit_predict(data)
                        silhouette_avg.append(silhouette_score(data, cluster_labels))
                    optimal_clusters = silhouette_avg.index(max(silhouette_avg)) + 2
                    st.write(f"Optimal number of clusters: {optimal_clusters}")
                    
                
                # Ensure parameters are set correctly
                if all(v is not None for v in cluster_params.values()):
                    # Run clustering algorithm
                    if clustering_algorithm == "K-Means":
                        model = KMeans(**cluster_params)
                    elif clustering_algorithm == "Hierarchical Clustering":
                        model = AgglomerativeClustering(**cluster_params)
                    elif clustering_algorithm == "DBSCAN":
                        model = DBSCAN(**cluster_params)
                    elif clustering_algorithm == "Mean-Shift":
                        model = MeanShift(**cluster_params)
                    elif clustering_algorithm == "Affinity Propagation":
                        model = AffinityPropagation(**cluster_params)
                    elif clustering_algorithm == "BIRCH":
                        model = Birch(**cluster_params)

                    # Fit and predict clusters
                    cluster_labels = model.fit_predict(data)
                    data['Cluster'] = cluster_labels
                    st.write("Clustered Data:")
                    st.write(data.head())

                    #######################################################
                    
                    cluster_centers = None
                    if hasattr(model, 'cluster_centers_'):
                        cluster_centers = model.cluster_centers_

                    if cluster_centers is not None:
                        st.write("Cluster Centers:")
                        st.write(cluster_centers)

                    #######################################################

                    #  Visualizations with PCA and t-SNE
                    st.title("Cluster Visualizations")
                    pca = PCA(n_components=2)
                    pca_result = pca.fit_transform(data.drop('Cluster', axis=1))
                    data['PCA1'] = pca_result[:, 0]
                    data['PCA2'] = pca_result[:, 1]
                    fig = px.scatter(data, x='PCA1', y='PCA2', color='Cluster', title="PCA Visualization")
                    st.plotly_chart(fig)

                    tsne_perplexity = min(30, len(data)-1)  
                    tsne = TSNE(n_components=2, perplexity=tsne_perplexity)
                    tsne_result = tsne.fit_transform(data.drop(['Cluster', 'PCA1', 'PCA2'], axis=1))
                    data['TSNE1'] = tsne_result[:, 0]
                    data['TSNE2'] = tsne_result[:, 1]
                    fig = px.scatter(data, x='TSNE1', y='TSNE2', color='Cluster', title="t-SNE Visualization")
                    st.plotly_chart(fig)

                    # F13: Cluster Evaluation Scores
                    st.title("Cluster Evaluation")
                    silhouette_avg = silhouette_score(data.drop(['Cluster', 'PCA1', 'PCA2', 'TSNE1', 'TSNE2'], axis=1), cluster_labels)
                    calinski_harabasz = calinski_harabasz_score(data.drop(['Cluster', 'PCA1', 'PCA2', 'TSNE1', 'TSNE2'], axis=1), cluster_labels)
                    davies_bouldin = davies_bouldin_score(data.drop(['Cluster', 'PCA1', 'PCA2', 'TSNE1', 'TSNE2'], axis=1), cluster_labels)
                    st.write(f"Silhouette Score: {silhouette_avg}")
                    st.write(f"Calinski-Harabasz Score: {calinski_harabasz}")
                    st.write(f"Davies-Bouldin Score: {davies_bouldin}")
                    ########################################################
                    # Predict User's Cluster
                 

                        # Distances between input-vector and centers
                    dist = []
                    for center in cluster_centers:
                        dist.append(np.linalg.norm(user_input_df - center))
                    min_dist = min(dist)

                    assigned_Cluster = dist.index(min_dist)
                    st.subheader('Your Cluster')
                    # Clustername
                    clustername = 'Cluster ' + str(assigned_Cluster)
                    st.write('You belong to ', clustername)

                    
                    user_input_array = user_input_df.values[0]
                    

                    # Radarchart
               
                    fig_cluster = go.Figure()
                    

                    fig_cluster.add_trace(go.Scatterpolar(
                        r=cluster_centers[assigned_Cluster],
                        theta=features,
                        fill='toself',
                        marker=dict(color='darkblue'),
                        name=clustername
                    ))


                    fig_cluster.add_trace(go.Scatterpolar(
                        r=user_input_array,
                        theta=features,
                        fill='toself',
                        marker=dict(color='red'),
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
                
                

                st.subheader('Your Features vs Cluster Center Features')
            

                col1, col2 = st.columns(2, gap="large")
                for i in range(0, len(features)):
                    if i % 2 == 0:
                        with col1:
                            gauge = go.Figure(go.Indicator(
                                mode="gauge+number+delta",
                                value=user_input_array[i],
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': features[i], 'font': {'size': 24}},
                                delta={'reference': cluster_centers[assigned_Cluster][i],
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
                                        'value': cluster_centers[assigned_Cluster][i]}}))

                            gauge.update_layout(paper_bgcolor="white", font={'color': "black", 'family': "Arial"},
                                                height=250, width=300)

                            st.write(gauge)

                    if i % 2 == 1:
                        with col2:
                            gauge = go.Figure(go.Indicator(
                                mode="gauge+number+delta",
                                value=user_input_array[i],
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': features[i], 'font': {'size': 24}},
                                delta={'reference': cluster_centers[assigned_Cluster][i],
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
                                        'value': cluster_centers[assigned_Cluster][i]}}))

                            gauge.update_layout(paper_bgcolor="white", font={'color': "black", 'family': "Arial"},
                                                height=250, width=300)

                            st.write(gauge)
                        ##################################################

                else:
                    st.sidebar.error("Please set all parameters correctly.")

if __name__ == "__main__":
    main()
