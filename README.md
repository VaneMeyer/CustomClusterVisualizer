# CustomClusterVisualizer
This Streamlit app enables interaction with any data set. The data can be clustered using selected features with k-means. Details of the associated cluster are visualized using custom values for features. Radar charts and gauge charts are used for visualization.

Note: We use k-means clustering and preprocessing of the scikit-learn library. Problems may occur with non-numeric features. So far, our Streamlit app does not yet contain a function to replace non-numeric values with numeric values. Therefore, only numeric features should be selected.


## Installation

1. Clone Repository:
   git clone https://github.com/VaneMeyer/CustomClusterVisualizer.git
2. Download Streamlit:
   pip install streamlit

### Usage

1. Navigate to project folder:
   cd PathToProjectFolder

2. Start streamlit app:
   streamlit run userinterface.py

#### Screenshots

![streamlit09](https://github.com/VaneMeyer/CustomClusterVisualizer/assets/108664895/db9df322-72ea-4c02-9f91-f546c93c4d5f)

![streamlit10](https://github.com/VaneMeyer/CustomClusterVisualizer/assets/108664895/4a48d82a-df9f-4e0d-a6e0-3adf85198a40)
