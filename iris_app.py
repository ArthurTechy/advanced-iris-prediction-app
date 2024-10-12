import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors

# Constants
RANDOM_STATE = 42

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the Iris dataset."""
    try:
        iris = datasets.load_iris()
        features = iris.data
        labels = iris.target
        feature_names = iris.feature_names
        target_names = iris.target_names
        
        # Create a DataFrame for easier handling
        df = pd.DataFrame(features, columns=feature_names)
        df['species'] = pd.Categorical.from_codes(labels, target_names)
        
        return df, feature_names, target_names
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

def train_and_evaluate_model(X, y, n_estimators, max_depth, min_samples_split):
    """Train the Random Forest model and evaluate using cross-validation."""
    try:
        clf = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            min_samples_split=min_samples_split,
            random_state=RANDOM_STATE
        )
        scores = cross_val_score(clf, X, y, cv=5)
        clf.fit(X, y)  # Fit the model on all data for final model
        return clf, scores
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None

def plot_feature_comparison(df, user_input, feature_names):
    st.subheader("Feature Comparison")
    
    fig = go.Figure()
    
    for feature in feature_names:
        fig.add_trace(go.Box(y=df[feature], name=feature, boxpoints='all', jitter=0.3, pointpos=-1.8))
        fig.add_trace(go.Scatter(x=[feature], y=[user_input[feature]], 
                                 mode='markers', name='Your Input',
                                 marker=dict(color='red', size=10, symbol='star')))

    fig.update_layout(title="Your Input vs. Dataset Distribution",
                      yaxis_title="Value")
    st.plotly_chart(fig, use_container_width=True)

def plot_radar_chart(user_input, feature_names):
    st.subheader("Radar Chart of Your Input")
    
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[user_input[feature] for feature in feature_names],
        theta=feature_names,
        fill='toself',
        name='Your Input'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max([user_input[feature] for feature in feature_names])]
            )),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

def plot_prediction_probability(prediction_proba, target_names):
    st.subheader("Prediction Probability")
    
    fig = px.bar(x=target_names, y=prediction_proba[0], 
                 labels={'x': 'Species', 'y': 'Probability'},
                 title='Prediction Probability for Each Species')
    
    st.plotly_chart(fig, use_container_width=True)

def plot_nearest_neighbors(df, user_input, feature_names, k=5):
    st.subheader(f"Nearest {k} Neighbors")
    
    # Convert user_input to a DataFrame
    user_input_df = pd.DataFrame([user_input])
    
    # Find nearest neighbors
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn.fit(df[feature_names])
    distances, indices = nn.kneighbors(user_input_df)
    
    # Create a DataFrame for the nearest neighbors
    nearest_df = df.iloc[indices[0]].copy()
    nearest_df['distance'] = distances[0]
    
    # Add user input to the DataFrame
    user_input_df['species'] = 'Your Input'
    user_input_df['distance'] = 0
    
    plot_df = pd.concat([user_input_df, nearest_df], ignore_index=True)
    
    fig = px.scatter_matrix(plot_df, dimensions=feature_names, color='species', 
                            symbol='species', symbol_map={'Your Input': 'star'},
                            title=f"Your Input and {k} Nearest Neighbors")
    
    st.plotly_chart(fig, use_container_width=True)

def main():
    # Set up session state
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'scores' not in st.session_state:
        st.session_state.scores = None
    if 'params' not in st.session_state:
        st.session_state.params = {}
    if 'model_id' not in st.session_state:
        st.session_state.model_id = 0

    st.write("""
    # Advanced Iris Flower Prediction App

    This app predicts the **Iris flower** type using a Random Forest classifier.
    """)

    # Load and preprocess data
    df, feature_names, target_names = load_and_preprocess_data()

    if df is not None:
        X = df[feature_names]
        y = df['species'].cat.codes
        
        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Model parameters (in the main area now, not sidebar)
        st.subheader('Model Parameters')
        col1, col2, col3 = st.columns(3)
        with col1:
            n_estimators = st.slider('Number of trees', 10, 100, 50)
        with col2:
            max_depth = st.slider('Max depth', 1, 20, 5)
        with col3:
            min_samples_split = st.slider('Min samples to split', 2, 10, 2)
        
        # Check if parameters have changed
        current_params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'min_samples_split': min_samples_split}
        params_changed = current_params != st.session_state.params
        
        # Train or use existing model
        if st.session_state.model is None or params_changed or st.button('Retrain Model'):
            st.session_state.model, st.session_state.scores = train_and_evaluate_model(X_scaled, y, n_estimators, max_depth, min_samples_split)
            st.session_state.params = current_params
            st.session_state.model_id += 1  # Increment model ID to force plot redraw
        
        if st.session_state.model is not None:
            # Model evaluation
            st.subheader('Model Performance (Cross-Validation)')
            if st.session_state.scores is not None:
                st.write(f"Mean Accuracy: {st.session_state.scores.mean():.2f} (+/- {st.session_state.scores.std() * 2:.2f})")
            else:
                st.write("No scores available. Please retrain the model.")
            
            # User input for prediction (in sidebar)
            st.sidebar.header('Predict Iris Type')
            user_input = {}
            for feature in feature_names:
                user_input[feature] = st.sidebar.slider(f"{feature}", 
                                                        float(df[feature].min()), 
                                                        float(df[feature].max()), 
                                                        float(df[feature].mean()))
            
            # Make prediction
            input_df = pd.DataFrame([user_input])
            input_scaled = scaler.transform(input_df)
            prediction = st.session_state.model.predict(input_scaled)
            prediction_proba = st.session_state.model.predict_proba(input_scaled)
            
            # Display results
            st.sidebar.subheader('Prediction:')
            st.sidebar.write(target_names[prediction[0]])
            
            st.sidebar.subheader('Prediction Probability')
            st.sidebar.write(pd.DataFrame(prediction_proba, columns=target_names))
            
            # Visualizations based on user input
            plot_feature_comparison(df, user_input, feature_names)
            plot_radar_chart(user_input, feature_names)
            plot_prediction_probability(prediction_proba, target_names)
            plot_nearest_neighbors(df, user_input, feature_names)

    else:
        st.error("Failed to load data. Please check your data source and try again.")

if __name__ == "__main__":
    main()
