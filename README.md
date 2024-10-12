# Advanced Iris Flower Prediction App

## Overview

This Streamlit-based web application predicts the type of Iris flower using a Random Forest classifier. It offers an interactive interface for users to input flower measurements, adjust model parameters, and receive predictions. The app also provides insights into model performance and feature importance.

## Features

- Predicts Iris flower species based on sepal and petal measurements
- Interactive sliders for inputting flower measurements and adjusting model parameters
- Displays prediction probabilities for each Iris species
- Shows model performance using cross-validation
- Visualizes feature importance
- Allows model retraining with adjustable parameters
- Saves and loads trained models for persistence

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/ArthurTechy/advanced-iris-prediction-app.git
   cd advanced-iris-prediction-app
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run iris_app.py
   ```

2. Open your web browser and go to `http://localhost:8501`

3. Use the sidebar sliders to input flower measurements and adjust model parameters

4. Click the "Retrain Model" button to train a new model with the current parameters

5. Explore the predictions, model performance, and feature importance visualization in the main panel

## Dependencies

- streamlit
- pandas
- numpy
- scikit-learn
- plotly
- joblib

For a complete list of dependencies with versions, see `requirements.txt`.

## Model

The app uses a Random Forest classifier from scikit-learn. The model can be retrained with different parameters:

- Number of trees (n_estimators)
- Maximum depth of trees (max_depth)
- Minimum samples required to split an internal node (min_samples_split)

## Data

The app uses the classic Iris dataset from scikit-learn, which includes measurements for three Iris species: setosa, versicolor, and virginica.

## File Structure

- `iris_app.py`: Main application script
- `requirements.txt`: List of Python dependencies
- `iris_model.joblib`: Saved model file (created when the model is trained)

## Model Persistence

The trained model is saved to `iris_model.joblib` after training. On subsequent runs, the app will load this model unless retraining is requested.

## Cross-Validation

The app uses 5-fold cross-validation to evaluate model performance, providing a more robust estimate of how well the model generalizes to unseen data.

## Contributing

Contributions to improve the app are welcome. Please feel free to submit a Pull Request or open an Issue for discussion.

## License

This project is open source and available under the [MIT License](LICENSE).

## Contact

For any queries or suggestions, please open an issue on the GitHub repository.

View the app here: [Link](https://advanced-iris-prediction-app.streamlit.app/)
