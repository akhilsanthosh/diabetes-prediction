# Early Diabetes Prediction System

## Overview
The Early Diabetes Prediction System leverages advanced machine learning techniques to predict the likelihood of diabetes, aiding in early diagnosis and prevention. This system focuses on helping high-risk individuals make informed lifestyle changes, reducing the risk of complications associated with diabetes.

## Features
- Implements multiple machine learning models, including:
  - K-Nearest Neighbors (KNN)
  - Logistic Regression
  - XGBoost
  - Support Vector Machines (SVM)
  - Naïve Bayes
  - Random Forest
- Data preprocessing to handle missing or zero values.
- Exploratory data analysis with visualizations (e.g., heatmaps, KDE plots) to identify correlations and patterns.
- Performance evaluation of models to determine the best-performing algorithm.

## Dataset
The system utilizes the `Diabetes.csv` dataset, which includes features such as:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age
- Outcome (0 or 1 indicating the absence or presence of diabetes)

## Installation
### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook (optional for notebook execution)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/DiabetesPredictionSystem.git
   ```
2. Navigate to the project directory:
   ```bash
   cd DiabetesPredictionSystem
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Running the Python Script
Run the main script to train models and view results:
```bash
python Diabetes\ prediction.py
```

### Using the Jupyter Notebook
Open the notebook to explore step-by-step implementation and visualizations:
```bash
jupyter notebook Diabetes\ predictions.ipynb
```

## Visualizations
The project includes:
- **Correlation Heatmaps**: Understand relationships between features.
- **KDE and Violin Plots**: Compare feature distributions across outcomes.

## Results
- Models are evaluated based on accuracy, precision, recall, and other metrics.
- The best-performing model is highlighted, ensuring reliable predictions.

## Contributions
Contributions are welcome! Feel free to open issues or submit pull requests to improve the system.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- Dataset sourced from [PIMA Indians Diabetes Database].
- Inspiration from the integration of machine learning in healthcare.

