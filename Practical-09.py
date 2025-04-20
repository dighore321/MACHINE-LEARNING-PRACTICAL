import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the Heart Disease Dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/heart.dat'
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

# Read dataset from URL
data = pd.read_csv(url, header=None, names=column_names, sep=' ')

# Preprocess: Label encode categorical variables (target is 'target', which is binary)
label_encoder = LabelEncoder()
data['target'] = label_encoder.fit_transform(data['target'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'], test_size=0.3, random_state=42)

# Build Bayesian Network Structure: We assume basic dependencies based on domain knowledge
# You can customize this structure depending on your domain knowledge or data exploration.
model = BayesianNetwork([
    ('age', 'target'),
    ('sex', 'target'),
    ('cp', 'target'),
    ('trestbps', 'target'),
    ('chol', 'target'),
    ('fbs', 'target'),
    ('restecg', 'target'),
    ('thalach', 'target'),
    ('exang', 'target'),
    ('oldpeak', 'target'),
    ('slope', 'target'),
    ('ca', 'target'),
    ('thal', 'target')
])

# Define Conditional Probability Distributions (CPDs) for each node in the network.
# These CPDs should ideally come from expert knowledge or be learned from the data.
# For simplicity, we use random initialization for CPDs. You can replace these with more realistic ones.

cpd_age = TabularCPD(variable='age', variable_card=5, values=[[0.1], [0.2], [0.3], [0.2], [0.2]])
cpd_sex = TabularCPD(variable='sex', variable_card=2, values=[[0.7], [0.3]])
cpd_target = TabularCPD(variable='target', variable_card=2, values=[[0.8, 0.2], [0.3, 0.7]])

# Add CPDs to the model
model.add_cpds(cpd_age, cpd_sex, cpd_target)

# Verify the model
model.check_model()

# Train the model using the training data
# We are learning CPDs here using the data
from pgmpy.factors.discrete import TabularCPD
model.fit(X_train)

# Inference: Make predictions (diagnosis)
inference = VariableElimination(model)
predicted_class = inference.predict(X_test)

# Evaluate accuracy
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Convert predictions to match the original target format
predictions = predicted_class.values

# Compute metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)

# Output the results
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
