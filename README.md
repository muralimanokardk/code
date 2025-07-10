*Breast Cancer Classification using Naive Bayes*

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
df = pd.read_csv('breast_cancer.csv')

# a) Print the 1st five rows
print(df.head())

# b) Basic statistical computations
print(df.describe())

# c) The columns and their data types
print(df.info())

# d) Detect null values and replace with mode
for col in df.columns:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

# e) Split the data into test and train
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Evaluate the performance of the model
y_pred = gnb.predict(X_test)
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('F1-score:', f1_score(y_test, y_pred))


 *Find _S algorithm*
 
 import pandas as pd
# Define the dataset
data = pd.DataFrame({
    'Size': ['Big', 'Small', 'Small', 'Big', 'Small'],
    'Color': ['Red', 'Red', 'Red', 'Blue', 'Blue'],
    'Shape': ['Circle', 'Triangle', 'Circle', 'Circle', 'Circle'],
    'Class': ['No', 'No', 'Yes', 'No', 'Yes']
})
# Define the Find-S algorithm
def find_s(data):
    hypothesis = ['0'] * (len(data.columns) - 1)
    for index, row in data.iterrows():
        if row['Class'] == 'Yes':
            for i, col in enumerate(data.columns[:-1]):
                if hypothesis[i] == '0':
                    hypothesis[i] = row[col]
                elif hypothesis[i] != row[col]:
                    hypothesis[i] = '?'
    return hypothesis
# Find the most specific hypothesis
hypothesis = find_s(data)
print('Most Specific Hypothesis:', hypothesis)

 *Polynomial regression* 
 
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
# Generate some data
np.random.seed(0)
X = np.random.rand(100, 1)
y = 3 * X**2 + 2 * X + 1 + np.random.randn(100, 1)
# Train a polynomial regression model
model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
model.fit(X, y)
# Evaluate the performance of the model
y_pred = model.predict(X)
mse = np.mean((y_pred - y)**2)
print('Mean Squared Error:', mse)
# Plot the data and the regression curve
plt.scatter(X, y)
plt

 *Polynomial and KNN* 
 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
# Generate some data
np.random.seed(0)
X = np.random.rand(100, 1)
y = 3 * X**2 + 2 * X + 1 + np.random.randn(100, 1)
# Train a polynomial regression model
model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
model.fit(X, y)
# Predict
y_pred = model.predict(X)
# Evaluate the performance of the model
mse = mean_squared_error(y, y_pred)
print('Mean Squared Error:', mse)
# Plot the data and the regression curve
plt.scatter(X, y, label='Data')
plt.plot(X, y_pred, color='red', label='Polynomial Regression')
plt.legend()
plt.show()

 *KNN ALGORITHM* 
 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Generate some data
np.random.seed(0)
X = np.random.rand(100, 2)
y = np.where(X[:, 0] + X[:, 1] > 1, 1, 0)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Standardize features by removing the mean and scaling to unit variance
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
# Train a KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_std, y_train)
# Predict
y_pred = knn.predict(X_test_std)
# Evaluate the performance of the model
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

 *Logistic regression* 
 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Generate some data
np.random.seed(0)
X = np.random.rand(100, 2)
y = np.where(X[:, 0] + X[:, 1] > 1, 1, 0)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Standardize features
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
# Train a logistic regression model
lr = LogisticRegression()
lr.fit(X_train_std, y_train)
# Predict
y_pred = lr.predict(X_test_std)
# Evaluate the performance of the model
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

 *EM ALGORITHM*
 
import numpy as np
import matplotlib.pyplot as plt
# Generate some data from a mixture of two Gaussians
np.random.seed(0)
mu1, mu2 = 0, 5
sigma1, sigma2 = 1, 2
pi = 0.5
n_samples = 1000
X = np.concatenate([np.random.normal(mu1, sigma1, int(n_samples * pi)),
                    np.random.normal(mu2, sigma2, int(n_samples * (1 - pi)))])
X = X.reshape(-1, 1)
# Initialize parameters
pi = 0.5
mu1, mu2 = np.random.rand(2)
sigma1, sigma2 = np.random.rand(2) + 1

# EM algorithm

for _ in range(100):
    # E-step
    responsibilities = np.zeros((len(X), 2))
    responsibilities[:, 0] = pi * np.exp(-((X - mu1) ** 2) / (2 * sigma1 ** 2)) / (sigma1 * np.sqrt(2 * np.pi))
    responsibilities[:, 1] = (1 - pi) * np.exp(-((X - mu2) ** 2) / (2 * sigma2 ** 2)) / (sigma2 * np.sqrt(2 * np.pi))
    responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)

    # M-step
    pi = np.mean(responsibilities[:, 0])
    mu1 = np.sum(responsibilities[:, 0] * X[:, 0]) / np.sum(responsibilities[:, 0])
    mu2 = np.sum(responsibilities[:, 1] * X[:, 0]) / np.sum(responsibilities[:, 1])
    sigma1 = np.sqrt(np.sum(responsibilities[:, 0] * (X[:, 0] - mu1) ** 2) / np.sum(responsibilities[:, 0]))
    sigma2 = np.sqrt(np.sum(responsibilities[:, 1] * (X[:, 0] - mu2) ** 2) / np.sum(responsibilities[:, 1]))

# Print the estimated parameters
print("Estimated parameters:")
print(f"pi: {pi:.2f}")
print(f"mu1: {mu1:.2f}")
print(f"mu2: {mu2:.2f}")
print(f"sigma1: {sigma1:.2f}")
print(f"sigma2: {sigma2:.2f}")

# Plot the data and the estimated Gaussian components
plt.hist(X, bins=30, density=True, alpha=0.5, label='Data')
x = np.linspace(X.min(), X.max(), 100)
plt.plot(x, pi * np.exp(-((x - mu1) ** 2) / (2 * sigma1 ** 2)) / (sigma1 * np.sqrt(2 * np.pi)), label='Component 1')
plt.plot(x, (1 - pi) * np.exp(-((x - mu2) ** 2) / (2 * sigma2 ** 2)) / (sigma2 * np.sqrt(2 * np.pi)), label='Component 2')
plt.legend()
plt.show() give this paragraphs without hashtags and make the titles bold
