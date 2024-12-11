# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.mixture import GaussianMixture
# from scipy.stats import multivariate_normal
#
# # Generate synthetic data
# np.random.seed(0)
# n_samples = 300
#
# # Create three separate clusters
# C1 = np.random.randn(n_samples, 2) + np.array([0, 0])
# C2 = np.random.randn(n_samples, 2) + np.array([5, 5])
# C3 = np.random.randn(n_samples, 2) + np.array([5, 0])
#
# # Combine the clusters to form the dataset
# X = np.vstack([C1, C2, C3])
#
# # Fit a Gaussian Mixture Model
# n_components = 3  # Number of Gaussian components
# gmm = GaussianMixture(n_components=n_components, covariance_type='full')
# gmm.fit(X)
#
# # Predict the cluster for each data point
# labels = gmm.predict(X)
#
# # Plot the data points and the Gaussian components
# plt.figure(figsize=(10, 8))
# plt.scatter(X[:, 0], X[:, 1], c=labels, s=30, cmap='viridis', zorder=2)
#
# # Plot the Gaussian components
# x = np.linspace(-3, 8, 100)
# y = np.linspace(-3, 8, 100)
# X_grid, Y_grid = np.meshgrid(x, y)
# pos = np.dstack((X_grid, Y_grid))
#
# # Evaluate the probability density function for each component
# for i in range(n_components):
#     mean = gmm.means_[i]
#     cov = gmm.covariances_[i]
#     rv = multivariate_normal(mean=mean, cov=cov)
#     plt.contour(X_grid, Y_grid, rv.pdf(pos), levels=5, colors='black', linewidths=1)
#
# plt.title('Mixture of Gaussians')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.show()


#
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.mixture import GaussianMixture
# from scipy.stats import multivariate_normal
#
# # Load the dataset from an Excel file
# file_path = 'path/to/your/dataset.xlsx'  # Change this to the path of your Excel file
# sheet_name = 'Sheet1'  # Change this if your data is in a different sheet
# data = pd.read_excel(file_path, sheet_name=sheet_name)
#
# # Assuming your data is in two columns (you may need to adjust this)
# X = data.iloc[:, :2].values  # Adjust the column selection as necessary
#
# # Fit a Gaussian Mixture Model
# n_components = 3  # Number of Gaussian components
# gmm = GaussianMixture(n_components=n_components, covariance_type='full')
# gmm.fit(X)
#
# # Predict the cluster for each data point
# labels = gmm.predict(X)
#
# # Plot the data points and the Gaussian components
# plt.figure(figsize=(10, 8))
# plt.scatter(X[:, 0], X[:, 1], c=labels, s=30, cmap='viridis', zorder=2)
#
# # Plot the Gaussian components
# x = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
# y = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)
# X_grid, Y_grid = np.meshgrid(x, y)
# pos = np.dstack((X_grid, Y_grid))
#
# # Evaluate the probability density function for each component
# for i in range(n_components):
#     mean = gmm.means_[i]
#     cov = gmm.covariances_[i]
#     rv = multivariate_normal(mean=mean, cov=cov)
#     plt.contour(X_grid, Y_grid, rv.pdf(pos), levels=5, colors='black', linewidths=1)
#
# plt.title('Mixture of Gaussians')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.show()


import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 1. הכנת הנתונים
# לדוגמה, נשתמש בנתוני Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 2. חילוק נתונים
cv = StratifiedKFold(n_splits=10)

# 3. אימון המודל
model = make_pipeline(StandardScaler(), SVC(kernel='linear'))
scores = cross_val_score(model, X, y, cv=cv)

print(f'Cross-validation scores: {scores}')
print(f'Mean score: {scores.mean()}')

# 4. חישוב חשיבות המאפיינים
model.fit(X, y)  # לאימון על כל הנתונים
importance = model.named_steps['svc'].coef_

# 5. שמירת התוצאות
feature_importance = pd.DataFrame(importance, columns=iris.feature_names, index=iris.target_names)
feature_importance = feature_importance.abs()  # ערכים מוחלטים
top_features = feature_importance.apply(lambda x: x.nlargest(20), axis=1)

# נשמור לקובץ Excel
top_features.to_excel('top_features.xlsx')

print(top_features)
