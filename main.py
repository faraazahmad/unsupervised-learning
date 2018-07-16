from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# loading dataset
iris_df = datasets.load_iris()

# Available methods on dataset
print(dir(iris_df))

# Available features
print(iris_df.feature_names)

# targets
print(iris_df.target)

# target names
print(iris_df.target_names)
label = {0: 'red', 1: 'blue', 2: 'green'}

# Dataset Slicing
x_axis = iris_df.data[:, 0]		# sepal length
y_axis = iris_df.data[:, 2]		# sepal width

# Plotting
plt.scatter(x_axis, y_axis, c=iris_df.target)
plt.show()

# Declaring model
model = KMeans(n_clusters=3)

# Fitting model
model.fit(iris_df.data)

# predicting a single putput
predicted_label = model.predict([[7.2, 3.5, 0.8, 1.6]])

# prediction on the entire data
all_predictions = model.predict(iris_df.data)

# Printing predictions
print(predicted_label)
print(all_predictions)

# plot the predictions of the model
plt.scatter(x_axis, y_axis, c=all_predictions)
plt.show()