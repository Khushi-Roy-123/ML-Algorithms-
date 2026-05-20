import numpy as np
import pandas as pd

from jklearn.linear_model.linear_regression import LinearRegression


def main():
	data = pd.read_csv("data/data.csv")

	target = data["price"].to_numpy(dtype=float)
	features = data[["bedrooms"]].to_numpy(dtype=float)

	split_index = int(0.8 * len(features))
	x_train = features[:split_index]
	x_test = features[split_index:]
	y_train = target[:split_index]
	y_test = target[split_index:]

	model = LinearRegression(fit_intercept=True, solver="normal")
	model.fit(x_train, y_train)

	predictions = model.predict(x_test)
	mse = np.mean((predictions - y_test) ** 2)

	print("Feature: bedrooms")
	print("Target: price")
	print("Coefficient:", model.coef_)
	print("Intercept:", model.intercept_)
	print("Mean squared error:", mse)
	print("First 5 predictions:", predictions[:5])


if __name__ == "__main__":
	main()

