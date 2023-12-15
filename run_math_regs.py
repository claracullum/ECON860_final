import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as pyplot

data1 = pd.read_csv("results.csv")
dataset_final = pd.read_csv("dataset_final.csv")

math_data = pd.concat([data1, dataset_final['math']], axis=1)
math_data = math_data[:21642]

#to show relationship between each factor and math ability
pyplot.scatter(math_data.iloc[:, 0], math_data.iloc[:, 4])
pyplot.savefig("math_on_factorX.png")
pyplot.close()

pyplot.scatter(math_data.iloc[:, 1], math_data.iloc[:, 4])
pyplot.savefig("math_on_factorY.png")
pyplot.close()

pyplot.scatter(math_data.iloc[:, 2], math_data.iloc[:, 4])
pyplot.savefig("math_on_factorZ.png")
pyplot.close()

pyplot.scatter(math_data.iloc[:, 3], math_data.iloc[:, 4])
pyplot.savefig("math_on_factorW.png")
pyplot.close()


target = math_data['math']
data_factors = math_data[['Factor X', 'Factor Y', 'Factor Z', 'Factor W']]

machineX = LinearRegression()
machineY = LinearRegression()
machineZ = LinearRegression()
machineW = LinearRegression()

machineX.fit(data_factors[['Factor X']], target)
machineY.fit(data_factors[['Factor Y']], target)
machineZ.fit(data_factors[['Factor Z']], target)
machineW.fit(data_factors[['Factor W']], target)

predictionsX = machineX.predict(data_factors[['Factor X']])
predictionsY = machineY.predict(data_factors[['Factor Y']])
predictionsZ = machineZ.predict(data_factors[['Factor Z']])
predictionsW = machineW.predict(data_factors[['Factor W']])

#all r squared 
r2_x = r2_score(target, predictionsX)
print("Factor X R square score: ", r2_x)
print("\n\n")

r2_y = r2_score(target, predictionsY)
print("Factor Y R square score: ", r2_y)
print("\n\n")

r2_z = r2_score(target, predictionsZ)
print("Factor Z R square score: ", r2_z)
print("\n\n")

r2_w = r2_score(target, predictionsW)
print("Factor W R square score: ", r2_w)
print("\n\n")


