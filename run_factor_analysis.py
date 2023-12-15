import pandas

from factor_analyzer import FactorAnalyzer

from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity

import numpy
import matplotlib.pyplot as plt 

dataset = pandas.read_csv("dataset_final.csv")

newdata = dataset.iloc[:,:40].copy()
# print(newdata)

chi2 ,p=calculate_bartlett_sphericity(newdata)
print("chi2:", chi2)
print("p-value:", p)

machine = FactorAnalyzer(n_factors=40, rotation=None)
machine.fit(newdata)
ev, v = machine.get_eigenvalues()
# print(ev)

# machine = FactorAnalyzer(n_factors=4, rotation=None)
# machine.fit(newdata)
# output = machine.loadings_
# print(output)

machine = FactorAnalyzer(n_factors=4, rotation='varimax')
machine.fit(newdata)
factor_loadings = machine.loadings_
# print(factor_loadings)

factor_key = pandas.DataFrame(factor_loadings)
factor_key.to_csv("factor_key.csv", index=False)

results = pandas.DataFrame(columns=['Factor X', 'Factor Y', 'Factor Z', 'Factor W'])

factor_x_list = []
factor_y_list = []
factor_z_list = []
factor_w_list = []

q = 0 
f = 0 

for p in range(1,21645):
	factor_x = 0 
	factor_y = 0 
	factor_z = 0 
	factor_w = 0 

	for i in range(40):
		# Plist = []
		Q1X = newdata.iloc[p,q]*factor_key.iloc[f,0]
		factor_x = factor_x + Q1X
		Q1Y = newdata.iloc[p,q]*factor_key.iloc[f,1]
		factor_y = factor_y + Q1Y
		Q1Z = newdata.iloc[p,q]*factor_key.iloc[f,2]
		factor_z = factor_z + Q1Z
		Q1W = newdata.iloc[p,q]*factor_key.iloc[f,3]
		factor_w = factor_w + Q1W
		q = q+1
		f = f+1
		new_row = pandas.DataFrame([[factor_x, factor_y, factor_z, factor_w]], columns=results.columns)
		results = pandas.concat([results,new_row], ignore_index=True)

	q = 0
	f = 0
	print(p)
	# p = p+1
	# factor_x_list.append(factor_x)
	# factor_y_list.append(factor_y)
	# factor_z_list.append(factor_z)
	# factor_w_list.append(factor_w)
	



# results = pandas.DataFrame.from_records([{
# 		'Factor X': factor_x_list,
# 		'Factor Y': factor_y_list,
# 		'Factor Z': factor_z_list,
# 		'Factor W': factor_w_list
# 	}])

print(results)

results.to_csv("results.csv", index=False)






# Used to visualize the correct number of factors (eigenvalue > 1)
# machine_fa = FactorAnalyzer(rotation = None,impute = "drop",n_factors=newdata.shape[1])
# machine_fa.fit(newdata)
# ev,_ = machine_fa.get_eigenvalues()
# plt.scatter(range(1,newdata.shape[1]+1),ev)
# plt.plot(range(1,newdata.shape[1]+1),ev)
# plt.title('Scree Plot')
# plt.xlabel('Factors')
# plt.ylabel('Eigen Value')
# plt.ylim(0, 18)
# plt.grid()
# plt.show()

