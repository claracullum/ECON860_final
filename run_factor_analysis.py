import pandas

from factor_analyzer import FactorAnalyzer

from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity

import numpy
import matplotlib.pyplot as plt 

dataset = pandas.read_csv("dataset_final.csv")

#print(dataset)

chi2 ,p=calculate_bartlett_sphericity(dataset)
print(chi2, p)

machine = FactorAnalyzer(n_factors=40, rotation=None)
machine.fit(dataset)
ev, v = machine.get_eigenvalues()
print(ev)

machine = FactorAnalyzer(n_factors=5, rotation=None)
machine.fit(dataset)
output = machine.loadings_
print(output)

machine = FactorAnalyzer(n_factors=5, rotation='varimax')
machine.fit(dataset)
factor_loadings = machine.loadings_
print(factor_loadings)

# machine_fa = FactorAnalyzer(rotation = None,impute = "drop",n_factors=dataset.shape[1])
# machine_fa.fit(dataset)
# ev,_ = machine_fa.get_eigenvalues()
# plt.scatter(range(1,dataset.shape[1]+1),ev)
# plt.plot(range(1,dataset.shape[1]+1),ev)
# plt.title('Scree Plot')
# plt.xlabel('Factors')
# plt.ylabel('Eigen Value')
# plt.ylim(0, 18)
# plt.grid()
# plt.show()

