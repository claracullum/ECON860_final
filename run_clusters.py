import pandas
import matplotlib.pyplot as pyplot
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

dataset = pandas.read_csv("results.csv")
dataset = dataset.values

# scatterplot of data
pyplot.scatter(dataset[:, 0], dataset[:, 1])
pyplot.savefig("scatterplot.png")
pyplot.close()


def run_kmeans(n, dataset):
  machine = KMeans(n_clusters=n)
  machine.fit(dataset)
  results = machine.predict(dataset)
  centroids = machine.cluster_centers_
  pyplot.scatter(dataset[:,0],dataset[:,1], c=results)
  pyplot.scatter(centroids[:,0], centroids[:, 1], c="red", marker="*", s=300)
  pyplot.savefig("scatterplot_kmeans_" + str(n) + ".png")
  pyplot.close()
  return silhouette_score(dataset, results, metric="euclidean")


n_list = [2,3,4,5,6,7,8]
silhouette_score_list = [run_kmeans(i, dataset) for i in n_list]

pyplot.scatter(n_list, silhouette_score_list)
pyplot.savefig("silhouette_score_kmeans.png")
pyplot.close()


def run_gmm(n, dataset):
  machine = GaussianMixture(n_components=n)
  machine.fit(dataset)
  results = machine.predict(dataset)
  centroids = machine.means_
  pyplot.scatter(dataset[:,0],dataset[:,1], c=results)
  pyplot.scatter(centroids[:,0], centroids[:, 1], c="red", marker="*", s=300)
  pyplot.savefig("scatterplot_gmm_" + str(n) + ".png")
  pyplot.close()
  return silhouette_score(dataset, results, metric="euclidean")


n_list = [2,3,4,5,6,7,8]
silhouette_score_list = [run_gmm(i, dataset) for i in n_list]

pyplot.scatter(n_list, silhouette_score_list)
pyplot.savefig("silhouette_score_gmm.png")
pyplot.close()



# kmedoids did not work well with the large dataset so i used 10000 observations to get an idea of its result. 
test_data = dataset[:10000,:]

def run_kmedoids(n, test_data):
  print(f"Running KMedoids with {n} clusters.")
  machine = KMedoids(n_clusters=n)

  try:
      machine.fit(test_data)
      print("fit")
  except Exception as e:
      print(f"Error during KMedoids fit: {e}")
      return 0
  # machine = KMedoids(n_clusters=n)
  # machine.fit(test_data)
  results = machine.predict(test_data)
  centroids = machine.cluster_centers_
  pyplot.scatter(test_data[:,0],test_data[:,1], c=results)
  pyplot.scatter(centroids[:,0], centroids[:, 1], c="red", marker="*", s=300)
  pyplot.savefig("scatterplot_kmedoids_" + str(n) + ".png")
  pyplot.close()
  return silhouette_score(test_data, results, metric="euclidean")

n_list = [2,3,4,5,6,7,8]
silhouette_score_list = [run_kmedoids(i, test_data) for i in n_list]

pyplot.scatter(n_list, silhouette_score_list)
pyplot.savefig("silhouette_score_kmedoids.png")
pyplot.close()


