# Instructions

1. run_factor_analysis.py
>This takes the original dataset and uses the 40 question responses to create categories of people.
>Using factor analysis, we find that the best number of factors is 4 for the dataset.
>This can be visualized by uncommenting lines 38-48 in plot form.
>results.csv is the output. Each row is an individual respondant and each column is their score
>for each factor. 

2.  run_clusters.py
>This file shows the results of running Kmean, Kmedoids and GMM clustering. The results data is
>plotted then grouped based on the algorithm. The outputs are: 1 scatterplot showing the overall
>shape of the data, 7 scatterplots showing Kmean clusters, 7 plots showing Kmedoids cluster, 7 plots
>showing GMM clusters, and 3 plots showing the silhoutte scores for each number of clusters. 

3. run_math_regs.py
>This regresses each factor on math ability to see if there is a linear relationship. 4 plots show
>the relationship. Each one also indicates its corresponding R square score.
