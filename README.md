# Cluster Analysis Exercise - k-Means(++) and Data-Compression

Today we will dive into unsupervised learning and experiment with *clustering*. k-means(++) is a popular choice (maybe even the most widely used clustering algorithm) when it comes to clustering data. Therefore, we will explore this algorithm a bit more in the following three tasks. In this exercise, we will continue to use `sklearn`, which implements a variety of clustering algorithms in the `sklearn.cluster` package. 

### Task 1: A detailed look into k-Means

 The goal of this exercise part is to peek under the hood of this algorithm in order to empirically explore its strengths and weaknesses. Initially, we will use synthetic data to develop a basic understanding of the algorithm's performance characteristics.

In the literature on cluster analysis, k-means often refers not only to the clustering algorithm but also to the underlying optimization problem:

$$\min_{C \subset \mathbb{R}^d, |C| = k} \underbrace{\sum_{x \in P} \min_{c \in C} \lVert x - c \rVert^2}_{\text{Inertia}}$$
    
In cases where k-means refers to the problem formulation, the algorithm itself is sometimes called the Lloyd's algorithm. The algorithm repeats two steps until it converges:
***
1. Assign each data point to its nearest cluster center based on the squared Euclidean norm.

2. Update the centers by calculating the mean of each cluster center and using it as the new cluster center. 
***
This algorithm will always converge and find a solution. However, there is no guarantee that this solution is the best solution, and the algorithm may converge slowly. Also, the algorithm requires an initial guess for the cluster centers. This is usually done by randomly selecting some of the points to be the initial centers. Therefore, it is good practice to run the algorithm several times with different initial center guesses to find a solution that is hopefully close to the best solution.

Fortunately, the implementation of k-means in `sklearn` takes care of all these details and provides us with a simple interface to control all these aspects.

Navigate to `src/ex1_kmeans.py`. Implement the first part of the `plot_kmeans_clustering` function as follows:

1. Load the input data from the given path. You can now run the file and examine the data.
2. k-means clustering is scale sensitive. This means that we generally need to rescale our input data before performing clustering. Note that our `plot_kmeans_clustering` function has a `standardize` parameter that is set to `False` by default. Standardize the data according to $x_i = \frac{x_i - \mu}{\sigma}$ where $\mu$ is the sample mean, $\sigma$ is the sample standard deviation, in case that `standardize` is set to `True`. `sklearn.preprocessing` may be helpful.

Now we want to perform k-means clustering. Implement the `perform_kmeans_clustering` function following these steps:
3. Use `sklearn.cluster.KMeans` to train on the given data. Set the parameter `init`, which controls the initialization of the cluster centers, to `random`. There is a better way to set this value, but we will discuss that in Task 3. 
4. Retrieve the cluster centers and predict the cluster index for each point. 
5. Return the inertia as a float, the cluster centers and the predicted cluster indices as an array each. 

Go back to the `plot_kmeans_clustering` function and finish the remaining TODOs:

6. Call the `perform_kmeans_clustering` function three times. Visualize the data points, cluster centers and the assignment of data points to cluster centers in a single scatter plot. To do this, use a for-loop and `scatter_clusters_2d` to plot all the results in one plot (have a closer look at matplotlib subplots and axes).
7. Return the figure object.

8. Now have a look at the `main` function. The default number $k$ of clusters is not optimal. Experiment with different values and set the number of k-means clusters you want to use.

Note: Data preprocessing benefits greatly from expert knowledge of the field/application in which the data was measured. Some preprocessing methods may not be applicable in certain settings.


#### Decision Boundaries

Recall that k-means assigns a given point $x$ to a center $c_i$ if there is no center $c_j$ with a smaller squared Euclidean distance. This corresponds to the cells of a Voronoi diagram and could look like this:

![Voronoi diagram](./figures/Euclidean_Voronoi_diagram.svg)

Each cell in this diagram is the set of points which are closest to a center:

$$R_j = \{x \in X \mid d(x, c_j) \leq d(x, c_i) \text{ for all } j \neq i\}.$$

A Voronoi diagram can be used as a tool to visualize the boundaries of the k-means cluster, but is also useful as a tool to understand the algorithm.

9. Navigate to the `plot_decision_boundaries` function and load, preprocess and cluster the synthetic data using the function`perform_kmeans_clustering` again.
10. Use `Voronoi` and `voronoi_plot_2d` from the `scipy.spatial` package to visualize the boundaries of the k-means clusters. Again use the `ax` object of the plot and `scatter_clusters_2d`.
11. Test your code with the test framework of vscode or by typing `nox -r -s test` in your terminal.
12. (Optional) Which assumptions/limitations of the k-Means algorithm are illustrated by this visualization?

### Task 2: Data compression - Color Quantization

A common application of clustering is data compression, where large amounts of data are reduced by extracting and keeping only the "important" aspects (i.e., cluster centers, covariance matrices and weights). You may want to use this technique if you cannot store or transmit all the measurements. It can also be useful to reduce the amount of data if a tool/function you want to use in your analysis has a runtime that makes it infeasible to use on thousands or millions of data points. Sometimes, k-means takes a long time to perform the clustering (although you can apply it to large datasets). If you want to speed things up, you can consider using `MiniBatchKMeans`, which performs k-means clustering by drawing multiple random subsets of the entire data.

In this task the goal is to reduce the storage requirement of an image with width $w$ and height $h$ from the dimension $3\cdot w\cdot h$ to $w\cdot h + 3\cdot k$ via clustering: 

1. Open the file `src/ex2_image_compression.py`. The image is loaded in the `main` function using the `load_image` function. Inspect the `input_img` variable and print the information about its dimensions.

Implement the `compress_colorspace` function using the k-means algorithm: 
2. Reshape the input image into $(w\cdot h, 3)$ to perform clustering on colors.
3. Use `MiniBatchKMeans` to cluster the image into $k$ clusters.
4. Return a compressed image where the number of unique colors where reduced from $256^3$ to $k$ via k-means clustering. The compressed image must have the same shape as the original one.

5. Use `compress_colorspace` in your `main` function to compress the image for $k \in \{2,8,64,256\}$ and plot the result using imshow. Set the corresponding value of $k$ as title for each result. 

6. Test your code with the test framework of vscode or by typing `nox -r -s test` in your terminal.

### Task 3 (Optional): k-Means++

As mentioned above, Lloyd's algorithm requires an initial set of centers. Looking more closely at the sklearn documentation, the second argument allows us to use either uniformly randomly selected points or something called "kmeans++" or user-defined array as the initial set of centers. The main contribution of k-means++ is a clever strategy for choosing the initial centers:
***
1. Choose a point $x_1 \in P$ uniformly at random, set $C^1 = \{ x_1 \}$.
2. **for** $i = 0$ to $k$ **do**:
3. $\qquad$ Draw a point $x_i \in P$ according to the probability distribution

$$\frac{\min_{c \in C^{i-1}} \lVert x-c \rVert_2^2}{\sum_{y \in P} \min_{c \in C^{i-1}} \lVert y - c \rVert_2^2}$$

4. $\qquad$ Set $C^{i} = C^{i-1} \cup \{x_i\}$.
5. **end for**
***

Navigate into `src/ex3_kmeans_plus_plus.py` and have a look at the code.

1. Implement the `uniform_sampling` function by drawing points uniformly from the datasets.
2. Implement the `d2_sampling` function using the $D^2$ sampling algorithm described above.
3. Compare the results on the two datasets by executing the scirpt with `python ./src/ex3_kmeans_plus_plus.py`. Which advantages does $D^2$ sampling provide as an initialization?
   **Hint**: (Weighted) sampling with and without replacement can be performed using `np.random.choice`.
4. Test your code with the test framework of vscode or by typing `nox -r -s test` in your terminal.

### Task 4 (Optional): Comparison between k-Means and Gaussian Mixture Models

Navigate into `src/ex4_gmm.py` and have a look at the code. We are creating synthetic dataset with three classes (the same that we used in the lecture) an want to compare k-means clustering and GMMs. If you want, you can take the diabetes dataset from Day 04. Implement the TODOs in the file.
