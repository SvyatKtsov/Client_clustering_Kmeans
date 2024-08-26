import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score

from Second_Part_of_Main_Module.Kmeans_Class import *

print(f"centroids are:{centroids}")

s = X.to_numpy()
X = list(s)
Xnp = np.array(X)

# elbow Method
from sklearn.cluster import KMeans

wcss = []
for e_cluster in range(1, 11):
    kmeans = KMeans(n_clusters=e_cluster, init='k-means++', random_state=78)
    kmeans.fit(Xnp)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()

# silhouette analysis
range_n_clusters = [2, 3, 4, 5, 6]

for n_clusters in range_n_clusters:
    fig = plt.figure(figsize=(18, 7))

    # the 1st subplot is the silhouette plot
    ax1 = fig.add_subplot(121)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # initialize the clusterer with n_clusters value and a random generator seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # the silhouette_score gives the average value for all the samples.
    silhouette_avg = silhouette_score(Xnp, cluster_labels)
    print(f"For n_clusters = {n_clusters}, The average silhouette_score is : {silhouette_avg}")

    # compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # second plot showing the actual clusters formed
    ax2 = fig.add_subplot(122, projection='3d')
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(Xnp[:, 0], Xnp[:, 1], Xnp[:, 2], marker=".", lw=0, alpha=0.7, c=colors, edgecolor="k")

    # labeling the clusters
    centers = clusterer.cluster_centers_
    ax2.scatter(centers[:, 0], centers[:, 1], centers[:, 2], marker="o",
                c="white", alpha=1, s=200, edgecolor="k")

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], c[2], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Тривалість тел.розмов клієнта (у хв.)") #total_day_minutes
    ax2.set_ylabel("Вартість тел.розмов клієнта") #total_day_charge
    ax2.set_zlabel("К-ість дзвінків клієнта у службу підтримки") #number_customer_service_calls

    plt.suptitle(
        f"Silhouette analysis for KMeans clustering on sample data with n_clusters = {n_clusters}",
        fontsize=14, fontweight="bold"
    )

plt.show()
