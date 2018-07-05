from sklearn.cluster import DBSCAN


def remove_outliers(data, filter_sample_vectors = None, eps=None, min_samples=5):
    if filter_sample_vectors is not None:
        data = data[filter_sample_vectors]
    if eps is None:
        eps = 5 * np.sqrt(np.square(data.std()).sum())
    db = DBSCAN(eps, min_samples).fit(data)
    inliers = data[db.labels_ != -1]
    outliers = data[db.labels_ == -1]
    return inliers, outliers