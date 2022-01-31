"""
K-prototypes clustering for mixed categorical and numerical data
"""
# import ipdb
# from collections import defaultdict

from joblib import Parallel, delayed

import numpy as np
from scipy import sparse
from scipy import stats

from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array

from . import kmodes
# from .util import get_max_value_key
from .util import encode_features
# from .util import get_unique_rows
from .util import decode_centroids
from .util import pandas_to_numpy
from .util.dissim import matching_dissim, euclidean_dissim
from .util.init_methods import init_cao, init_huang

# Number of tries we give the initialization methods to find non-empty
# clusters before we switch to random initialization.
MAX_INIT_TRIES = 20
# Number of tries we give the initialization before we raise an
# initialization error.
RAISE_INIT_TRIES = 100


class KPrototypes(kmodes.KModes):
    """k-protoypes clustering algorithm for mixed numerical/categorical data.

    Parameters
    -----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.

    max_iter : int, default: 100
        Maximum number of iterations of the k-modes algorithm for a
        single run.

    num_dissim : func, default: euclidian_dissim
        Dissimilarity function used by the algorithm for numerical variables.
        Defaults to the Euclidian dissimilarity function.

    cat_dissim : func, default: matching_dissim
        Dissimilarity function used by the kmodes algorithm for categorical variables.
        Defaults to the matching dissimilarity function.

    n_init : int, default: 10
        Number of time the k-modes algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of cost.

    init : {'Huang', 'Cao', 'random' or a list of ndarrays}, default: 'Cao'
        Method for initialization:
        'Huang': Method in Huang [1997, 1998]
        'Cao': Method in Cao et al. [2009]
        'random': choose 'n_clusters' observations (rows) at random from
        data for the initial centroids.
        If a list of ndarrays is passed, it should be of length 2, with
        shapes (n_clusters, n_features) for numerical and categorical
        data respectively. These are the initial encoded centroids.

    gamma : float, default: None
        Weighing factor that determines relative importance of numerical vs.
        categorical attributes (see discussion in Huang [1997]). By default,
        automatically calculated from data.

    verbose : integer, optional
        Verbosity mode.

    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    n_jobs : int, default: 1
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    Attributes
    ----------
    cluster_centroids_ : array, [n_clusters, n_features]
        Categories of cluster centroids

    labels_ :
        Labels of each point

    cost_ : float
        Clustering cost, defined as the sum distance of all points to
        their respective cluster centroids.

    n_iter_ : int
        The number of iterations the algorithm ran for.

    epoch_costs_ :
        The cost of the algorithm at each epoch from start to completion.

    gamma : float
        The (potentially calculated) weighing factor.

    Notes
    -----
    See:
    Huang, Z.: Extensions to the k-modes algorithm for clustering large
    data sets with categorical values, Data Mining and Knowledge
    Discovery 2(3), 1998.

    """

    def __init__(self, n_clusters=8, max_iter=100, num_dissim=euclidean_dissim,
                 cat_dissim=matching_dissim, init='Cao', n_init=10, gamma=None,
                 verbose=0, random_state=None, n_jobs=1):

        super(KPrototypes, self).__init__(
            n_clusters, max_iter, cat_dissim, init,
            n_init=n_init,
            verbose=verbose, random_state=random_state,
            n_jobs=n_jobs)

        #
        # KPrototypes adds some numerical columns to KModes
        # num_dissim: distance function for numerical columns
        # gamma scales the categorical attributes distance:
        #       tot_dist = num_dist + gamma*cat_dist
        #
        self.num_dissim = num_dissim
        self.gamma = gamma
        if isinstance(self.init, list) and self.n_init > 1:
            if self.verbose:
                print("Initialization method is deterministic. "
                      "Setting n_init to 1.")
            self.n_init = 1

    def fit(self, X, y=None, categorical=None):
        """Compute k-prototypes clustering.

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
        categorical : Index of columns that contain categorical data
        """
        if categorical is not None:
            assert isinstance(categorical, (int, list, tuple)), "The 'categorical' \
            argument needs to be an integer with the index of the categorical \
            column in your data, or a list or tuple of several of them, \
            but it is a {}.".format(type(categorical))

        # if X a pandas class, then grab values as ndarray
        X = pandas_to_numpy(X)

        # If self.gamma is None, gamma will be automatically determined from
        # the data. The function below returns its value.
        (self._enc_cluster_centroids, self._enc_map,
         self.labels_, self.cost_,
         self.n_iter_, self.epoch_costs_,
         self.gamma) = k_prototypes(X,
                                    categorical=categorical,
                                    n_clusters=self.n_clusters,
                                    max_iter=self.max_iter,
                                    num_dissim_f=self.num_dissim,
                                    cat_dissim_f=self.cat_dissim,
                                    gamma=self.gamma,
                                    init=self.init,  # 'huang' or 'cao' or...
                                    n_init=self.n_init,
                                    verbose=self.verbose,
                                    random_state=self.random_state,
                                    n_jobs=self.n_jobs
                                    )
        return self

    def fit_predict(self, X, y=None, **kwargs):
        """Compute cluster centroids and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).
        """
        # return the labels and the distance for each point
        return self.fit(X, **kwargs).predict(X, **kwargs)

    def predict(self, X, categorical=None):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            New data to predict.
        categorical : Indices of columns that contain categorical data

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """

        # only return the labels, skip the distance vector
        return self.predict2(X, categorical)[0]

    def predict2(self, X, categorical=None):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            New data to predict.
        categorical : Indices of columns that contain categorical data

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        assert hasattr(self, '_enc_cluster_centroids'), "Model not yet fitted."

        if categorical is not None:
            assert isinstance(categorical, (int, list, tuple)), "The 'categorical' \
                argument needs to be an integer with the index of the categorical \
                column in your data, or a list or tuple of several of them, \
                but it is a {}.".format(type(categorical))

        X = pandas_to_numpy(X)
        Xnum, Xcat = _split_num_cat(X, categorical)
        Xnum, Xcat = check_array(Xnum), check_array(Xcat, dtype=None)
        Xcat, _ = encode_features(Xcat, enc_map=self._enc_map)
        return labels_cost(Xnum, Xcat, self._enc_cluster_centroids,
                           self.num_dissim, self.cat_dissim, self.gamma)[0:2]

    @property
    def cluster_centroids_(self):
        if hasattr(self, '_enc_cluster_centroids'):
            return np.hstack((
                self._enc_cluster_centroids[0],
                decode_centroids(self._enc_cluster_centroids[1], self._enc_map)
            ))
        else:
            raise AttributeError("'{}' object has no attribute 'cluster_centroids_' "
                                 "because the model is not yet fitted.")


def _k_prototypes_point_distance(Xnum, Xcat, centroids,
                                 num_dissim_f=euclidean_dissim,
                                 cat_dissim_f=matching_dissim,
                                 gamma=1.):
    # iterate over the centroids, vectorizing the points
    n_clusters = centroids[0].shape[0]
    n_points = Xnum.shape[0]
    pt_dist_num = np.zeros((n_points, n_clusters))
    pt_dist_cat = np.zeros((n_points, n_clusters))

    # for each centroid, compute distance of every point to that centroid
    # store in:  pt_dist(n_points, n_centroids)
    for icentroid in range(n_clusters):
        pt_dist_num[:, icentroid] = num_dissim_f(centroids[0][icentroid], Xnum)
        pt_dist_cat[:, icentroid] = cat_dissim_f(centroids[1][icentroid], Xcat)
        pt_dist = pt_dist_num + (gamma * pt_dist_cat)
    return pt_dist


def labels_cost(Xnum, Xcat, centroids,
                num_dissim_f, cat_dissim_f, gamma):
    """Calculate labels and cost function given a matrix of points and
    a list of centroids for the k-prototypes algorithm.
    """
    # compute distance from every point to every centroid,
    #     store in:  distances(n_points, n_centroids)
    # For each point, the column number of min distance is the cluster label
    # Distance vector is collapsed to hold distance to cluster centroid
    # Overall cost of this configuration is sum of these distances
    distances = _k_prototypes_point_distance(Xnum, Xcat, centroids,
                                             num_dissim_f, cat_dissim_f, gamma)
    labels = distances.argmin(axis=1).astype(np.uint16)
    distances = distances.min(axis=1).astype(np.float64)
    cost = distances.sum()
    return labels, distances, cost


def _k_prototypes_update_centroids(Xnum, Xcat, n_clusters, labels):
    centroids_num = np.zeros((n_clusters, Xnum.shape[1]))
    centroids_cat = np.zeros((n_clusters, Xcat.shape[1]))
    for iclust in range(n_clusters):
        indexset = (labels == iclust)  # list of points in that cluster
        clust_memb_num = Xnum[indexset]  # numer attr of elements of cluster
        clust_memb_cat = Xcat[indexset]  # categ attr of elements of cluster
        # update centroid coords to mean(numer attrs), mode (categ attrs)
        centroids_num[iclust, :] = clust_memb_num.mean(axis=0)
        centroids_cat[iclust, :] = stats.mode(clust_memb_cat, axis=0).mode[0]
    return [centroids_num, centroids_cat]


def k_prototypes(X,
                 categorical=None, n_clusters=1, max_iter=1,
                 num_dissim_f=None, cat_dissim_f=None, gamma=None,
                 init=None, n_init=1, verbose=0, random_state=None, n_jobs=1):
    """k-prototypes algorithm
    :param categorical_cols:
    :param init_method_name:
    """
    random_state = check_random_state(random_state)
    if sparse.issparse(X):
        raise TypeError("k-prototypes does not support sparse data.")

    if categorical is None or not categorical:
        raise NotImplementedError(
            "No categorical data selected, effectively doing k-means. "
            "Present a list of categorical columns, or use scikit-learn's "
            "KMeans instead."
        )
    if isinstance(categorical, int):
        categorical = [categorical]
        assert len(categorical) != X.shape[1], \
            "All columns are categorical, use k-modes instead of k-prototypes."
        assert max(categorical) < X.shape[1], \
            "Categorical index larger than number of columns."

    #
    # split the data into numerical and categorical arrays
    #
    n_cat_attrs = len(categorical)
    n_num_attrs = X.shape[1] - n_cat_attrs
    n_points = X.shape[0]
    assert n_clusters <= n_points, "Cannot have more clusters ({}) " \
        "than data points ({}).".format(n_clusters, n_points)

    Xnum, Xcat = _split_num_cat(X, categorical)
    Xnum, Xcat = check_array(Xnum), check_array(Xcat, dtype=None)

    # Convert the categorical values in Xcat to integers for speed.
    # Based on the unique values in Xcat, we can make a mapping to achieve this.
    Xcat, enc_map = encode_features(Xcat, enc_map=None)  # create the mapping

    # Are there more n_clusters than unique rows? Then set the unique
    # rows as initial values and skip iteration.
    #
    # unique = get_unique_rows(X)
    # n_unique = unique.shape[0]
    # if n_unique <= n_clusters:
    #     max_iter = 0
    #     n_init = 1
    #     n_clusters = n_unique
    #     init = list(_split_num_cat(unique, categorical))
    #     init[1], _ = encode_features(init[1], enc_map)

    # Estimate a good value for gamma, which determines the weighing of
    # categorical values in clusters (see Huang [1997]).
    if gamma is None:
        gamma = 0.5 * Xnum.std()

    results = []
    seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
    if n_jobs == 1:
        for init_no in range(n_init):
            results.append(_k_prototypes_single(Xnum, Xcat, n_num_attrs, n_cat_attrs,
                                                n_clusters, n_points, max_iter,
                                                num_dissim_f, cat_dissim_f, gamma,
                                                init, init_no, verbose, seeds[init_no]))
    else:
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_k_prototypes_single)(Xnum, Xcat, n_num_attrs, n_cat_attrs,
                                          n_clusters, n_points, max_iter,
                                          num_dissim_f, cat_dissim_f, gamma,
                                          init, init_no, verbose, seed)
            for init_no, seed in enumerate(seeds))

    all_centroids, all_labels, all_costs, all_n_iters, all_epoch_costs = zip(*results)

    best = np.argmin(all_costs)
    if n_init > 1 and verbose:
        print("Best run was number {}".format(best + 1))

    # Note: return gamma in case it was automatically determined.
    return all_centroids[best], enc_map, all_labels[best], all_costs[best], \
        all_n_iters[best], all_epoch_costs[best], gamma


def _k_prototypes_init_cat_centroids(Xcat, n_clusters, init, cat_dissim, random_state):
    random_state = check_random_state(random_state)
    if isinstance(init, str) and init.lower() == 'huang':
        centroids = init_huang(Xcat, n_clusters, cat_dissim, random_state)
    elif isinstance(init, str) and init.lower() == 'cao':
        centroids = init_cao(Xcat, n_clusters, cat_dissim)
    elif isinstance(init, str) and init.lower() == 'random':
        seeds = random_state.choice(range(n_points), n_clusters)
        centroids = Xcat[seeds]
    elif isinstance(init, list):
        # Make sure inits are 2D arrays.
        init = [np.atleast_2d(cur_init).T if len(cur_init.shape) == 1
                else cur_init
                for cur_init in init]
        assert init[0].shape[0] == n_clusters, \
            "Wrong number of initial numerical centroids in init " \
            "({}, should be {}).".format(init[0].shape[0], n_clusters)
        assert init[0].shape[1] == nnumattrs, \
            "Wrong number of numerical attributes in init ({}, should be {})." \
            .format(init[0].shape[1], nnumattrs)
        assert init[1].shape[0] == n_clusters, \
            "Wrong number of initial categorical centroids in init ({}, " \
            "should be {}).".format(init[1].shape[0], n_clusters)
        assert init[1].shape[1] == ncatattrs, \
            "Wrong number of categorical attributes in init ({}, should be {})." \
            .format(init[1].shape[1], ncatattrs)
        centroids = [np.asarray(init[0], dtype=np.float64),
                     np.asarray(init[1], dtype=np.uint16)]
    else:
        raise NotImplementedError("Initialization method not supported.")

    return centroids


def _k_prototypes_init_num_centroids(Xnum, n_clusters, random_state):
    # Numerical dimensions of centroid
    # initialized by drawing from normal distribution for each attribute
    random_state = check_random_state(random_state)
    nnumattrs = Xnum.shape[1]
    meanx = np.mean(Xnum, axis=0)
    stdx = np.std(Xnum, axis=0)
    centroids = meanx + random_state.randn(n_clusters, nnumattrs) * stdx
    return centroids


def _k_prototypes_single(Xnum, Xcat, nnumattrs, ncatattrs, n_clusters, n_points,
                         max_iter, num_dissim, cat_dissim, gamma, init, init_no,
                         verbose, random_state):
    '''
    Perform a complete run of k-prototypes with random_state as the seed
    Initialization:
    1) Pick starting centroids
    1a) Use Huang or Cao for categorical attributes
    1b) Draw from a normal distribution centered on each numerical attribute
    1c) Make initial labeling, compute initial cost function

    Iteration:
    1) Update centroid coordinates
    2) Compute distance from every point to every centroid (cost function)
    3) Assign point to nearest centroid    
    4) Cost increased? No points moved?
    '''

    if verbose:
        print("Init: initializing centroids")
    # centroids[0] is (nclusters, n_num_attrs)
    # centroids[1] is (nclusters, n_cat_attrs)
    if verbose:
        print("Initializing centroid numerical attributes")
    centroids_num = _k_prototypes_init_num_centroids(Xnum, n_clusters, random_state)

    if verbose:
        print("Initializing centroid categorical attributes")
    centroids_cat = _k_prototypes_init_cat_centroids(Xcat, n_clusters, init,
                                                     cat_dissim, random_state)

    centroids = [centroids_num, centroids_cat]

    if verbose:
        print("Init: initial cluster assignment")
    labels, distances, cost = labels_cost(Xnum, Xcat, centroids,
                                          num_dissim, cat_dissim, gamma)

    converged = False
    n_iters = 0
    epoch_costs = [cost]
    while not converged:
        # update centroids
        centroids_prev = centroids
        labels_prev = labels
        cost_prev = cost
        n_iters += 1

        if verbose > 1:
            print("Before centroid update:")
            print(centroids[0])
            print(centroids[1])

        centroids = \
            _k_prototypes_update_centroids(Xnum, Xcat, n_clusters, labels_prev)

        if verbose > 1:
            print("After centroid update:")
            print(centroids[0])
            print(centroids[1])

        # labels, distances, cost
        labels, distances, cost = labels_cost(Xnum, Xcat, centroids,
                                              num_dissim, cat_dissim, gamma)
        num_points_moved = np.sum(labels != labels_prev)
        epoch_costs.append(cost)
        if verbose:
            print("{:3d}/{:3d}: {:6d} pts moved, cost {:.6f} to {:.6f}".
                  format(n_iters, max_iter, num_points_moved,
                         cost_prev, cost))

        converged = ((num_points_moved == 0) or
                     (cost > cost_prev) or
                     (n_iters > max_iter))
    return (centroids_prev, labels_prev, cost_prev, n_iters, cost_prev)


def _split_num_cat(X, categorical):
    """Extract numerical and categorical columns.
    Convert to numpy arrays, if needed.

    :param X: Feature matrix
    :param categorical: Indices of categorical columns
    """
    Xnum = np.asanyarray(X[:, [ii for ii in range(X.shape[1])
                               if ii not in categorical]]).astype(np.float64)
    Xcat = np.asanyarray(X[:, categorical])
    return Xnum, Xcat
