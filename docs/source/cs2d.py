def predict(X):

    """
    Predict of given input.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Vector to be scored, where `n_samples` is the number of samples and
        `n_features` is the number of features.
    Returns
    -------
    T : array-like of shape (n_samples, n_classes)
        Returns the label classes
    """
    prob = 1

    return prob


class CC:
    """


    Parameters
    ----------
    backdoor_penalty : float
       Penalty parameter for the backdoor loss

    selection_penalty : float
       Penalty parameter for the selection mechanism

    solver : str, default = 'adam'
       Algorithm to use in the optimization problem.

    max_iter : int, default=100
        Maximum number of iterations taken for the solvers to converge.

    random_state : int, RandomState instance, default=None
        RandomState instance for shuffling data

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    backdoor_weight: array of shape (n_samples, 1), default = None
           Weights associated with backdoor importance. Default value is an array of all 1.

    """
    def __init__(
        self,
        backdoor_penalty = 0.1,
        selection_penalty = 0.1,
        tol = 1e-4,
        random_state = None,
        solver = "adam",
        max_iter = 100,
        warm_start = False,
        backdoor_weight =  None
    ):

        self.backdoor_penalty = backdoor_penalty
        self.selection_penalty = selection_penalty
        self.tol = tol
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.warm_start = warm_start
        self.backdoor_weight = backdoor_weight

    def fit(self, X, y, backdoor_target, num_of_backdoor):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
          Training vector, where `n_samples` is the number of samples and \
             `n_features` is the number of features.
        y : array-like of shape (n_samples,)
          Target vector relative to X.

        backdoor_target: str
          Target label(s) of backdoor

        num_of_backdoor: int
          Number of backdoor data


        Returns
        ---------
        estimator: self
                 Fitted estimator.


        .. note::
           'sag' and 'saga' fast convergence is only guaranteed on
           features with approximately the same scale. You can


        """

    def score(self, X, y, sample_weight=None):
        """Score using the `scoring` option on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.


        Returns
        -------
        score : float
            Score of self.predict(X) wrt. y.
        """
        scoreing = 1

        return scoreing

    """
    .. note::
        jere
    """

    def backdoor_score(self, X, backdoor_target, sample_weight=None):
        """Score using the `scoring` option on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        backdoor_target : array-like of shape (n_samples,)
            Target label for backdoor.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.


        Returns
        -------
        score : float
            Score of self.predict(X) wrt. backdoor_target.
        """
        scoreing = 1

        return scoreing

    def backdoor_selection(self, X, cutoff, ratio):
        """Return the index of selected backdoor sample for given cutoff value.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        cutoff : float
            The thresholding value for selecting backdoor.
        ratio : float
            The proportion on selecting backdoor samples.


        Returns
        -------
        index : array
            The index of selected backdoor samples.
        """
        scoreing = 1

        return scoreing
