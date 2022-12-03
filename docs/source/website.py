

class Dog:
 
    # Class Variable
    animal = 'dog'
 
    # The init method or constructor
    def __init__(self, breed):
 
        # Instance Variable
        self.breed = breed
 
    # Adds an instance variable
    def setColor(self, color):
    	
       """
	 This method will be used to add two numbers
         
         :param int color: The first number
         
         :returns: The sum of two numbers
         
         :rtype: int
    	

         .. note::  The above predict and score methods serve for general prediction/inference purposes.
       """


       self.color = color
 

class CS2D:
    
    """

    Parameters
    ----------
    penalty : {'l1', 'l2', 'elasticnet', 'none'}, default='l2'
        Specify the norm of the penalty:
        - `'none'`: no penalty is added;
        - `'l2'`: add a L2 penalty term and it is the default choice;
        - `'l1'`: add a L1 penalty term;
        - `'elasticnet'`: both L1 and L2 penalty terms are added.


    dual : bool, default=False
        Dual or primal formulation. Dual formulation is only implemented for
        l2 penalty with liblinear solver. Prefer dual=False when
        n_samples > n_features.
    tol : float, default=1e-4
        Tolerance for stopping criteria.
    C : float, default=1.0
        Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger
        regularization.
    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.
    intercept_scaling : float, default=1
        Useful only when the solver 'liblinear' is used
        and self.fit_intercept is set to True. In this case, x becomes
        [x, self.intercept_scaling],
        i.e. a "synthetic" feature with constant value equal to
        intercept_scaling is appended to the instance vector.
        The intercept becomes ``intercept_scaling * synthetic_feature_weight``.
        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased.
    class_weight : dict or 'balanced', default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.
        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.
        .. versionadded:: 0.17
           *class_weight='balanced'*
    random_state : int, RandomState instance, default=None
        Used when ``solver`` == 'sag', 'saga' or 'liblinear' to shuffle the
        data. 
    solver : {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}, \
            default='lbfgs'
        Algorithm to use in the optimization problem. Default is 'lbfgs'.
        To choose a solver, you might want to consider the following aspects:
            - For small datasets, 'liblinear' is a good choice, whereas 'sag'
              and 'saga' are faster for large ones;
            - For multiclass problems, only 'newton-cg', 'sag', 'saga' and
              'lbfgs' handle multinomial loss;
            - 'liblinear' is limited to one-versus-rest schemes.
        .. warning::
           The choice of the algorithm depends on the penalty chosen:
           Supported penalties by solver:
           - 'newton-cg'   -   ['l2', 'none']
           - 'lbfgs'       -   ['l2', 'none']
           - 'liblinear'   -   ['l1', 'l2']
           - 'sag'         -   ['l2', 'none']
           - 'saga'        -   ['elasticnet', 'l1', 'l2', 'none']
        .. note::
           'sag' and 'saga' fast convergence is only guaranteed on
           features with approximately the same scale.
        .. seealso::
           Refer to the User Guide for more information regarding
           summarizing solver/penalty supports.
    max_iter : int, default=100
        Maximum number of iterations taken for the solvers to converge.
    multi_class : {'auto', 'ovr', 'multinomial'}, default='auto'
        If the option chosen is 'ovr', then a binary problem is fit for each
        label. For 'multinomial' the loss minimised is the multinomial loss fit
        across the entire probability distribution, *even when the data is
        binary*. 'multinomial' is unavailable when solver='liblinear'.
        'auto' selects 'ovr' if the data is binary, or if solver='liblinear',
        and otherwise selects 'multinomial'.

    verbose : int, default=0
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.
    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        Useless for liblinear solver. 
  
    n_jobs : int, default=None
        Number of CPU cores used when parallelizing over classes if
        multi_class='ovr'". This parameter is ignored when the ``solver`` is
        set to 'liblinear' regardless of whether 'multi_class' is specified or
        not. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors.

    l1_ratio : float, default=None
        The Elastic-Net mixing parameter, with ``0 <= l1_ratio <= 1``. Only
        used if ``penalty='elasticnet'``. Setting ``l1_ratio=0`` is equivalent
        to using ``penalty='l2'``, while setting ``l1_ratio=1`` is equivalent
        to using ``penalty='l1'``. For ``0 < l1_ratio <1``, the penalty is a
        combination of L1 and L2.
    
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes, )
        A list of class labels known to the classifier.
    coef_ : ndarray of shape (1, n_features) or (n_classes, n_features)
        Coefficient of the features in the decision function.
        `coef_` is of shape (1, n_features) when the given problem is binary.
        In particular, when `multi_class='multinomial'`, `coef_` corresponds
        to outcome 1 (True) and `-coef_` corresponds to outcome 0 (False).
    intercept_ : ndarray of shape (1,) or (n_classes,)
        Intercept (a.k.a. bias) added to the decision function.
        If `fit_intercept` is set to False, the intercept is set to zero.
        `intercept_` is of shape (1,) when the given problem is binary.
        In particular, when `multi_class='multinomial'`, `intercept_`
        corresponds to outcome 1 (True) and `-intercept_` corresponds to
        outcome 0 (False).
    n_features_in_ : int
        Number of features seen during.
        .. versionadded:: 0.24
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during. Defined only when `X`
        has feature names that are all strings.
        .. versionadded:: 1.0
    n_iter_ : ndarray of shape (n_classes,) or (1, )
        Actual number of iterations for all classes. If binary or multinomial,
        it returns only 1 element. For liblinear solver, only the maximum
        number of iteration across all classes is given.
        .. versionchanged:: 0.20
            In SciPy <= 1.0.0 the number of lbfgs iterations may exceed
            ``max_iter``. ``n_iter_`` will now report at most ``max_iter``.
    
    See Also
    --------
    SGDClassifier : Incrementally trained logistic regression (when given
        the parameter ``loss="log"``).
    LogisticRegressionCV : Logistic regression with built-in cross validation.
    
    Notes
    -----
    The underlying C implementation uses a random number generator to
    select features when fitting the model. It is thus not uncommon,
    to have slightly different results for the same input data. If
    that happens, try with a smaller tol parameter.
    Predict output may not match that of standalone liblinear in certain
    cases. 
 
    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = LogisticRegression(random_state=0).fit(X, y)
    >>> clf.predict(X[:2, :])
    array([0, 0])
    >>> clf.predict_proba(X[:2, :])
    array([[9.8...e-01, 1.8...e-02, 1.4...e-08],
           [9.7...e-01, 2.8...e-02, ...e-08]])
    >>> clf.score(X, y)
    0.97...
    """
      


    def _init(self, test):
           self.test = test
     