import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings(action='ignore')


class FeatureReducer:
    def __init__(self, method='pca', no_of_final_features=None, kbest_score_func=None, plots=False, threshold=None):
        self.method = method.lower()
        self.no_of_final_features = no_of_final_features
        self.kbest_score_func = kbest_score_func or 'f_classif'
        self.plots = plots
        self.threshold = threshold or {'missing': 0.3, 
                                         'variance': 0.01, 
                                         'vif': 10, 
                                         'f_values': 10,
                                         'alpha': 0.01}

    class methods:
        @staticmethod
        def available():
            return ['missing', 'variance', 'pca', 'factor_analysis', 'selectkbest', 'vif', 'backward_elimination', 'forward_elimination', 'lasso', 'random_forest']

        def __getattr__(self, name):
            if name == 'available':
                return self.available()
            raise AttributeError(f"'Methods object has no attribute '{name}'")
        
    class thresholds:
        @staticmethod
        def description():
            return {'missing': 'Threshold for missing values in a column.',
                    'variance': 'Threshold for variance values in a column.',
                    'vif': 'Variance Inflation Factor Threshold.',
                    'f_values' : 'Threshold for f_values in Forward Elimination.',
                    'alpha': 'Learning Rate in Lasso Regression.'}

        def __getattr__(self, name):
            if name == 'description':
                return self.description()
            raise AttributeError(f"'Thresholds object has no attribute '{name}'")


    # Perform reduction based on the specified method
    def reduce_features(self, X, y=None):
        """
        Reduce the number of features in the given dataset using the specified method.

        Parameters:
        - X (DataFrame): The input dataset containing features to be reduced.
        - y (Series or None, optional): The target variable for methods that require it. Default is None.

        Returns:
        - X_reduced (DataFrame): The dataset with reduced features based on the chosen method.

        Raises:
        - ValueError: If the specified reduction method is invalid.
        
        Methods supported for reduction:
            - 'missing': Remove features with missing values above a specified threshold.
            - 'variance': Remove features with variance below a specified threshold.
            - 'pca': Reduce features using Principal Component Analysis.
            - 'factor_analysis': Reduce features using Factor Analysis.
            - 'selectkbest': Select top k features using univariate statistical tests.
            - 'vif': Remove features with high Variance Inflation Factor.
            - 'backward_elimination': Perform Backward Feature Elimination using RFE.
            - 'forward_elimination': Perform Forward Feature Elimination using f_regression.
            - 'lasso': Reduce features using Lasso Regression.
            - 'random_forest': Select top features using Random Forest feature importances.
        
        Note:
        - For specific method parameters and additional details, refer to the class documentation.
        """

        if self.method == 'missing':
            return self._filter_missing_values(X)
        elif self.method == 'variance':
            return self._filter_low_variance(X)
        elif self.method == 'pca':
            return self._reduce_with_pca(X)
        elif self.method == 'factor_analysis':
            return self._reduce_with_factor_analysis(X)
        elif self.method == 'selectkbest':
            return self._reduce_with_selectkbest(X, y)
        elif self.method == 'vif':
            return self._reduce_with_vif(X)
        elif self.method == 'backward_elimination':
            return self._backward_elimination(X, y)
        elif self.method == 'forward_elimination':
            return self._forward_elimination(X, y)
        elif self.method == 'lasso':
            return self._reduce_with_lasso(X, y)        
        elif self.method == 'random_forest':
            return self._reduce_with_random_forest(X, y)
        else:
            raise ValueError(f"Invalid reduction method: {self.method}")


    # Missing Value Filter
    def _filter_missing_values(self, X):
        threshold = self.threshold.get('missing', None)
        if threshold is not None:
            columns_to_drop = X.columns[X.isnull().mean() > threshold]
            X_reduced = X.drop(columns=columns_to_drop)
        return X_reduced


    # Low Variance filter
    def _filter_low_variance(self, X):
        threshold = self.threshold.get('variance', None)
        if threshold is not None:
            columns_to_drop = X.columns[X.var() < threshold]
            X_reduced = X.drop(columns=columns_to_drop)
        return X_reduced


    # Principle Component Analysis
    def _reduce_with_pca(self, X):
        from sklearn.decomposition import PCA 

        pca = PCA(n_components=self.no_of_final_features)
        reduced_features = pca.fit_transform(X)   # Returns a numpy array
        X_reduced = pd.DataFrame(reduced_features)

        if self.plots:
            # cov_matrix = pca.get_covariance()
            # eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            eigenvalues = pca.explained_variance_ratio_    # Variance ratios = associated eigenvalues with each component
            cumulative_var_ratio = np.cumsum(eigenvalues)

            plt.style.use('dark_background')
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # Explained Variance Ratio
            sns.barplot(x=np.arange(1, len(eigenvalues) + 1), y=eigenvalues, palette='viridis', alpha=0.75, ax=axes[0])
            axes[0].set_title('Explained Variance Ratio')
            axes[0].set_xlabel('Principal Components')
            axes[0].set_ylabel('Explained Variance Ratio')

            # Scree Plot
            axes[1].plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='--', color='b', alpha=0.75)
            axes[1].set_title('Scree Plot - Eigenvalues')
            axes[1].set_xlabel('Principal Components')
            axes[1].set_ylabel('EigenValues')

            # Cumulative explained variance
            axes[2].plot(np.arange(1, len(eigenvalues) + 1), cumulative_var_ratio, marker='o', linestyle='-', color='r', alpha=0.75)
            axes[2].set_title('Cumulative Explained Variance')
            axes[2].set_xlabel('Principal Components')
            axes[2].set_ylabel('Cumulative Explained Variance')

            plt.tight_layout()
            plt.show()

        return X_reduced


    # Factor Analysis
    def _reduce_with_factor_analysis(self, X):
        from sklearn.decomposition import FactorAnalysis

        fa = FactorAnalysis(n_components=self.no_of_final_features, random_state=42)
        X_reduced = fa.fit_transform(X)   # Returns a numpy array
        X_reduced = pd.DataFrame(X_reduced)
        X_reduced = X_reduced.loc[:, (X_reduced != 0).any(axis=0)]  # Remove columns with all 0s (if any)
        
        if self.plots:
            cov_matrix = fa.get_covariance()   
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

            plt.style.use('dark_background')
            plt.figure(figsize=(10, 7))
            plt.plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='--', color='b', alpha=0.75)
            plt.title('Scree Plot - Eigenvalues')
            plt.xlabel('Factors')
            plt.ylabel('Eigenvalues')
            plt.tight_layout()
            plt.show()

        return X_reduced
    

    # Select K-Best
    def _reduce_with_selectkbest(self, X, y):
        from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression, mutual_info_classif, f_classif, chi2 
        
        n = self.no_of_final_features or len(X.columns)-1
        
        if self.kbest_score_func == 'mutual_info_classif':
            kbest = SelectKBest(score_func=mutual_info_classif, k=n)
        elif self.kbest_score_func == 'mutual_info_regression':
            kbest = SelectKBest(score_func=mutual_info_regression, k=n)
        elif self.kbest_score_func == 'f_classif':
            kbest = SelectKBest(score_func=f_classif, k=n)
        elif self.kbest_score_func == 'f_regression':
            kbest = SelectKBest(score_func=f_regression, k=n)
        elif self.kbest_score_func == 'chi2':
            kbest = SelectKBest(score_func=chi2, k=n)

        kbest.fit_transform(X, y)
        columns_to_keep = X.columns[kbest.get_support()]

        return X[columns_to_keep]


    # Variance inflation factor
    def _reduce_with_vif(self, X):
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        try:
            while True:
                VIF_data = pd.DataFrame({'features': X.columns,
                                        'vif': [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]})

                max_vif_index = VIF_data['vif'].idxmax()

                VIF_max = VIF_data.iloc[max_vif_index, :]   # Returns a series

                if (self.no_of_final_features is not None and len(VIF_data['features']) > self.no_of_final_features) and (VIF_max['vif'] > 10):
                    X.drop(columns=VIF_max['features'], inplace=True)  
                else:
                    break
            return X

        except Exception as e:
            print(f"Error in VIF calculation: {e}")


    # Backward Feature Elimination using RFE
    def _backward_elimination(self, X, y):
        from sklearn.linear_model import LinearRegression
        from sklearn.feature_selection import RFE

        estimator = LinearRegression()
        rfe = RFE(estimator, n_features_to_select=self.no_of_final_features)
        rfe.fit(X, y)
        X_reduced = rfe.transform(X)   # Returns a numpy array

        selected_feature_names = X.columns[rfe.support_]   # rfe.support_ returns a binary array
        X_reduced = pd.DataFrame(X_reduced, columns=selected_feature_names)

        return X_reduced


    # Forward Feature Elimination using f_regression
    def _forward_elimination(self, X, y):
        from sklearn.feature_selection import f_regression

        f_values, p_values = f_regression(X, y)

        mask = f_values > self.threshold['f_values']  # Boolean mask for features that meet the threshold

        # Limit the number of selected features to no_of_final_features
        if self.no_of_final_features is not None:
            selected_indices = np.where(mask)[0][:self.no_of_final_features]
        else:
            selected_indices = np.where(mask)[0]
        
        columns_to_keep = X.columns[selected_indices]  # Filter columns based on the selected indices

        return X[columns_to_keep]
    

    # Lasso Regression
    def _reduce_with_lasso(self, X, y):
        from sklearn.linear_model import Lasso
        from sklearn.model_selection import train_test_split

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        lasso = Lasso(alpha=self.threshold['alpha'])  # Adjust alpha based on the strength of regularization
        lasso.fit(X_train, y_train)

        # Get the selected features (non-zero coefficients)
        columns_to_keep = X_train.columns[lasso.coef_ != 0]

        return X[columns_to_keep]
    

    # Random Forest
    def _reduce_with_random_forest(self, X, y):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
     
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)

        n = len(X.columns)-1 if self.no_of_final_features is None else self.no_of_final_features
        columns_to_keep = X.columns[np.argsort(rf.feature_importances_)[::-1]][:n]

        return X[columns_to_keep]