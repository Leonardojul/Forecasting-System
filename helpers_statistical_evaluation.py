from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, median_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


def statistical_evaluation(clf : None,
                           y_real : pd.DataFrame,
                           y_predicted: pd.DataFrame,
                           features
                           ):
    """
    Args:
        clf: The model that is used for the analysis
        y_real: pd.DataFrame, containing the real values for the period we made a prediction for.
        y_predicted: pd.DataFrame, containing the predicted values for the period we made a prediction for.
        features: the features we use in the model.

    Returns:
         importances: dict, containg key-value pairs with the feature name and the importance value.
         statistical_overview: dict, overview of the most important statistical scores.

    """
    try:
        feature_importance = list(clf.feature_importances_)
        importances = dict(zip(features, feature_importance))
    except AttributeError:
        importances = 'No Feature Importance Found'
    # feature_importance = list(clf.feature_importances_)
    # features = cols
    score = r2_score(y_real,y_predicted)
    r2_adjusted_numerator = (1 - score) * (len(y_real) - 1)
    r2_adjusted_denominator = len(y_real) - len(features) - 1
    r2_adjusted = 1 - (r2_adjusted_numerator / r2_adjusted_denominator)
    # r2_adjusted = 1 - (1 - score) * ((len(y_real) - 1) / (len(y_real) - len(features) - 1))
    MSE = mean_squared_error(y_real,y_predicted)
    MAE = mean_absolute_error(y_real, y_predicted)
    MAPE = mean_absolute_percentage_error(y_real, y_predicted)
    NRMSE = MSE/(y_real.max()-y_real.min())
    statistical_overview = {"R2_score": score,
                            "Adjusted_R2_score": r2_adjusted,
                            "MSE": MSE,
                            "NRMSE": NRMSE,
                            "MAE": MAE,
                            "MAPE": MAPE}
    return importances, statistical_overview

def analyze(oc_train: pd.DataFrame):
    """
    Decomposes the time series to analize it. Accepts only single-column DataFrames

    Args:
        oc_train (pd.DataFrame): Stands for one-column (+ date) train dataset: The datframe to be analyzed

    Returns:
        Plots series decomposition and prints Dickey-Fuller test results

    Example:

        .. code-block:: python

            import pandas as pd
            test_data = {'Daily_volumes':[10,15,84,32,56,78,94,321,45,97,64,13,64,97,163,6,716,5,6,71,6,4,6]}
            date_generated = pd.date_range(start='1/1/2022',end='1/23/2022')
            test_dataframe = pd.DataFrame(test_data, index=date_generated)
            analyze(test_dataframe)
    """
    result=seasonal_decompose(oc_train, model='multiplicative')
 
    result.plot()

    print(adfuller(oc_train))

    plt.show()