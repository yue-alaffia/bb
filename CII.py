#Evaluation
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression


class single_feature_cr(BaseEstimator, ClassifierMixin):
    def __init__(self, feature_name):
        self.feature_name = feature_name

    def fit(self, X, y):
        X_false = X[y == 0]
        X_true = X[y != 0]
        y_false = y[y == 0]
        y_true = y[y != 0]
        # Probability that a training example where feature_name == True has a finding
        self.p = y_true.sum() / len(y_true)
        X = X[[self.feature_name] + ["amount_charged"]]
        self.linear_reg = LinearRegression()
        self.linear_reg.fit(X_true, y_true)

    def predict(self, X):
        e = self.p * self.linear_clf(X)
        return e


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pathlib
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from seaborn import scatterplot
from matplotlib.pyplot import show


class Ev_estimator(BaseEstimator):
    # %% Define expected value regressor
    def __init__(self, feature_col):
        self.feature_col = feature_col
        self.regs = {}
        self.rev_code_p_findings = None
        self.rev_code_true_count = None
        self.rev_code_all_count = None

    def fit(self, X, y):
        # Only fit the regressor on claim lines with findings
        X = X[[self.feature_col, "amount_charged"]]  # ignore all other columns
        y_true = y[y > 0]
        X_true = X.loc[y_true.index]

        # Compute p
        self.rev_code_all_count = X.groupby(self.feature_col).count()["amount_charged"]
        self.rev_code_true_count = pd.Series(
            data=X[y > 0].groupby(self.feature_col).count()["amount_charged"],
            index=self.rev_code_all_count.index,
        ).fillna(0)
        self.rev_code_p_findings = (
            self.rev_code_true_count / self.rev_code_all_count
        ).fillna(0)

        # Compute regressions
        groups = X_true.groupby(self.feature_col).groups
        for key in groups.keys():
            reg = LinearRegression()
            reg.fit(
                X_true.loc[groups[key], ["amount_charged"]], y_true.loc[groups[key]]
            )
            self.regs[key] = reg

    def predict_line_values(self, X):
        """
        X: a dataframe containing columns [["amount_charged", feature_col]], one claim line per df line
        returns: a dataframe containing the expected value of each line
        """
        X = X[[self.feature_col, "amount_charged"]]  # ignore all other columns
        groups = X.groupby(self.feature_col).groups

        output = pd.Series(index=X.index, dtype="float64")
        for cat, index in groups.items():
            if cat in self.regs.keys():
                reg = self.regs[cat]
                output.loc[index] = (
                    reg.predict(X.loc[index, ["amount_charged"]])
                    * self.rev_code_p_findings.loc[cat]
                )
            else:
                output.loc[index] = np.zeros_like(
                    index
                )  # Return 0 expected value for out-of-dist examples
        return output.astype(float)


def visualize_prediction_ev(ev_estimator, X, y_ground):
    """Plot prediction EV against ground truth EV"""
    cat = X[feature_col]
    y_pred = ev_estimator.predict_line_values(X)
    df = (
        pd.DataFrame(data=[y_ground, y_pred, cat], index=["y_ground", "y_pred", "cat"],)
        .transpose()
        .infer_objects()
    )
    groups = df.groupby("cat")

    props = pd.DataFrame(
        data=[
            groups["y_ground"].mean(),
            groups["y_pred"].mean(),
            ev_estimator.rev_code_p_findings,
            ev_estimator.rev_code_true_count,
            ev_estimator.rev_code_all_count,
        ],
        # index=groups.groups.keys(),
        # columns=["ground mean", "pred mean", "count",],
        columns=groups.groups.keys(),
        index=["y_ground_mean", "y_pred_mean", "p", "true_count", "all_count",],
    ).transpose()
    # If we want to reduce uncertainty we can filter out rev codes with the following code
    # However, this doesn't really have much effect anywhere, since these data by definition
    # comprise a fairly small portion of all our data

    # props = props[
    #    props["all_count"] >= MIN_EXAMPLES
    # ]  # Filter for minimum number of examples

    # Visualization
    xmin, xmax, ymin, ymax = (10e-2, 10e4, 10e-2, 10e4)
    f, ax = plt.subplots(figsize=(7, 7))
    ax.set(xscale="log", yscale="log")
    scatterplot(
        data=props,
        x="y_ground_mean",
        y="y_pred_mean",
        s=props["all_count"],
        markers="o",
        alpha=0.2,
    ).set(xlim=[xmin, xmax], ylim=[ymin, ymax])

    # Draw a y=x line
    plt.plot(
        np.linspace(xmin, xmax, 1000),
        np.linspace(ymin, ymax, 1000),
        "r-",
        color=(0.8, 0.8, 0.8),
    )
    # Add labels to points on figure
    # for x, y, name in zip(props["y_ground_mean"], props["y_pred_mean"], props.index):
    #     ax.text(x, y, name)
    show()

    # r^2 value
    print("r2:          ", r2_score(props["y_ground_mean"], props["y_pred_mean"]))
    print(
        "r2 weighted: ",
        r2_score(
            props["y_ground_mean"],
            props["y_pred_mean"],
            sample_weight=props["all_count"],
        ),
    )


# %%
IN_EXAMPLES = 10

# %% Get Data
# TODO: Generalize paths
csv_path = audit_findings_df = pd.read_csv('audit_findings.csv')
audit_findings_df = csv_path


# %%
y_cols = [
    "improper_payment_cost",
    "improper_payment_units_charged",
]
num_cols = [
    "units",
    "unit_allowable_charge",
    "unit_charge",
    "amount_charged",
    "amount_reimbursed",
]
cat_cols = ["rev_code", "procedure_code", "adm_type", "claim_type"]
diag_cols = ["diag_1", "diag_2", "diag_3", "diag_4"]

# %% Create train/validation/test sets for finding prediction
y = audit_findings_df["improper_payment_cost"]
X = audit_findings_df.drop("improper_payment_cost", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=1
)  # 0.25 x 0.8 = 0.2

feature_col = "rev_description"
ev_estimator = Ev_estimator(feature_col)
ev_estimator.fit(X_train, y_train)
predictions = ev_estimator.predict_line_values(X_val)

visualize_prediction_ev(ev_estimator, X_train, y_train)


# %%
from scipy import stats
from scipy.stats import norm
import math


# %%
def get_expected_and_actual_value(df, predict_line_values):
    """
    Returns a tuple containing the actual and the expected value of the 
    top `num_selections` predicted claim lines in df. 
    Parameters
    __________
    df : Pandas Dataframe 
        df containing columns necessary for `predict_fn` to make predictions
    
    num_selections : int
        The number of claim lines that will be used in expected/actual value 
        calculations. num_selections <= len(df)
    predict_fn : a function which predicts expected value given df
    """
    ########################
    # Yue's code goes here #
    ########################
    y = df["improper_payment_cost"]
    X = df.drop("improper_payment_cost", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    X_train2, X_val2, y_train2, y_val2 = train_test_split(
        X_train, y_train, test_size=0.25, random_state=1
    )  # 0.25 x 0.8 = 0.2

    cat = X[feature_col]
    y_pred = predict_line_values(X_train2)        # Use defined function to predict the expected value.
    df = pd.DataFrame(data=[y_train2, y_pred, cat], index=['y_ground', 'y_pred', 'cat']).transpose()
    df["y_ground"] = pd.to_numeric(df["y_ground"])
    return df


# %%
### Profits part 


# %%
def profit2(df, num_samples, num_size, num_top_expected_value):
    """
    Returns a graph containing the distribution of sum of the expected values 
    based on the samples, it also returns the confidence intervals 
    when the confidence level is 95%.
    _____________
    df : Pandas Dataframe 
        df containing columns about the expected value, predict claim value and the category.
    
    num_samples : int
        The number of times that will be used in randomly selecting rows to make a graph.
    num_size : int
        How many rows it would select everytime in order to sum up the expect values together.
    num_top_expected_value : int
        The number of claim lines that will be used in expected/actual value 
        calculations. num_top_expected_value <= len(df)
    """
    point_estimates = []        
    for x in range(num_samples):
        df = df.sample(n= num_size)                     # Random select rows 
        df2 = df.nlargest(num_top_expected_value, 'y_ground') 
        sample = np.random.choice(a= df2['y_ground'], size=num_top_expected_value)
        point_estimates.append(sample.sum())            # Sum up largest n expected values
        
    pd.DataFrame(point_estimates).plot(kind="density",  # Plot sample mean density
                                   figsize=(6,6)
                                   )  
    
    sample_size = num_size
    sample = np.random.choice(a= point_estimates, size = num_samples)
    sample_mean = sample.mean()

    z_critical = stats.norm.ppf(q = 0.95)  # Get the z-critical value*

    print("z-critical value:")              # Check the z-critical value
    print(z_critical)  
    point_estimates2 = np.asarray(point_estimates)

    pop_stdev = point_estimates2.std()  # Get the population standard deviation

    margin_of_error = z_critical * (pop_stdev/math.sqrt(sample_size))

    confidence_interval = (sample_mean - margin_of_error,
                           sample_mean + margin_of_error)  

    print("Confidence interval:")
    print(confidence_interval)
    return 


# %%
table = get_expected_and_actual_value(audit_findings_df, ev_estimator.predict_line_values)


# %%
profit2(table, 1000, 10000, 150)


# %%



