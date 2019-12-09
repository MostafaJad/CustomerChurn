#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
#import plotly.graph_objs as go
import plotly.graph_objects as go
import plotly.offline as py
import plotly.express as px
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from IPython.display import display
from graphviz import Source

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')


def import_data():
    balance_data = pd.read_csv('/Users/jado/Desktop/data.nosync/Telco-Customer-Churn.csv')

    # Printing the dataswet shape
    print("Data set Length: ", len(balance_data))
    print("Data set Shape: ", balance_data.shape)

    # Printing the data set obseravtions
    print("Dataset: ", balance_data.head())
    return balance_data


def data_manipulation(telcom):
    telcom['TotalCharges'] = telcom["TotalCharges"].replace(" ", np.nan)

    # Dropping null values from total charges column which contain .15% missing data
    telcom = telcom[telcom["TotalCharges"].notnull()]
    telcom = telcom.reset_index()[telcom.columns]

    # convert to float type
    telcom["TotalCharges"] = telcom["TotalCharges"].astype(float)

    # replace 'No internet service' to No for the following columns
    replace_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    for i in replace_cols:
        telcom[i] = telcom[i].replace({'No internet service': 'No'})

    # replace values
    telcom["SeniorCitizen"] = telcom["SeniorCitizen"].replace({1: "Yes", 0: "No"})

    # Tenure to categorical column
    def tenure_lab(telcom):

        if telcom["tenure"] <= 12:
            return "Tenure_0-12"
        elif (telcom["tenure"] > 12) & (telcom["tenure"] <= 24):
            return "Tenure_12-24"
        elif (telcom["tenure"] > 24) & (telcom["tenure"] <= 48):
            return "Tenure_24-48"
        elif (telcom["tenure"] > 48) & (telcom["tenure"] <= 60):
            return "Tenure_48-60"
        elif telcom["tenure"] > 60:
            return "Tenure_gt_60"

    telcom["tenure_group"] = telcom.apply(lambda telcom: tenure_lab(telcom),
                                          axis=1)

    # Separating churn and non churn customers
    churn = telcom[telcom["Churn"] == "Yes"]
    not_churn = telcom[telcom["Churn"] == "No"]
    return telcom


def data_processing(telcom):
    # customer id col
    Id_col = ['customerID']
    # Target columns
    target_col = ["Churn"]
    # categorical columns
    cat_cols = telcom.nunique()[telcom.nunique() < 6].keys().tolist()
    cat_cols = [x for x in cat_cols if x not in target_col]
    # numerical columns
    num_cols = [x for x in telcom.columns if x not in cat_cols + target_col + Id_col]
    # Binary columns with 2 values
    bin_cols = telcom.nunique()[telcom.nunique() == 2].keys().tolist()
    # Columns more than 2 values
    multi_cols = [i for i in cat_cols if i not in bin_cols]

    # Label encoding Binary columns
    le = LabelEncoder()
    for i in bin_cols:
        telcom[i] = le.fit_transform(telcom[i])

    # Duplicating columns for multi value columns
    telcom = pd.get_dummies(data=telcom, columns=multi_cols)

    # Scaling Numerical columns
    std = StandardScaler()
    scaled = std.fit_transform(telcom[num_cols])
    scaled = pd.DataFrame(scaled, columns=num_cols)

    # dropping original values merging scaled values for numerical columns
    df_telcom_og = telcom.copy()
    telcom = telcom.drop(columns=num_cols, axis=1)
    telcom = telcom.merge(scaled, left_index=True, right_index=True, how="left")
    return telcom, df_telcom_og


def customer_attrition(telcom):
    lab = telcom["Churn"].value_counts().keys().tolist()
    # values
    val = telcom["Churn"].value_counts().values.tolist()

    trace = go.Pie(labels=lab,
                   values=val,
                   marker=dict(colors=['royalblue', 'lime'],
                               line=dict(color="white",
                                         width=1.3)
                               ),
                   rotation=90,
                   hoverinfo="label+value+text",
                   hole=.5
                   )
    layout = go.Layout(dict(title="Customer attrition in data",
                            plot_bgcolor="rgb(243,243,243)",
                            paper_bgcolor="rgb(243,243,243)",
                            )
                       )

    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig)


# function  for pie plot for customer attrition types
def plot_pie(column, telcom):
    # Separating churn and non churn customers
    churn = telcom[telcom["Churn"] == "Yes"]
    not_churn = telcom[telcom["Churn"] == "No"]
    trace1 = go.Pie(values=churn[column].value_counts().values.tolist(),
                    labels=churn[column].value_counts().keys().tolist(),
                    hoverinfo="label+percent+name",
                    domain=dict(x=[0, .48]),
                    name="Churn Customers",
                    marker=dict(line=dict(width=2,
                                          color="rgb(243,243,243)")
                                ),
                    hole=.6
                    )
    trace2 = go.Pie(values=not_churn[column].value_counts().values.tolist(),
                    labels=not_churn[column].value_counts().keys().tolist(),
                    hoverinfo="label+percent+name",
                    marker=dict(line=dict(width=2,
                                          color="rgb(243,243,243)")
                                ),
                    domain=dict(x=[.52, 1]),
                    hole=.6,
                    name="Non churn customers"
                    )

    layout = go.Layout(dict(title=column + " distribution in customer attrition ",
                            plot_bgcolor="rgb(243,243,243)",
                            paper_bgcolor="rgb(243,243,243)",
                            annotations=[dict(text="churn customers",
                                              font=dict(size=13),
                                              showarrow=False,
                                              x=.15, y=.5),
                                         dict(text="Non churn customers",
                                              font=dict(size=13),
                                              showarrow=False,
                                              x=.88, y=.5
                                              )
                                         ]
                            )
                       )
    data = [trace1, trace2]
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig)


# function  for histogram for customer attrition types
def histogram(column, telcom):
    # Separating churn and non churn customers
    churn = telcom[telcom["Churn"] == "Yes"]
    not_churn = telcom[telcom["Churn"] == "No"]
    trace1 = go.Histogram(x=churn[column],
                          histnorm="percent",
                          name="Churn Customers",
                          marker=dict(line=dict(width=.5,
                                                color="black"
                                                )
                                      ),
                          opacity=.9
                          )
    x = np.random.randn(500)
    fig = go.Figure(data=[go.Histogram(x=x, histnorm='probability')])
    
    fig.show()
    
    trace2 = go.Histogram(x=not_churn[column],
                          histnorm="percent",
                          name="Non churn customers",
                          marker=dict(line=dict(width=.5,
                                                color="black"
                                                )
                                      ),
                          opacity=.9
                          )

    data = [trace1, trace2]
    layout = go.Layout(dict(title=column + " distribution in customer attrition ",
                            plot_bgcolor="rgb(243,243,243)",
                            paper_bgcolor="rgb(243,243,243)",
                            xaxis=dict(gridcolor='rgb(255, 255, 255)',
                                       title=column,
                                       zerolinewidth=1,
                                       ticklen=5,
                                       gridwidth=2
                                       ),
                            yaxis=dict(gridcolor='rgb(255, 255, 255)',
                                       title="percent",
                                       zerolinewidth=1,
                                       ticklen=5,
                                       gridwidth=2
                                       ),
                            )
                       )
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig)
    


# function  for scatter plot matrix  for numerical columns in data
def scatter_matrix(df):
    df = df.sort_values(by="Churn", ascending=True)
    classes = df["Churn"].unique().tolist()
    classes

    class_code = {classes[k]: k for k in range(2)}
    class_code

    color_vals = [class_code[cl] for cl in df["Churn"]]
    color_vals

    pl_colorscale = "Portland"

    pl_colorscale

    text = [df.loc[k, "Churn"] for k in range(len(df))]
    text

    trace = go.Splom(dimensions=[dict(label="tenure",
                                      values=df["tenure"]),
                                 dict(label='MonthlyCharges',
                                      values=df['MonthlyCharges']),
                                 dict(label='TotalCharges',
                                      values=df['TotalCharges'])],
                     text=text,
                     marker=dict(color=color_vals,
                                 colorscale=pl_colorscale,
                                 size=3,
                                 showscale=False,
                                 line=dict(width=.1,
                                           color='rgb(230,230,230)'
                                           )
                                 )
                     )
    axis = dict(showline=True,
                zeroline=False,
                gridcolor="#fff",
                ticklen=4
                )

    layout = go.Layout(dict(title=
                            "Scatter plot matrix for Numerical columns for customer attrition",
                            autosize=False,
                            height=800,
                            width=800,
                            dragmode="select",
                            hovermode="closest",
                            plot_bgcolor='rgba(240,240,240, 0.95)',
                            xaxis1=dict(axis),
                            yaxis1=dict(axis),
                            xaxis2=dict(axis),
                            yaxis2=dict(axis),
                            xaxis3=dict(axis),
                            yaxis3=dict(axis),
                            )
                       )
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig)


def variables(df_telcom_og):
    Id_col = ['customerID']
    summary = (df_telcom_og[[i for i in df_telcom_og.columns if i not in Id_col]].
               describe().transpose().reset_index())

    summary = summary.rename(columns={"index": "feature"})
    summary = np.around(summary, 3)

    val_lst = [summary['feature'], summary['count'],
               summary['mean'], summary['std'],
               summary['min'], summary['25%'],
                summary['50%'], summary['75%'], summary['max']]

    trace = go.Table(header=dict(values=summary.columns.tolist(),
                                 line=dict(color=['#506784']),
                                 fill=dict(color=['#119DFF']),
                                 ),
                     cells=dict(values=val_lst,
                                line=dict(color=['#506784']),
                                fill=dict(color=["lightgrey", '#F5F8FF'])
                                ),
                     columnwidth=[200, 60, 100, 100, 60, 60, 80, 80, 80])
    layout = go.Layout(dict(title="Variable Summary"))
    figure = go.Figure(data=[trace], layout=layout)
    py.plot(figure)


def feature_scores(telcom, df_telcom_og):
    telcom["gender"] = telcom["gender"].replace({1: "Female", 0: "Male"})
    # customer id col
    Id_col = ['customerID']
    # Target columns
    target_col = ["Churn"]
    # categorical columns
    cat_cols = telcom.nunique()[telcom.nunique() < 6].keys().tolist()
    cat_cols = [x for x in cat_cols if x not in target_col]
    # numerical columns
    num_cols = [x for x in telcom.columns if x not in cat_cols + target_col + Id_col]
    # select columns
    cols = [i for i in telcom.columns if i not in Id_col + target_col ]

    # dataframe with non negative values
    df_x = df_telcom_og[cols]
    df_y = df_telcom_og[target_col]

    # fit model with k= 3
    select = SelectKBest(score_func=chi2, k=3)
    fit = select.fit(df_x, df_y)

    # Summerize scores
    print("scores")
    print(fit.scores_)
    print("P - Values")
    print(fit.pvalues_)


def split_data(telcom):
    # splitting train and test data
    train, test = train_test_split(telcom, test_size=.25, random_state=111)
    # customer id col
    Id_col = ['customerID']
    # Target columns
    target_col = ["Churn"]

    #seperating dependent and independent variables
    cols = [i for i in telcom.columns if i not in Id_col + target_col]
    train_X = train[cols]
    train_Y = train[target_col]
    test_X = test[cols]
    test_Y = test[target_col]
    return train_X, train_Y, test_X, test_Y, cols


def logistic_regression(val, algorithm, training_x, testing_x,
                             training_y, testing_y, cols, cf, threshold_plot):
    # model
    algorithm.fit(training_x, training_y.values.ravel())
    predictions = algorithm.predict(testing_x)
    probabilities = algorithm.predict_proba(testing_x)
    # coeffs
    if cf == "coefficients":
        coefficients = pd.DataFrame(algorithm.coef_.ravel())
    elif cf == "features":
        coefficients = pd.DataFrame(algorithm.feature_importances_)

    column_df = pd.DataFrame(cols)
    coef_sumry = (pd.merge(coefficients, column_df, left_index=True,
                           right_index=True, how="left"))
    coef_sumry.columns = ["coefficients", "features"]
    coef_sumry = coef_sumry.sort_values(by="coefficients", ascending=False)

    print(algorithm)
    print("\n Classification report : \n", classification_report(testing_y, predictions))
    print("Accuracy   Score : ", accuracy_score(testing_y, predictions))
    # confusion matrix
    conf_matrix = confusion_matrix(testing_y, predictions)
    # roc_auc_score
    model_roc_auc = roc_auc_score(testing_y, predictions)
    print("Area under curve : ", model_roc_auc, "\n")
    fpr, tpr, thresholds = roc_curve(testing_y, probabilities[:, 1])

    # plot confusion matrix
    trace1 = go.Heatmap(z=conf_matrix,
                        x=["Not churn", "Churn"],
                        y=["Not churn", "Churn"],
                        showscale=False, colorscale="Picnic",
                        name="matrix")

    if val == 0:
        # subplots
        layout = go.Layout(dict(title="Confusion Matrix for Logistic Regression"))
        figure = go.Figure(data=[trace1], layout=layout)
        py.plot(figure)
    else:
        # subplots
        layout = go.Layout(dict(title="Confusion Matrix for Decision Tree"))
        figure = go.Figure(data=[trace1], layout=layout)
        py.plot(figure)

def plot_decision_tree(telcom, df_telcom_og, test_X, test_Y, columns, maximum_depth, criterion_type,
                       split_type, model_performance=None):
    # customer id col
    Id_col = ['customerID']
    # Target columns
    target_col = ["Churn"]
    cols = [i for i in telcom.columns if i not in Id_col + target_col]
    # dataframe with non negative values
    df_x = df_telcom_og[cols]
    df_y = df_telcom_og[target_col]
    # separating dependent and in dependent variables
    dtc_x = df_x[columns]
    dtc_y = df_y[target_col]

    # model
    dt_classifier = DecisionTreeClassifier(max_depth=maximum_depth,
                                           splitter=split_type,
                                           criterion=criterion_type,
                                           )
    dt_classifier.fit(dtc_x, dtc_y)

    # plot decision tree
    graph = Source(tree.export_graphviz(dt_classifier, out_file=None,
                                        rounded=True, proportion=False,
                                        feature_names=columns,
                                        precision=2,
                                        class_names=["Not churn", "Churn"],
                                        filled=True
                                        )
                   )
    display(graph)

    #model performance
    if model_performance == True :
        logistic_regression(1, dt_classifier,
                                 dtc_x, test_X[columns],
                                 dtc_y, test_Y,
                                 columns, "features", threshold_plot = True)


# Driver code
def main():
    """x = np.random.randn(500)
    print(x)
    fig = go.Figure(data=[go.Histogram(x=x, histnorm='probability')])
    print(fig)
    fig.show()
    fig.write_image('figure.png')"""
    # Building Phase
    data = import_data()
    print("\nMissing values :  ", data.isnull().sum().values.sum())
    print("\nUnique values :  \n", data.nunique())
    data = data_manipulation(data)
    customer_attrition(data)
    histogram('gender', data)
    plot_pie('SeniorCitizen', data)
    scatter_matrix(data)
    data, df_telcom_og = data_processing(data)
    variables(data)
    feature_scores(data, df_telcom_og)
    train_X, train_Y, test_X, test_Y, cols = split_data(df_telcom_og)
    logit = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                           intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                           penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
                           verbose=0, warm_start=False)
    logistic_regression(0, logit, train_X, test_X, train_Y, test_Y,
                              cols, "coefficients", threshold_plot=True)
    # customer id col
    Id_col = ['customerID']
    # Target columns
    target_col = ["Churn"]
    # categorical columns
    cat_cols = data.nunique()[data.nunique() < 6].keys().tolist()
    cat_cols = [x for x in cat_cols if x not in target_col]
    # numerical columns
    num_cols = [x for x in data.columns if x not in cat_cols + target_col + Id_col]
    df_x = df_telcom_og[cols]
    df_y = df_telcom_og[target_col]

    # fit model with k= 3
    select = SelectKBest(score_func=chi2, k=3)
    fit = select.fit(df_x, df_y)
    # create dataframe
    score = pd.DataFrame({"features": cols, "scores": fit.scores_, "p_values": fit.pvalues_})
    score = score.sort_values(by="scores", ascending=False)

    # createing new label for categorical and numerical columns
    score["feature_type"] = np.where(score["features"].isin(num_cols), "Numerical", "Categorical")
    # top 3 categorical features
    features_cat = score[score["feature_type"] == "Categorical"]["features"][:3].tolist()

    # top 3 numerical features
    features_num = score[score["feature_type"] == "Numerical"]["features"][:3].tolist()
    plot_decision_tree(data, df_telcom_og, test_X, test_Y, features_num, 3, "gini", "best")
    plot_decision_tree(data, df_telcom_og, test_X, test_Y, features_cat, 3, "entropy", "best",
                       model_performance=True, )


# Calling main function
if __name__ == "__main__":
    main()
