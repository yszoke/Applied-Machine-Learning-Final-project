import pandas as pd
from sklearn import metrics
import glob
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import statistics as st
from sklearn.ensemble import AdaBoostClassifier
from skopt import BayesSearchCV

def create_models():
    model_arr = []
    clf = AdaBoostClassifier()
    param_lr = {
        'n_estimators': [100]
    }
    model_arr.append(('AdaBoost', (clf, param_lr)))

    lr = LogisticRegression()
    param_lr = {
        'solver': ['lbfgs'],
        'max_iter': [500, 1000]
    }
    model_arr.append(('LR', (lr, param_lr)))
    return model_arr


def model_optimization(X, y, model_arr):
    results = []
    names = []
    models = []
    seed = 7
    k_inner = 3
    # change k_outer to 10
    k_outer = 2

    X = X.to_numpy()
    y = y.to_numpy()

    inner_cv = KFold(n_splits=k_inner, shuffle=True, random_state=seed)
    outer_cv = KFold(n_splits=k_outer, shuffle=True, random_state=seed)

    # create models to run
    for name, model in model_arr:
        models.append((name, GridSearchCV(model[0], model[1], cv=inner_cv, n_jobs=1))) # BayesSearchCV or RandomizedSearchCV

    chosen_models = {}
    for name, model in models:
        print("------")
        print(name)
        print("------")
        best_model_f1 = (None, 0)
        results = []
        for train_ix, test_ix in outer_cv.split(X):

            # split data
            X_train, X_test = X[train_ix, :], X[test_ix, :]
            y_train, y_test = y[train_ix], y[test_ix]
            X_train, X_test, y_train, y_test = pd.DataFrame(X_train), pd.DataFrame(X_test), pd.DataFrame(
                y_train), pd.DataFrame(y_test),

            X_train, y_train = pd.DataFrame(X_train), pd.DataFrame(y_train)

            # execute search
            result = model.fit(X_train, y_train.values.ravel())

            # get the best performing model fit on the whole training set
            best_model = result.best_estimator_

            # evaluate model on the hold out dataset
            yhat = best_model.predict(X_test)

            # evaluate the model
            acc = accuracy_score(y_test, yhat)

            f1 = f1_score(y_test, yhat, average='micro')
            if f1 > best_model_f1[1]:
                # the current f1 is better than the maximum found
                best_model_f1 = (best_model, f1)

            # report progress
            print(f'acc={acc}, f1={f1}, est={result.best_score_}, cfg={result.best_params_}')

        # print the best model results
        print(f'Best f1 score found for {name}: {f1}')
        chosen_models[name] = best_model_f1[0]
    return chosen_models

def train_and_predict(X, y, chosen_models):
    seed = 7
    k = 10

    cv = KFold(n_splits=k, shuffle=True, random_state=seed)

    X = X.to_numpy()
    y = y.to_numpy()

    for name, model in chosen_models.items():
        results = {'f1 score': [], 'accuracy': [], 'recall': [], 'precision': [], 'AUROC': []}
        for train_ix, test_ix in cv.split(X):
            # split data
            X_train, X_test = X[train_ix, :], X[test_ix, :]
            y_train, y_test = y[train_ix], y[test_ix]
            X_train, X_test, y_train, y_test = pd.DataFrame(X_train), pd.DataFrame(X_test), pd.DataFrame(
                y_train), pd.DataFrame(y_test),

            X_train, y_train = pd.DataFrame(X_train), pd.DataFrame(y_train)

            # execute train
            result = model.fit(X_train, y_train.values.ravel())

            yhat = model.predict(X_test)

            # evaluate the model
            # results['f1 score'].append(f1_score(y_test, yhat, average='micro'))
            results['accuracy'].append(accuracy_score(y_test, yhat))
            results['recall'].append(recall_score(y_test, yhat, average='micro'))
            results['precision'].append(precision_score(y_test, yhat, average='micro'))
            results['AUROC'].append(roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr'))

        acc = results['accuracy']
        # f1 = results['f1 score']
        recall = results['recall']
        precision = results['precision']
        auroc = results['AUROC']
        # print results
        print("\n")
        print(f'Results for {name}:')
        print(f'acc={st.mean(acc)} +-{st.stdev(acc)}')
        # print(f'f1={st.mean(f1)} +-{st.stdev(f1)}')
        print(f'recall={st.mean(recall)} +-{st.stdev(recall)}')
        print(f'precision={st.mean(precision)} +-{st.stdev(precision)}')
        print(f'AUROC={st.mean(auroc)} +-{st.stdev(auroc)}')

def get_log_loss(y_test, y_pred, label_name):
    y_test_df = y_test.to_frame().pivot_table(index=y_test.index, columns=[label_name], aggfunc=[len], fill_value=0)
    return metrics.log_loss(y_test_df.to_numpy(), y_pred)


if __name__ == '__main__':
    model_arr = []
    model_arr = create_models()

    # change to '../datasets/classification_datasets/*.csv'
    for filepath in glob.iglob('../datasets/classification_datasets/abalon.csv'):
        print(filepath)
        chosen_models = {}
        try:
            print('---------------------------------------------')

            # Read data from csv
            data = pd.read_csv(filepath)
            label = 'class'
            # Split the data
            X = data.loc[:, data.columns != label]
            y = data[label]
            chosen_models = model_optimization(X, y, model_arr)
            results = train_and_predict(X, y, chosen_models)
        except:
            print('Exception:' + filepath)
            continue

        print("\n")
        print('---------------------------------------------')


