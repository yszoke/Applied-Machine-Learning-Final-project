import csv
import os
import warnings

import numpy as np
import pandas as pd
from sklearn import metrics
import glob
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import statistics as st
from sklearn.ensemble import AdaBoostClassifier
from bin.ensemble import ensemble_main, create_model, fit_method
from toupee.data import one_hot_numpy
import optuna
import toupee as tp
import tensorflow as tf

# constants
PARAMS_FILE = r"C:\Users\ortal\PycharmProjects\Applied-Machine-Learning-Final-project\examples\experiments\cifar-100\adaboost.yaml"
DATA_PATH = r'C:\Users\ortal\PycharmProjects\Applied-Machine-Learning-Final-project\datasets\cifar100'
DATA_SIZE = 700
OUT_DATA_PATH = r'C:\Users\ortal\PycharmProjects\Applied-Machine-Learning-Final-project\datasets\cifar100\fold'


def create_models():
    model_arr = []
    dib_model = create_model(PARAMS_FILE) # give parameters
    params_dib_model = {}
    model_arr.append(('DIB', (dib_model, params_dib_model)))

    clf = AdaBoostClassifier()
    param_clf = {
        'n_estimators': [100]
    }
    model_arr.append(('AdaBoost', (clf, param_clf)))

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


def main_regular_models():
    global data
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


def _load_npz(filename: str, convert_labels_to_one_hot=False) -> tuple:
    """ Load an NPZ file """
    data1 = np.load(filename)
    data = (data1['x'],data1['y'])
    if convert_labels_to_one_hot:
        data = (data[0], one_hot_numpy(data[1]))
    return data


def main_npz_files():
    algorithm_name = "adaboost"
    # load all data
    test_data = _load_npz(DATA_PATH+r'\test.npz')
    train_data = _load_npz(DATA_PATH + r'\train.npz')
    valid_data = _load_npz(DATA_PATH + r'\valid.npz')

    # combine the data into one numpy array
    X = np.concatenate((test_data[0], train_data[0], valid_data[0]))
    y = np.concatenate((test_data[1], train_data[1], valid_data[1]))

    # divide

    # randomly select part of the data to reduce runtime
    # idx = np.random.choice(X.shape[0], DATA_SIZE, replace=False)
    # X = X[idx]
    # y = y[idx]

    # divide data into 20 datasets
    X_shape = X.shape[0]
    array_size = int(X_shape/20)
    for i in range(0, X_shape, array_size):
        start = i
        stop = i + array_size

        X = X[start:stop]
        y = y[start:stop]

        # randomly select test data -> 20% test
        idx = np.random.choice(X.shape[0], int(DATA_SIZE*0.2), replace=False)
        X_test = X[idx]
        y_test = y[idx]
        X = np.delete(X, idx, axis=0)
        y = np.delete(y, idx, axis=0)

        # create loops with cv
        seed = 7
        k = 10

        cv = KFold(n_splits=k, shuffle=True, random_state=seed)

        results = pd.DataFrame(columns=['acc','micro_precision','micro_recall','micro_f1','macro_precision','macro_recall','macro_f1','TPR','FPR','AUC','PR_Curve','time'])

        for train_ix, test_ix in cv.split(X):
            try:
                # split data
                X_train, X_validation = X[train_ix], X[test_ix]
                y_train, y_validation = y[train_ix], y[test_ix]

                # create folders with this data for the run
                np.savez(OUT_DATA_PATH+r'\test.npz', x=X_test, y=y_test)
                np.savez(OUT_DATA_PATH + r'\train.npz', x=X_train, y=y_train)
                np.savez(OUT_DATA_PATH + r'\valid.npz', x=X_validation, y=y_validation)

                # optuna -> current validation train -> best model -> metrics of the cross validation
                study = optuna.create_study(direction="maximize")
                study.optimize(objective_new, n_trials=2)
                print("Number of finished trials: ", len(study.trials))
                print("Best trial:")
                trial = study.best_trial
                print("  Value: ", trial.value)
                print("  Params: ")
                for key, value in trial.params.items():
                    print("    {}: {}".format(key, value))

                # run best model
                size = trial.params['ensemble_size']
                optimizer_name = trial.params['optimizer']

                params = tp.config.load_parameters(PARAMS_FILE)
                params.ensemble_method['params']['size'] = size
                params.optimizer[0]['class_name'] = optimizer_name

                # Training and validating cycle.
                model, params = create_model(params=params)
                cv_metrics = fit_method(method=model, params=params)

                ensemble_metrics = cv_metrics['ensemble']

                # write to csv
                results.append({'acc': ensemble_metrics['accuracy_score'],
                                'micro_precision': ensemble_metrics['micro_precision_score'],
                                'micro_recall': ensemble_metrics['micro_recall_score'],
                                'micro_f1': ensemble_metrics['micro_f1_score'],
                                'macro_precision':ensemble_metrics['macro_precision_score'],
                                'macro_recall':ensemble_metrics['macro_recall_score'],
                                'macro_f1':ensemble_metrics['macro_f1_score'],
                                'TPR':ensemble_metrics['TPR'],
                                'FPR':ensemble_metrics['FPR'],
                                'AUC':ensemble_metrics['AUC'],
                                'PR_Curve':ensemble_metrics['PR_Curve'],
                                'time':cv_metrics['time']}, ignore_index=True)
            except Exception as e:
                # swallow errors
                # write to csv
                results.append({'acc': 0,
                                'micro_precision': 0,
                                'micro_recall': 0,
                                'micro_f1': 0,
                                'macro_precision': 0,
                                'macro_recall': 0,
                                'macro_f1': 0,
                                'TPR': 0,
                                'FPR': 0,
                                'AUC': 0,
                                'PR_Curve': 0,
                                'time': 0}, ignore_index=True)
                print(e.__traceback__)
                print(e.__cause__)
                print(e.__str__())
                print(e)
                continue

        dataset_num = int(X_shape/i)
        results.to_csv(f'results\\result_{algorithm_name}_{dataset_num}.csv')

        # predict with best model and get results+weights ??


def size_trial(trial):
    # We optimize the numbers of layers, their units and weight decay parameter.
    return trial.suggest_int("ensemble_size", 2, 6, step=2)


def optimizer_trial(trial):
    optimizer_options = ["Adam", "SGD"]
    return trial.suggest_categorical("optimizer", optimizer_options)


def objective_new(trial):
    # Build model and optimizer.
    size = size_trial(trial)
    optimizer_name = optimizer_trial(trial)

    params = tp.config.load_parameters(PARAMS_FILE)
    params.ensemble_method['params']['size'] = size
    params.optimizer[0]['class_name'] = optimizer_name

    # Training and validating cycle.
    model, params = create_model(params=params)
    cv_metrics = fit_method(method=model, params=params)

    return cv_metrics['ensemble']['accuracy_score']


def objective(trial):
    # load all data
    test_data = _load_npz(DATA_PATH + r'\test.npz')
    train_data = _load_npz(DATA_PATH + r'\train.npz')
    valid_data = _load_npz(DATA_PATH + r'\valid.npz')

    # combine the data into one numpy array
    X = np.concatenate((test_data[0], train_data[0], valid_data[0]))
    y = np.concatenate((test_data[1], train_data[1], valid_data[1]))

    # randomly select part of the data to reduce runtime
    idx = np.random.choice(X.shape[0], DATA_SIZE, replace=False)
    X = X[idx]
    y = y[idx]

    # randomly select test data -> 20% test
    idx = np.random.choice(X.shape[0], int(DATA_SIZE * 0.2), replace=False)
    X_test = X[idx]
    y_test = y[idx]
    X = np.delete(X, idx, axis=0)
    y = np.delete(y, idx, axis=0)

    # create loops with cv
    seed = 7
    k_inner = 3
    k_outer = 10

    inner_cv = KFold(n_splits=k_inner, shuffle=True, random_state=seed)
    outer_cv = KFold(n_splits=k_outer, shuffle=True, random_state=seed)

    for train_ix, test_ix in outer_cv.split(X):
        # split data
        X_train, X_validation = X[train_ix], X[test_ix]
        y_train, y_validation = y[train_ix], y[test_ix]

        # create folders with this data for the run
        np.savez(OUT_DATA_PATH + r'\test.npz', x=X_test, y=y_test)
        np.savez(OUT_DATA_PATH + r'\train.npz', x=X_train, y=y_train)
        np.savez(OUT_DATA_PATH + r'\valid.npz', x=X_validation, y=y_validation)

        # Build model and optimizer.
        size = size_trial(trial)
        optimizer_name = optimizer_trial(trial)

        params = tp.config.load_parameters(PARAMS_FILE)
        params.ensemble_method['params']['size'] = size
        params.optimizer[0]['class_name'] = optimizer_name

        # Training and validating cycle.
        model, params = create_model(params=params)
        cv_metrics = fit_method(method=model, params=params)

        accuracy = cv_metrics['ensemble']['accuracy_score']

        break

    # Return last validation accuracy.
    return accuracy


def optuna_main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=2)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == '__main__':
    with warnings.catch_warnings():
        # ignore all caught warnings
        warnings.filterwarnings("ignore")
        # execute code that will generate warnings

        #optuna_main()

        main_npz_files()

        #ensemble_main(params_file=PARAMS_FILE)

        #main_regular_models()


