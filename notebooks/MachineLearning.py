import numpy as np
import sklearn
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


def splitting_tr_te(df, size=0.95):
    
    split_point = int(df.shape[0]*size)

    train = df.iloc[:split_point]
    test = df.iloc[split_point:]
    return train, test


def unshuffled_CV(X, y, list_of_models, model_names, folds=5):
    results = {}
    cv = KFold(n_splits=folds, shuffle=False)
    
    for model in range(len(list_of_models)):
        metrics = {}
        MSE = []
        MAE = []
        
        for train, test in cv.split(X):
            list_of_models[model].fit(X.iloc[train], y.iloc[train])
            preds = list_of_models[model].predict(X.iloc[test])
            
            MSE.append(mean_squared_error(y.iloc[test], preds))
            MAE.append(mean_absolute_error(y.iloc[test], preds))
            
        metrics["MSE"] = np.mean(MSE)
        metrics["MAE"] = np.mean(MAE)
        
        results[model_names[model]] = metrics
    
    return results


def hyperparameter_tuning(X_train, X_test, y_train, y_test, function, param_grid, scoring, cv):
    mod_cv = sklearn.model_selection.RandomizedSearchCV(function, n_iter=50, param_distributions=param_grid, scoring=scoring, cv=cv)
    
    mod_cv.fit(X_train, y_train)
    best_param = mod_cv.best_params_
    best_train_score = mod_cv.best_score_
    
    y_preds = mod_cv.predict(X_test)
    
    score_dict = {"Best Hyperparameters": best_param,
                  "Best training score": best_train_score,
                  "MSE": mean_squared_error(y_test, y_preds),
                  "MAE": mean_absolute_error(y_test, y_preds)}
    
    return score_dict, y_preds