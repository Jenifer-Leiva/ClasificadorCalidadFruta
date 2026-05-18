from libs import os, cv2, np, plt, sns, pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score,  KFold, cross_val_predict
from sklearn.metrics import (accuracy_score, recall_score, confusion_matrix, roc_auc_score, precision_score)
from sklearn.model_selection import GridSearchCV

#X -> Explanatory variables
#Y -> Target Variable

def SVM_Train_Manual (X_train, Y_train, X_val, Y_val):

    #Diccionario para probar con diferentes hiperparametros 
    scores = {}

    kernels = ['linear', 'rbf','poly']
    #np.logspace(-3, 2, num=6) # Para C y Gamma [0.001, 0.01, 0.1, 1, 10, 100]

    for kernel_ in kernels:
        for gamma_ in np.logspace(-3, 2, num=6):
            for C_ in np.logspace(-3, 2, num=6):
                #Modelo
                svm = SVC(kernel = kernel_, gamma = gamma_, C=C_, probability=True)
                svm.fit(X_train, Y_train) #Se ajusta a entrenamiento
                scores[(kernel_, gamma_, C_)] = svm.score(X_val, Y_val) #Puntuación obtenida para hiperparametros sobre set de validacion

    scores = pd.Series(scores)

    print(f'Best score: {scores.max():.2f}')
    print(f'Parametros que obtuvieron Best Score: {scores.idxmax()}')

    #K-Fold Manual

    # Perform 5-fold cross-validation
    scores = cross_val_score(tree, cancer.data, cancer.target, cv=5)

    # Display results
    print('Cross validation scores: {}'.format(scores))
    print('Cross validation scores: {:.3f}+-{:.3f}'.format(scores.mean(), scores.std()))



def 



def SVM_Train_GridSearch(X_train, Y_train, X_test, Y_test):

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    param_grid = {
    'kernel':['linear', 'rbf','poly'],
    'gamma': np.logspace(-3, 2, num=6),
    'C': np.logspace(-3, 2, num=6)
    }

    gs = GridSearchCV(
    estimator=SVC(),        # Modelo
    param_grid=param_grid,  # Hyperparameters a probar
    cv=kf                    # split cross-validation (k-fold)
    )   

    gs.fit(X_train, Y_train)

    print('Best cross-validation score: {:.3f}'.format(gs.best_score_))
    print('Best parameters: {}'.format(gs.best_params_))
    print('Test set score: {:.3f}'.format(gs.score(X_test, Y_test)))

def calcular_metricas_binarias(y_true, y_pred, y_prob): #Diferencia entre probabilidad y predicción (probabilidad de que pertenezca a la positiva) (predicción por umbral)
  acc = accuracy_score(y_true, y_pred)
  sens = recall_score(y_true, y_pred) #Sensibilidad (Recall)

  #Especificidad = TN / (TN + FP)
  cm = confusion_matrix(y_true, y_pred)
  tn, fp, fn, tp = cm.ravel()
  spec = tn / (tn + fp)

  auc = roc_auc_score(y_true, y_prob)

  return acc, sens, spec, auc, cm

def TestModel(model, X_test, y_test):
    y_pred = model.predict(X_test)

    calcular_metricas_binarias(y_test, y_pred, )