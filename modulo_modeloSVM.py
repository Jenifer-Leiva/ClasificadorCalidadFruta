from libs import os, cv2, np, plt, sns, pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score,  KFold, cross_val_predict
from sklearn.metrics import (accuracy_score, recall_score, confusion_matrix, roc_auc_score, precision_score,  roc_curve)
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


    # Initialize and train the model using best hyperparameters
    model = SVC(kernel= scores.idxmax()[0], gamma=scores.idxmax()[1], C=scores.idxmax()[2])
    model.fit(X_train, Y_train)

    #K-Fold Manual

    # Perform 5-fold cross-validation
    #scores = cross_val_score(tree, cancer.data, cancer.target, cv=5)

    # Display results
    print('Cross validation scores: {}'.format(scores))
    print('Cross validation scores: {:.3f}+-{:.3f}'.format(scores.mean(), scores.std()))


def SVM_Train_GridSearch(X_train, Y_train, X_test, Y_test):

    svm = SVC(probability=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    param_grid = {
    'kernel':['linear', 'rbf','poly'],
    'gamma': np.logspace(-3, 2, num=6),
    'C': np.logspace(-3, 2, num=6)
    }

    model_grid = GridSearchCV(
    estimator=svm,        # Modelo
    param_grid=param_grid,  # Hyperparameters a probar
    cv=kf, refit=True                    # split cross-validation (k-fold)
    )   

    model_grid.fit(X_train, Y_train)

    print('Best cross-validation score: {:.3f}'.format(model_grid.best_score_))
    print('Best parameters: {}'.format(model_grid.best_params_))
    print('Test set score: {:.3f}'.format(model_grid.score(X_test, Y_test)))

    #Refit en True no es necesario volver a aplicar fit
    model = model_grid.best_estimator_

    return model

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
    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:, 1]

    acc_t, sens_t, spec_t, auc_t, cm_t = calcular_metricas_binarias(y_test, y_pred_test, y_prob_test)

    #ROC
    # false positive rate and true positive rate
    fpr, tpr, thresholds = roc_curve(y_test, y_prob_test)

    return acc_t, sens_t, spec_t, auc_t, cm_t, fpr, tpr, thresholds


def SVM_Clasificador(X_train, Y_train, X_val, Y_val, X_test, Y_test, target_1, target_2, out_dir):
    target_names =  [target_1, target_2]
    model = SVM_Train_GridSearch(X_train, Y_train, X_test, Y_test)

    acc_t, sens_t, spec_t, auc_t, cm_t, fpr, tpr, thresholds = TestModel(model, X_test, Y_test)

    #Visualizacion de Matrices de confusión

    print(f"Accuracy: {acc_t: .4f} | Sensibilidad: {sens_t: .4f}")
    print(f"Especificidad: {spec_t: .4f} | AUC: {auc_t: .4f}")


    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(cm_t, annot=True, fmt='d', cmap='Greens', ax=ax[1], xticklabels=target_names, yticklabels=target_names)
    ax[0].set_title('Matriz de confusión: Test (30%)')
    ax[0].set_xlabel('Predicho')
    ax[0].set_ylabel('Real')

    plt.tight_layout()
    plt.show()

    path_mc = os.path.join(out_dir, "MatrizConfucionTestSVM")
    plt.savefig(path_mc)

    #ROC Curve

    # Plot the ROC curve
    plt.plot(fpr, tpr, color='red', label='ROC curve (area = %.3f)' % auc_t)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve')
    plt.legend(loc="best")

    path_Roc = os.path.join(out_dir, "ROCCurveSVM")
    plt.savefig(path_Roc)



