from core.libs import os, cv2, np, plt, sns, pd

from modulo_caracteristicas_forma import extraer_caracteristicas_forma
from modulo_aumentar import augmentar_dataset
from modulo_caracteristicas_color_textura import extraer_caracteristicas_Final, extraer_caracteristicas_Final_Test

from libs import os, cv2, np, plt, sns, pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score,  StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, recall_score, confusion_matrix, roc_auc_score, precision_score,  roc_curve)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

#X -> Explanatory variables
#Y -> Target Variable

def Train_Manual (clasificador, X, Y, target1, target2, first_column):

    auc_scores = []
    acc_scores = []
    sens_scores = []
    spec_scores = []

    if(clasificador == 'svm'):
        model = SVC(kernel= 'rbf', probability=True)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Train and evaluate using Startified K-Fold cross-validation
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, Y)):
        print(f"Fold {fold + 1}")

        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = Y.iloc[train_idx], Y.iloc[valid_idx]

        features_matrix, features_select = extraer_caracteristicas_Final(X_train, y_train, first_column, out_dir="./galeria_resultados/train")

        target_names =  [target1, target2]

        Y_train_features = features_matrix[first_column]
        X_train_features = features_matrix.drop(first_column, axis = 1)

        # Train the model
        model.fit(X_train_features,  Y_train_features)

        features_matrix_val = extraer_caracteristicas_Final_Test(X_valid, y_valid, features_selected = features_select, first_column_name = first_column, out_dir="./galeria_resultados/val")

        Y_val_features = features_matrix_val[first_column]
        X_val_features = features_matrix_val.drop(first_column, axis = 1)

        # Predict and score
        y_valid_pred = model.predict(X_val_features)
        y_valid_proba = model.predict_proba(X_val_features)[:, 1]
        auc = roc_auc_score(y_valid, y_valid_proba)
        #auc_scores.append(auc)

        acc_k, sens_k, spec_k, auc_k, cm_k = calcular_metricas_binarias(Y_val_features, y_valid_pred, y_valid_proba)
  
        print(f"Accuracy: {acc_k: .4f} | Sensibilidad: {sens_k: .4f}")
        print(f"Especificidad: {spec_k: .4f} | AUC: {auc_k: .4f}")

        auc_scores.append(auc_k)
        acc_scores.append(acc_k)
        sens_scores.append(sens_k)
        spec_scores.append(spec_k)

    # Print 
    mean_auc = np.mean(auc_scores)
    print("\nAverage Validation AUC:", round(mean_auc, 4))
    mean_acc = np.mean(acc_scores)
    print("\nAverage Validation ACC:", round(mean_acc, 4))
    mean_sens = np.mean(sens_scores)
    print("\nAverage Validation SENS:", round(mean_sens, 4))
    mean_spec = np.mean(spec_scores)
    print("\nAverage Validation SPEC:", round(mean_spec, 4))


    #K-Fold Manual

    # Perform 5-fold cross-validation
    #scores = cross_val_score(tree, cancer.data, cancer.target, cv=5)

    # Display results
    #print('Cross validation scores: {}'.format(scores))
    #print('Cross validation scores: {:.3f}+-{:.3f}'.format(scores.mean(), scores.std()))

    

    #kernels = ['linear', 'rbf','poly']
    #np.logspace(-3, 2, num=6) # Para C y Gamma [0.001, 0.01, 0.1, 1, 10, 100]

    #for kernel_ in kernels:
    #    for gamma_ in np.logspace(-3, 2, num=6):
    #        for C_ in np.logspace(-3, 2, num=6):
    #            #Modelo
    #            svm = SVC(kernel = kernel_, gamma = gamma_, C=C_, probability=True)
    #            svm.fit(X_train, Y_train) #Se ajusta a entrenamiento
    #            scores[(kernel_, gamma_, C_)] = svm.score(X_val, Y_val) #Puntuación obtenida para hiperparametros sobre set de validacion

    #scores = pd.Series(scores)

    #print(f'Best score: {scores.max():.2f}')
    #print(f'Parametros que obtuvieron Best Score: {scores.idxmax()}')


    # Initialize and train the model using best hyperparameters
    #model = SVC(kernel= scores.idxmax()[0], 
    #            gamma=scores.idxmax()[1], 
    #            C=scores.idxmax()[2])

  

def SVM_Train_GridSearch(X_train, Y_train, X_test, Y_test):

    svm = SVC(probability=True)
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    param_grid = {
    'kernel':['linear', 'rbf','poly'],
    #'gamma': np.logspace(-3, 2, num=6),
    #'C': np.logspace(-3, 2, num=6)
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

def TestModel(model, features_selected, X_test, Y_test, first_column, target1, target2, base_dir = "./DatasetFrutasSegmentadas/test", out_dir = "./galeria_resultados/test"):
       
    target_names =  [target1, target2]

    features_matrix_test = extraer_caracteristicas_Final_Test(X_test, Y_test, features_selected, first_column, out_dir)

    Y_test_features = features_matrix_test[first_column]
    X_test_features = features_matrix_test.drop(first_column, axis = 1)

    y_pred_test = model.predict(X_test_features)
    y_prob_test = model.predict_proba(X_test_features)[:, 1]

    acc_t, sens_t, spec_t, auc_t, cm_t = calcular_metricas_binarias(Y_test, y_pred_test, y_prob_test)

    #ROC
    # false positive rate and true positive rate
    fpr, tpr, thresholds = roc_curve(Y_test, y_prob_test)

    #Visualizacion de Matrices de confusión

    print(f"Accuracy: {acc_t: .4f} | Sensibilidad: {sens_t: .4f}")
    print(f"Especificidad: {spec_t: .4f} | AUC: {auc_t: .4f}")


    #fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    plt.figure(figsize=(5, 5))

    sns.heatmap(
        cm_t,
        annot=True,
        fmt='d',
        cmap='Greens',
        xticklabels=target_names,
        yticklabels=target_names
    )

    plt.title('Matriz de confusión: Test (30%)')
    plt.xlabel('Predicho')
    plt.ylabel('Real')

    #sns.heatmap(cm_t, annot=True, fmt='d', cmap='Greens', ax=ax[1], xticklabels=target_names, yticklabels=target_names)
    #ax[0].set_title('Matriz de confusión: Test (30%)')
    #ax[0].set_xlabel('Predicho')
    #ax[0].set_ylabel('Real')

    os.makedirs(out_dir, exist_ok=True)
    path_mc = os.path.join(out_dir, f"MatrizConfucionTestSVM{first_column}")
    plt.savefig(path_mc)

    plt.tight_layout()
    plt.show()

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
    path_Roc = os.path.join(out_dir, f"ROCCurveSVM{first_column}")
    plt.savefig(path_Roc)

    plt.tight_layout()
    plt.show()

    print(f"Test del modelo terminado {first_column}")


def Train_Final(clasificador, X, Y, first_column_name, target1, target2):

    if(clasificador == 'svm'):
        model = SVC(kernel= 'rbf', probability=True)

    elif(clasificador == 'knn'):
        model = KNeighborsClassifier(n_neighbors = 8, probability=True)

    features_matrix, features_select = extraer_caracteristicas_Final(X, Y, first_column_name, out_dir="./galeria_resultados/test")

    target_names =  [target1, target2]

    Y_train = features_matrix[first_column_name]
    X_train = features_matrix.drop(first_column_name, axis = 1)

    #Entreamiento con todo el dataset
    model.fit(X_train, Y_train)

    return model, features_select 


def SVM_Clasificador(matrix_train, matrix_test, target_1, target_2, out_dir, first_column_name):
    print(f"Test del modelo comienza {first_column_name}")

    target_names =  [target_1, target_2]

    Y_train = matrix_train[first_column_name]
    X_train = matrix_train.drop(first_column_name, axis = 1)

    Y_test = matrix_test[first_column_name]
    X_test = matrix_test.drop(first_column_name, axis = 1)

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

    os.makedirs(out_dir, exist_ok=True)
    path_mc = os.path.join(out_dir, "MatrizConfucionTestSVM")
    plt.savefig(path_mc)

    plt.tight_layout()
    plt.show()

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

    plt.tight_layout()
    plt.show()

    print(f"Test del modelo terminado {first_column_name}")

def modelo_Completo(base_dir_train, base_dir_test):
    X = []
    Y_fruit = []
    Y_state = []

    #Imagenes para train

    print("Procesando imagenes train")

    for class_name in os.listdir(base_dir_train):
        print(f"Procesando clase: {class_name}")
        class_path = os.path.join(base_dir_train, class_name)
        if not os.path.isdir(class_path):
            continue
        files = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg",".jpeg",".png"))]

        for f in files:
            src = os.path.join(class_path, f)
            img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
            if img is not None:
                class_parts = class_name.split("_")
                fruta = class_parts[0]
                estado = class_parts[1]

                X.append(src)
                Y_fruit.append(fruta)
                Y_state.append(estado)


    #Etiquetas numericas para fruta y estado (0 = Banana, 1 = Mango, 0 = Fresh, 1 = Rotten)
    le_fruit = LabelEncoder()
    Y_fruit = le_fruit.fit_transform(Y_fruit)

    le_state = LabelEncoder()
    Y_state = le_state.fit_transform(Y_state)

    X = pd.Series(X)
    Y_fruit = pd.Series(Y_fruit)
    Y_state = pd.Series(Y_state)

    print("Procesando imagenes test")

    #Imagenes para test

    X_test = []
    Y_fruit_test = []
    Y_state_test = []

    for class_name in os.listdir(base_dir_test):
        print(f"Procesando clase: {class_name}")
        class_path = os.path.join(base_dir_test, class_name)
        if not os.path.isdir(class_path):
            continue
        files = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg",".jpeg",".png"))]

        for f in files:
            src = os.path.join(class_path, f)
            img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
            if img is not None:
                class_parts = class_name.split("_")
                fruta = class_parts[0]
                estado = class_parts[1]

                X_test.append(src)
                Y_fruit_test.append(fruta)
                Y_state_test.append(estado)

    #Etiquetas numericas para fruta y estado (0 = Banana, 1 = Mango, 0 = Fresh, 1 = Rotten)
    Y_fruit_test = le_fruit.transform(Y_fruit_test)
    Y_state_test = le_state.transform(Y_state_test)

    X = pd.Series(X)
    Y_fruit_test = pd.Series(Y_fruit_test)
    Y_state_test = pd.Series(Y_state_test)

    print("Test para Tipo de fruta con SVM")

    target1 = 'Mango'
    target2 = 'Banano'
    first_column_name = 'fruit'

    model = SVC(kernel= 'rbf', probability=True)

    model, features_selected = Train_Final('svm', X, Y_fruit, first_column_name, target1, target2)

    TestModel(model, features_selected, X_test, Y_fruit_test, 'fruit', target1, target2, base_dir = "./DatasetFrutasSegmentadas/test", out_dir = "./galeria_resultados/test")

    print("Test para Estado de fruta con SVM")

    target1 = 'Fresca'
    target2 = 'Podrida'
    first_column_name = 'state'

    model, features_selected = Train_Final('svm', X, Y_state, first_column_name, target1, target2)

    TestModel(model, features_selected, X_test, Y_state_test, 'state', target1, target2, base_dir = "./DatasetFrutasSegmentadas/test", out_dir = "./galeria_resultados/test")


    """
    

    print("Cross-Validation para Tipo de fruta con SVM")

    Train_Manual ('svm', X, Y_fruit, target1, target2, first_column_name)

    print("Cross-Validation para Tipo de fruta con KNN")

    #Train_Manual ('knn', X, Y_fruit, target1, target2, first_column_name)

    target1 = 'Fresca'
    target2 = 'Podrida'
    first_column_name = 'state'

    print("Cross-Validation para Estado de fruta con SVM")

    Train_Manual ('svm', X, Y_fruit, target1, target2, first_column_name)

    print("Cross-Validation para Tipo de fruta con SVM")

    #Train_Manual ('knn', X, Y_fruit, target1, target2, first_column_name)

    """

   




