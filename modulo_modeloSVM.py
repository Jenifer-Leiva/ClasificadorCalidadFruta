from core.libs import os, cv2, np, plt, sns, pd

from modulo_caracteristicas_forma import extraer_caracteristicas_forma
from modulo_aumentar import augmentar_dataset
from modulo_caracteristicas_color_textura_forma import extraer_caracteristicas_Final, extraer_caracteristicas_Final_Test

from core.libs import os, cv2, np, plt, sns, pd, joblib
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score,  StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, recall_score, confusion_matrix, roc_auc_score, precision_score,  roc_curve)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

#X -> Explanatory variables
#Y -> Target Variable

def Seleccion_Caracteristicas_Train(X, Y, first_column_name, out_dir="./galeria_resultados/train"):
    
    features_matrix, features_select = extraer_caracteristicas_Final(X, Y, first_column_name, out_dir)

    return features_matrix, features_select

def Seleccion_Caracteristicas_Test(X_test, Y_test, first_column_name, features_selected):
    
    features_matrix_test = extraer_caracteristicas_Final_Test(X_test, Y_test, features_selected, first_column_name, out_dir="./galeria_resultados/test")

    return features_matrix_test

def Train_Final(clasificador, first_column_name, target1, target2, features_matrix):


    if(clasificador == 'svm'):
        model = SVC(kernel= 'rbf', gamma = 'scale', C=1.0, probability=True)


    #features_matrix, features_select = extraer_caracteristicas_Final(X, Y, first_column_name, out_dir="./galeria_resultados/test")

    target_names =  [target1, target2]

    Y_train = features_matrix[first_column_name]
    X_train = features_matrix.drop(first_column_name, axis = 1)

    #Entreamiento con todo el dataset
    model.fit(X_train, Y_train)
    joblib.dump(model, f'./Interfaz_Clasificar_Fruta/trained_model_{first_column_name}.pkl')

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

def TestModel(model, first_column, target1, target2, features_matrix_test, out_dir = "./galeria_resultados/test"):
       
    target_names =  [target1, target2]

    Y_test_features = features_matrix_test[first_column]
    X_test_features = features_matrix_test.drop(first_column, axis = 1)

    y_pred_test = model.predict(X_test_features)
    y_prob_test = model.predict_proba(X_test_features)[:, 1]

    acc_t, sens_t, spec_t, auc_t, cm_t = calcular_metricas_binarias(Y_test_features, y_pred_test, y_prob_test)

    #ROC
    # false positive rate and true positive rate
    fpr, tpr, thresholds = roc_curve(Y_test_features, y_prob_test)

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
    path_mc = os.path.join(out_dir, f"MatrizConfucionTest{first_column}")
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

def Test_completo():
    #---->TEST<----#

    print("Test para Tipo de fruta con SVM")

    target1 = 'Banano'
    target2 = 'Mango'
    first_column_name = 'fruit'

    #model = SVC(kernel= 'rbf', probability=True)

    features_matrix_test_fruit = pd.read_csv(f"./galeria_resultados/test/MatrizCaracteristicasNormalizada_Test{first_column_name}.csv")
    features_matrix_fruit = pd.read_csv(f"./galeria_resultados/train/MatrizCaracteristicasNormalizada_{first_column_name}.csv")

    modelSVM_fruit = Train_Final('svm',first_column_name, target1, target2, features_matrix_fruit)

    TestModel(modelSVM_fruit, 'fruit', target1, target2, features_matrix_test_fruit, out_dir = "./galeria_resultados/test")

    print("Test para Estado de fruta con SVM")

    target1 = 'Sano'
    target2 = 'Dañado'
    first_column_name = 'state'

    features_matrix_test_state = pd.read_csv(f"./galeria_resultados/test/MatrizCaracteristicasNormalizada_Test{first_column_name}.csv")
    features_matrix_state = pd.read_csv(f"./galeria_resultados/train/MatrizCaracteristicasNormalizada_{first_column_name}.csv")

    modelSVM_state = Train_Final('svm', first_column_name, target1, target2, features_matrix_state)

    TestModel(modelSVM_state, 'state', target1, target2, features_matrix_test_state, out_dir = "./galeria_resultados/test")


def Cross_Validation(model, features_matrix_train, target1, target2, first_column_name, out_dir = "./galeria_resultados/test"):

    if(model == 'svm'):
        model = SVC(kernel= 'rbf', gamma = 'scale', C=1.0, probability=True)

    elif(model == 'lr'):
        model = LogisticRegression()


    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    print(f"Cross Validation K-FOLD {model}")

    target_names =  [target1, target2]

    y_train = features_matrix_train[first_column_name]
    X_train = features_matrix_train.drop(first_column_name, axis = 1)

    y_pred_kf = cross_val_predict(model, X_train, y_train, cv=kf)
    y_prob_kf = cross_val_predict(model, X_train, y_train, cv=kf, method="predict_proba")[:, 1]

    acc_k, sens_k, spec_k, auc_k, cm_k = calcular_metricas_binarias(y_train, y_pred_kf, y_prob_kf)

    print(f"Accuracy: {acc_k: .4f} | Sensibilidad: {sens_k: .4f}")
    print(f"Especificidad: {spec_k: .4f} | AUC: {auc_k: .4f}")

    plt.figure(figsize=(5, 5))

    sns.heatmap(
        cm_k,
        annot=True,
        fmt='d',
        cmap='Greens',
        xticklabels=target_names,
        yticklabels=target_names
    )

    plt.title('Matriz de confusión: K-Fold (Entrenamiento) 70%')
    plt.xlabel('Predicho')
    plt.ylabel('Real')

    os.makedirs(out_dir, exist_ok=True)
    path_mc = os.path.join(out_dir, f"MatrizConfucionTrain{model} {first_column_name}")
    plt.savefig(path_mc)

    plt.tight_layout()
    plt.show()

def Cross_Validation_completo():
    
    print("Cross Validation Fruit")

    target1 = 'Banano'
    target2 = 'Mango'
    first_column_name = 'fruit'

    features_matrix_fruit = pd.read_csv(f"./galeria_resultados/val/MatrizCaracteristicasNormalizada_{first_column_name}.csv")

    Cross_Validation('svm', features_matrix_fruit, target1, target2, first_column_name, out_dir = "./galeria_resultados/test")
    Cross_Validation('lr', features_matrix_fruit, target1, target2, first_column_name, out_dir = "./galeria_resultados/test")

    print("Cross Validation State")

    target1 = 'Sano'
    target2 = 'Dañado'
    first_column_name = 'state'

    features_matrix_state = pd.read_csv(f"./galeria_resultados/val/MatrizCaracteristicasNormalizada_{first_column_name}.csv")


    Cross_Validation('svm', features_matrix_state, target1, target2, first_column_name, out_dir = "./galeria_resultados/test")
    Cross_Validation('lr', features_matrix_state, target1, target2, first_column_name, out_dir = "./galeria_resultados/test")


def modelo_Completo(base_dir_train_val, base_dir_train_test, base_dir_test):

    X_seg = []
    Y_fruit_seg = []
    Y_state_seg = []

    #Imagenes para train

    print("Procesando imagenes train solo segmentadas para cross-validation")

    for class_name in os.listdir(base_dir_train_val):
        print(f"Procesando clase: {class_name}")
        class_path = os.path.join(base_dir_train_val, class_name)
        if not os.path.isdir(class_path):
            continue
        files1 = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg",".jpeg",".png"))]

        for f in files1:
            src = os.path.join(class_path, f)
            img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
            if img is not None:
                class_parts = class_name.split("_")
                fruta = class_parts[0]
                estado = class_parts[1]

                X_seg.append(src)
                Y_fruit_seg.append(fruta)
                Y_state_seg.append(estado)


    #Etiquetas numericas para fruta y estado (0 = Banana, 1 = Mango, 0 = Fresh, 1 = Rotten)
    le_fruit = LabelEncoder()
    Y_fruit_seg = le_fruit.fit_transform(Y_fruit_seg)

    le_state = LabelEncoder()
    Y_state_seg = le_state.fit_transform(Y_state_seg)

    X_seg = pd.Series(X_seg)
    Y_fruit_seg = pd.Series(Y_fruit_seg)
    Y_state_seg = pd.Series(Y_state_seg)

    print("Procesando imagenes train aumentadas para test")
    #Imagenes para train
    X_aug = []
    Y_fruit_aug= []
    Y_state_aug= []

    for class_name in os.listdir(base_dir_train_test):
        print(f"Procesando clase: {class_name}")
        class_path = os.path.join(base_dir_train_test, class_name)
        if not os.path.isdir(class_path):
            continue
        files2 = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg",".jpeg",".png"))]

        for f in files2:
            src = os.path.join(class_path, f)
            img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
            if img is not None:
                class_parts = class_name.split("_")
                fruta = class_parts[0]
                estado = class_parts[1]

                X_aug.append(src)
                Y_fruit_aug.append(fruta)
                Y_state_aug.append(estado)


    #Etiquetas numericas para fruta y estado (0 = Banana, 1 = Mango, 0 = Fresh, 1 = Rotten)
    Y_fruit_aug = le_fruit.transform(Y_fruit_aug)
    Y_state_aug = le_state.transform(Y_state_aug)

    X_aug = pd.Series(X_aug)
    Y_fruit_aug = pd.Series(Y_fruit_aug)
    Y_state_aug = pd.Series(Y_state_aug)

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
        files3 = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg",".jpeg",".png"))]

        for f in files3:
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

    X_test = pd.Series(X_test)
    Y_fruit_test = pd.Series(Y_fruit_test)
    Y_state_test = pd.Series(Y_state_test)

    print("Extracción y seleccion de caracteristicas con train (70%) para fruit")
    features_matrix_fruit, features_select_fruit = Seleccion_Caracteristicas_Train(X_aug, Y_fruit_aug, 'fruit')
    pd.DataFrame(features_select_fruit, columns=["feature"]).to_csv("./Interfaz_Clasificar_Fruta/features_selected_fruit.csv", index=False)
    
    #print("Extracción y seleccion de caracteristicas con train (70%) para fruit Validación")
    features_matrix_fruit_val, features_select_fruit_val = Seleccion_Caracteristicas_Train(X_seg, Y_fruit_seg, 'fruit', "./galeria_resultados/val")

    #print("Extracción y seleccion de caracteristicas test (30%) para fruit")
    features_matrix_test_fruit = Seleccion_Caracteristicas_Test(X_test, Y_fruit_test, "fruit", features_select_fruit)

    print("Extracción y seleccion de caracteristicas con train (70%) para state")
    features_matrix_state, features_select_state = Seleccion_Caracteristicas_Train(X_aug, Y_state_aug, 'state')
    pd.DataFrame(features_select_state, columns=["feature"]).to_csv("./Interfaz_Clasificar_Fruta/features_selected_state.csv", index=False)

    #print("Extracción y seleccion de caracteristicas con train (70%) para state Validación")
    features_matrix_state_val, features_select_state_val = Seleccion_Caracteristicas_Train(X_seg, Y_state_seg, 'state',"./galeria_resultados/val")

    #print("Extracción y seleccion de caracteristicas test (30%) para state")
    features_matrix_test_state = Seleccion_Caracteristicas_Test(X_test, Y_state_test, 'state', features_select_state)

    print("Fruit:", le_fruit.classes_)
    print("State:", le_state.classes_)

    joblib.dump(le_fruit, "./Interfaz_Clasificar_Fruta/label_encoder_fruit.pkl")
    joblib.dump(le_state, "./Interfaz_Clasificar_Fruta/label_encoder_state.pkl")

    #Cross_Validation_completo()

    #Test_completo()


   




