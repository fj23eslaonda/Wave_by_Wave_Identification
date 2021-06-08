import tensorflow as tf
from tensorflow import keras
from keras import backend as K

def f1(y_true, y_pred):  # Creacion de Pr, Re y F1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    tp = K.sum(y_true_f * y_pred_f)  # suma de una AND   son los True Points
    fp = K.sum(K.clip(K.clip(y_pred_f + y_true_f, 0, 1) - y_true_f, 0,1))  # k.clip (argumento, min, max) el argumento le da un minimo y un maximo
    fn = K.sum(K.clip(K.clip(y_pred_f + y_true_f, 0, 1) - y_pred_f, 0, 1))
    Pr = tp / (tp + fp)
    Re = tp / (tp + fn)
    f1t = 2 * (Pr * Re) / (Pr + Re)
    return f1t 
