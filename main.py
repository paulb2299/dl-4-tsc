from utils.utils import generate_results_csv
from utils.utils import create_directory
from utils.utils import read_dataset
from utils.utils import transform_mts_to_ucr_format
from utils.utils import visualize_filter
from utils.utils import viz_for_survey_paper
from utils.utils import viz_cam #Se importan las librerías y utilidades necesarias desde utils
import os
import numpy as np
import sys
import sklearn
import utils
from utils.constants import CLASSIFIERS
from utils.constants import ARCHIVE_NAMES
from utils.constants import ITERATIONS
from utils.utils import read_all_datasets #Desde la carpeta constants se importan las iteraciones, nombres de clasificadores y nombres de archivos


def fit_classifier(datasets_dict, dataset_name, classifier_name, output_directory):
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]
    #Se crean vectores para los dataset de entrenamiento y prueba de los modelos
    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    input_shape = x_train.shape[1:]
    classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory)

    classifier.fit(x_train, y_train, x_test, y_test, y_true)


def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=True):
    if classifier_name == 'fcn':
        from classifiers import fcn
        return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mlp':
        from classifiers import mlp
        return mlp.Classifier_MLP(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'resnet':
        from classifiers import resnet
        return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcnn':
        from classifiers import mcnn
        return mcnn.Classifier_MCNN(output_directory, verbose)
    if classifier_name == 'tlenet':
        from classifiers import tlenet
        return tlenet.Classifier_TLENET(output_directory, verbose)
    if classifier_name == 'twiesn':
        from classifiers import twiesn
        return twiesn.Classifier_TWIESN(output_directory, verbose)
    if classifier_name == 'encoder':
        from classifiers import encoder
        return encoder.Classifier_ENCODER(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcdcnn':
        from classifiers import mcdcnn
        return mcdcnn.Classifier_MCDCNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'cnn':  # Time-CNN
        from classifiers import cnn
        return cnn.Classifier_CNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'inception':
        from classifiers import inception
        return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose)


############################################### main

def init_train(root_dir = './', results_dir = './'):
    # this is the code used to launch an experiment on a dataset
    archive_name = 'UCRArchive_2018'
    dataset_name = 'MITECG'
    classifier_names = ['fcn','mlp','resnet','mcdcnn','mcnn','cnn'] # Aqui se configuran los clasificadores que van a entrenarse
    itr = '_itr_1'

    for classifier_name in classifier_names:
        print(f'Empezando entrenamiento utilizando el algoritmo {classifier_name}')
        output_directory = results_dir + '/' + classifier_name + '/' + archive_name + itr + '/' + \
                           dataset_name + '/'
        test_dir_df_metrics = output_directory + 'df_metrics.csv'
        print('Method: ', archive_name, dataset_name, classifier_name, itr)
        if os.path.exists(test_dir_df_metrics):
            print('Already done')
        else:
            create_directory(output_directory)
            datasets_dict = read_dataset(root_dir, archive_name, dataset_name)
            fit_classifier(datasets_dict, dataset_name, classifier_name, output_directory)
            print('LISTO')
            # the creation of this directory means
            create_directory(output_directory + '/DONE')
