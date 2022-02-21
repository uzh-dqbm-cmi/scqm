"""
Model interface
"""
from typing import Tuple
from scqm.ports.model import Model
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold

import os.path
import re
import uuid
import joblib


class ModelDNN(Model):
    """
    Abstract model class that acts as an interface for training and evaluating machine learning models.
    Specific models inherit and implement the abstract methods of this class. The objective is to remove the dependency
    on specific frameworks (e.g., sklearn, pytorch, tensorflow) and make code shareable.
    """

    def __init__(self, config: dict):
        """
        Initialize a new model object, according to the configurations passed in

        Args:
            config: dictionary containing the configuration parameters specific to the model
        """
        super().__init__(config)

        # Model parameters
        self.config = config
        self.input_layer = config['input_layer']
        self.hidden_layers = config['hidden_layers']
        self.hidden_layers_activations = config['hidden_layer_activations']
        self.output_layer = config['output_layer']
        self.output_layer_activation = config['output_layer_activations']
        # Error handling - input/...

        self.model = self.build()

    def build(self):
        # Model definition via TF Functional API
        inputs = tf.keras.Input(shape=self.input_layer)
        length = len(self.hidden_layers)

        if length == 0:  # Logistic regression
            last_hidden_layer = inputs
        else:  # Deep neural network with at least 1 hidden later
            h_l = list()
            for i in range(length):
                h_l.append(tfl.Dense(units=self.hidden_layers[i], activation=self.hidden_layers_activations[i])(temp))
                temp = h_l[i]
            last_hidden_layer = h_l[-1]

        outputs = tfl.Dense(units=self.output_layer, activation=self.output_layer_activation)(last_hidden_layer)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        return model

    def train(self, train_data: Tuple[np.array, np.array], test_data: Tuple[np.array, np.array], config: dict) -> None:
        """
        Train the model using the train and validation data passed in as two tuples.

        Args:
            train_data (tuple): the training data to be used for the model (first element: features array of shape
            num_samples x num_features, second element: target array of shape num_samples x 1)
            test_data (tuple): the validation data to be used for the model (first element: features array of shape
            num_samples x num_features, second element: target array of shape num_samples x 1)
        """
        train_x, train_y = train_data
        test_x, test_y = test_data

        optimizer = config['optimiser']
        loss = config['loss']
        metrics = config['metrics']
        epochs = config['epochs']
        batch_size = config['batch_size']
        num_folds = config['num_of_folds']
        verbosity = config['verbosity']

        # K fold stratification
        # Merge inputs and targets
        inputs = np.concatenate((train_x, test_x), axis=0)
        targets = np.concatenate((train_y, test_y), axis=0)

        # Define the K-fold Cross Validator
        kfold = KFold(n_splits=num_folds, shuffle=True)

        # Define per-fold score containers
        acc_per_fold = []
        loss_per_fold = []

        # K-fold Cross Validation model evaluation
        fold_no = 1
        for train, test in kfold.split(inputs, targets):
            # Generate a print
            print('------------------------------------------------------------------------')
            print(f'Training for fold {fold_no} ...')

            # Build model
            self.model = self.build()

            # Compile the model
            self.model.compile(loss=loss,
                               optimizer=optimizer,
                               metrics=metrics)
            # Fit data to model
            self.history = self.model.fit(inputs[train], targets[train],
                                     batch_size=batch_size,
                                     epochs=epochs,
                                     verbose=verbosity)

            # Generate generalization metrics
            scores = self.model.evaluate(inputs[test], targets[test], verbose=0)
            print(
                f'Score for fold {fold_no}: {self.model.metrics_names[0]} of {scores[0]}; {self.model.metrics_names[1]} of {scores[1] * 100}%')
            acc_per_fold.append(scores[1] * 100)
            loss_per_fold.append(scores[0])

            # Increase fold number
            fold_no = fold_no + 1

    def test(self, test_data: Tuple[np.array, np.array]) -> None:


    def predict(self, x: np.array) -> np.array:
        """
        Perform predictions with the model on specific data

        Args:
            x: Input features to be used to perform prediction using the model. The data needs to be formatted to shape
            num_samples x num_features.

        Returns:
            y (np.array): array containing the predictions for the passed in data (num_samples x 1)
        """
        predictions = self.model.predict(x)

        return predictions

    def _generate_filename(self):
        new_uuid = str(uuid.uuid4())
        model_name = re.sub("\s+", "-", self.model_name.strip())
        filename = new_uuid + "_" + model_name + ".joblib.pkl"
        return filename

    def save(self, filename: os.path) -> os.path:
        """
        Save the model in a joblib.pkl file.
        Args:
            filename: Filename used to store the model. If None is provided, a filename of the following pattern will be
                      generated: <uuid>_<model_name>.joblib.pkl

        Returns:
            os.path: Filename tht was used to store the model.
        """
        self.filename = filename
        if filename is None:
            print("Specify a filename for the model to be saved.")
            self.filename = "output/models/" + self._generate_filename()
        with open(self.filename, 'wb') as f:
            joblib.dump(self.model, f, compress=9)

        print(self.model_name, "saved to", self.filename)


        # option 2
        tf.keras.models.save_model(
            self.model,
            self.filename,
            overwrite=True,
            include_optimizer=True,
            save_format=None,
            signatures=None,
            options=None
        )

        return self.filename

    def metrics(self):
            """
            Get all relevant performance metrics of the model.
            Returns:
                Metrics: Metrics object specified in Metrics class.
            """
            print("Calculate metrics")
            print(self.history())
            return self.history()
            if 0:
                self.predictions = self.model.predict(self.x_test)
                print("predictions:", self.predictions)
                confusion_matrix = sklearn.metrics.confusion_matrix(self.y_test, self.predictions)
                score = self.model.score(self.x_test, self.y_test)
                print(f"Accuracy: {score * 100:0.2f}%")
                # return BinaryClassificationMetrics(confusion_matrix, score)



