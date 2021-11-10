# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

# import keras
# import keras.backend as k
import numpy as np
# import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dfmenetwork

# from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
# from keras.models import Sequential

from art.attacks.extraction.copycat_cnn import CopycatCNN
from art.estimators.classification.keras import KerasClassifier
from art.estimators.classification.pytorch import PyTorchClassifier
from art.estimators.classification.tensorflow import TensorFlowClassifier

import torchvision
from torchvision import datasets, transforms

from tests.utils import (
    TestBase,
    get_image_classifier_kr,
    get_image_classifier_pt,
    get_image_classifier_tf,
    get_tabular_classifier_kr,
    get_tabular_classifier_pt,
    get_tabular_classifier_tf,
    master_seed,
)

logger = logging.getLogger(__name__)

NB_EPOCHS = 50
NB_STOLEN = 1000


class TestCopycatCNN(TestBase):
    """
    A unittest class for testing the CopycatCNN attack.
    """

    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234)
        super().setUpClass()

    def test_pytorch_classifier(self):
        """
        Third test with the PyTorchClassifier.
        :return:
        """
        x_train = np.reshape(self.x_train_mnist, (self.x_train_mnist.shape[0], 1, 28, 28)).astype(np.float32)
        x_test = np.reshape(self.x_test_mnist,
                             (self.x_test_mnist.shape[0], 1, 28, 28)).astype(
            np.float32)
        transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.13251461,), (0.31048025,))])
        base_dataset = torchvision.datasets.MNIST("/ssd003/home/akaleem/data/MNIST", train=True, download=False, transform=transform)
        base_datasettest = torchvision.datasets.MNIST(
            "/ssd003/home/akaleem/data/MNIST", train=False, download=False,
            transform=transform)
        # Build PyTorchClassifier
        victim_ptc = get_image_classifier_pt()  # Change this to below.
        victim_ptc.fit(  # train the victim
            x=x_train,
            y=base_dataset.targets,
            batch_size=64,
            nb_epochs=100,
        )




        # class Model(nn.Module):
        #     """
        #     Create model for pytorch.
        #     """
        #
        #     def __init__(self):
        #         super(Model, self).__init__()
        #
        #         self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7)
        #         self.pool = nn.MaxPool2d(4, 4)
        #         self.fullyconnected = nn.Linear(25, 10)
        #
        #     # pylint: disable=W0221
        #     # disable pylint because of API requirements for function
        #     def forward(self, x):
        #         """
        #         Forward function to evaluate the model
        #
        #         :param x: Input to the model
        #         :return: Prediction of the model
        #         """
        #         x = self.conv(x)
        #         x = torch.nn.functional.relu(x)
        #         x = self.pool(x)
        #         x = x.reshape(-1, 25)
        #         x = self.fullyconnected(x)
        #         x = torch.nn.functional.softmax(x, dim=1)
        #
        #         return x
        class Model(nn.Module):
            """Class used to initialize model of student/teacher"""

            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5, 1)
                self.conv2 = nn.Conv2d(20, 50, 5, 1)
                self.fc1 = nn.Linear(4 * 4 * 50, 500)
                self.fc2 = nn.Linear(500, 10)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.max_pool2d(x, 2, 2)
                x = F.relu(self.conv2(x))
                x = F.max_pool2d(x, 2, 2)
                x = x.view(-1, 4 * 4 * 50)
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        # Define the network
        model = Model()

        # Define a loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # Get classifier
        thieved_ptc = PyTorchClassifier(
            model=model, loss=loss_fn, optimizer=optimizer, input_shape=(1, 28, 28), nb_classes=10, clip_values=(0, 1)
        )

        # Create attack
        copycat_cnn = CopycatCNN(
            classifier=victim_ptc,
            batch_size_fit=self.batch_size,
            batch_size_query=self.batch_size,
            nb_epochs=NB_EPOCHS,
            nb_stolen=NB_STOLEN,
        )
        # Use x_test instead of x_train
        thieved_ptc = copycat_cnn.extract(x=x_test, thieved_classifier=thieved_ptc)
        victim_preds = np.argmax(victim_ptc.predict(x=x_test[:1000]), axis=1)
        thieved_preds = np.argmax(thieved_ptc.predict(x=x_test[:1000]), axis=1)
        count = 0
        count2 = 0
        for i in range(len(thieved_preds)):
            if (base_datasettest.targets[i] == thieved_preds[i]):
                count+= 1
            if base_datasettest.targets[i] == victim_preds[i]:
                count2 += 1

        # How do we access the dataset labels?. We could just manually import it. Assuming we have the same order. See maketargets.py for this
        acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)
        print("Fidelity Accuracy", acc)
        acc2 = count/len(victim_preds)
        print("Target Accuracy", acc2)
        acc3 = count2/len(victim_preds)
        print("Victim Accuracy", acc3)
        self.assertGreater(acc, 0.3)

# Need to train the victim model as well or otherwise use the victim model from the normal version.

class TestCopycatCNNVectors(TestBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def test_pytorch_iris(self):
        """
        Third test for PyTorch.
        :return:
        """
        # Build PyTorchClassifier
        victim_ptc = get_tabular_classifier_pt()

        class Model(nn.Module):
            """
            Create Iris model for PyTorch.
            """

            def __init__(self):
                super(Model, self).__init__()

                self.fully_connected1 = nn.Linear(4, 10)
                self.fully_connected2 = nn.Linear(10, 10)
                self.fully_connected3 = nn.Linear(10, 3)

            # pylint: disable=W0221
            # disable pylint because of API requirements for function
            def forward(self, x):
                x = self.fully_connected1(x)
                x = self.fully_connected2(x)
                logit_output = self.fully_connected3(x)

                return logit_output

        # Define the network
        model = Model()

        # Define a loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Get classifier
        thieved_ptc = PyTorchClassifier(
            model=model,
            loss=loss_fn,
            optimizer=optimizer,
            input_shape=(4,),
            nb_classes=3,
            clip_values=(0, 1),
            channels_first=True,
        )

        # Create attack
        copycat_cnn = CopycatCNN(
            classifier=victim_ptc,
            batch_size_fit=self.batch_size,
            batch_size_query=self.batch_size,
            nb_epochs=NB_EPOCHS,
            nb_stolen=NB_STOLEN,
        )
        thieved_ptc = copycat_cnn.extract(x=self.x_train_iris, thieved_classifier=thieved_ptc)

        victim_preds = np.argmax(victim_ptc.predict(x=self.x_train_iris[:100]), axis=1)
        thieved_preds = np.argmax(thieved_ptc.predict(x=self.x_train_iris[:100]), axis=1)
        acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)
        print("fidelity accuracy", acc)

        self.assertGreater(acc, 0.3)


if __name__ == "__main__":
    unittest.main()
