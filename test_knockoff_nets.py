# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
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

import numpy as np

from art.attacks.extraction.knockoff_nets import KnockoffNets
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin

from tests.utils import TestBase, master_seed
from tests.utils import get_image_classifier_pt
from tests.utils import get_tabular_classifier_pt
from tests.attacks.utils import backend_test_classifier_type_check_fail

import torchvision
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

BATCH_SIZE = 10
NB_EPOCHS = 100
NB_STOLEN = 500


class TestKnockoffNets(TestBase):
    """
    A unittest class for testing the KnockoffNets attack.
    """

    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234, set_tensorflow=True)
        super().setUpClass()

    def setUp(self):
        super().setUp()


    def test_5_pytorch_classifier(self):
        """
        Third test with the PyTorchClassifier.
        :return:
        """
        self.x_train_mnist = np.reshape(self.x_train_mnist, (self.x_train_mnist.shape[0], 1, 28, 28)).astype(np.float32)
        self.x_test_mnist = np.reshape(self.x_test_mnist,
                            (self.x_test_mnist.shape[0], 1, 28, 28)).astype(
            np.float32)
        self.x_train_cifar10 = np.reshape(self.x_train_cifar10, (self.x_train_cifar10.shape[0], 1, 28, 28)).astype(np.float32)
        self.x_test_cifar10 = np.reshape(self.x_test_cifar10,
                            (self.x_test_cifar10.shape[0], 1, 28, 28)).astype(
            np.float32)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.13251461,), (0.31048025,))])
        base_dataset = torchvision.datasets.MNIST(
            "/ssd003/home/akaleem/data/MNIST", train=True, download=False,
            transform=transform)
        base_datasettest = torchvision.datasets.MNIST(
            "/ssd003/home/akaleem/data/MNIST", train=False, download=False,
            transform=transform)

        # Build PyTorchClassifier
        victim_ptc = get_image_classifier_pt()
        print("Training Victim")
        victim_ptc.fit(  # train the victim
            x=self.x_train_mnist,
            y=self.y_train_mnist,
            batch_size=64,
            nb_epochs=100,
        )
        print("Done Training")
        count2 = 0
        victim_preds = np.argmax(victim_ptc.predict(x=self.x_test_mnist), axis=1)
        for i in range(len(victim_preds)):
            if base_datasettest.targets[i] == victim_preds[i]:
                count2 += 1
        print("Victim Accuracy", count2/len(victim_preds))   

        # Create thieved classifier
        thieved_ptc = get_image_classifier_pt(load_init=False)

        # # Create random attack
        attack = KnockoffNets(
            classifier=victim_ptc,
            batch_size_fit=BATCH_SIZE,
            batch_size_query=BATCH_SIZE,
            nb_epochs=NB_EPOCHS,
            nb_stolen=NB_STOLEN,
            sampling_strategy="random",
            verbose=False,
        )

        #thieved_ptc = attack.extract(x=self.x_train_mnist, thieved_classifier=thieved_ptc) # mnist on mnist
        thieved_ptc = attack.extract(x=self.x_train_cifar10, thieved_classifier=thieved_ptc) # cifar10 to attack mnist

        
        thieved_preds = np.argmax(thieved_ptc.predict(x=self.x_test_mnist), axis=1)
        acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)
        count = 0
        #print("y test", self.y_test_mnist)
        for i in range(len(thieved_preds)):
            if (base_datasettest.targets[i] == thieved_preds[i]):   # can also use y_test_mnist if we use the argmax. 
                count += 1
        # print("len1", len(victim_preds))
        # print("len2", len(self.x_train_mnist))
        # print("len3", len(self.x_test_mnist))

        print("Fidelity Accuracy", acc)
        print("Standard Accuracy", count/len(thieved_preds))
        #print("victim", count2)
        #self.assertGreater(acc, 0.3)

        # Create adaptive attack

        self.x_train_mnist = np.reshape(self.x_train_mnist, (self.x_train_mnist.shape[0], 1, 28, 28)).astype(np.float32)
        self.x_train_cifar10 = np.reshape(self.x_train_cifar10, (self.x_train_cifar10.shape[0], 1, 28, 28)).astype(np.float32)
        self.x_test_cifar10 = np.reshape(self.x_test_cifar10, (self.x_test_cifar10.shape[0], 1, 28, 28)).astype(np.float32)
        # print("shape1", self.x_train_mnist.shape)
        # print("shape2", self.x_test_cifar10.shape)
        # Create thieved classifier
        thieved_ptc = get_image_classifier_pt(load_init=False)

        print("Starting adaptive attack")
        attack = KnockoffNets(
            classifier=victim_ptc,
            batch_size_fit=BATCH_SIZE,
            batch_size_query=BATCH_SIZE,
            nb_epochs=NB_EPOCHS,
            nb_stolen=NB_STOLEN,
            sampling_strategy="adaptive",
            reward="all",
            verbose=False,
        )
        #thieved_ptc = attack.extract(x=self.x_train_mnist, y=self.y_train_mnist, thieved_classifier=thieved_ptc) # with mnist on mnist
        thieved_ptc = attack.extract(x=self.x_train_cifar10, y=self.y_train_cifar10, thieved_classifier=thieved_ptc) # cifar 10 to attack mnist
        #thieved_ptc = attack.extract(x=self.x_test_cifar10, y=self.y_test_cifar10, thieved_classifier=thieved_ptc) # doesnt work for some reason

        thieved_preds = np.argmax(thieved_ptc.predict(x=self.x_test_mnist), axis=1) 
        acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)
        count = 0
        count2 = 0
        for i in range(len(thieved_preds)):
            if base_datasettest.targets[i] == thieved_preds[i]:
                count += 1
        print("Fidelity Accuracy", acc)
        print("Target Accuracy", count / len(victim_preds))
        #print("victim", count2)

        #self.assertGreater(acc, 0.4)

    # def test_5_pytorch_classifier(self):
    #     """
    #     Third test with the PyTorchClassifier. (using cifar10)
    #     :return:
    #     """
    #     self.x_train_cifar10 = np.reshape(self.x_train_cifar10, (self.x_train_cifar10.shape[0], 3, 32, 32)).astype(np.float32)
    #     self.x_test_cifar10 = np.reshape(self.x_test_cifar10,
    #                         (self.x_test_cifar10.shape[0], 3, 32, 32)).astype(
    #         np.float32)
    #     transform=transforms.Compose([
    #         transforms.Pad(4),
    #         transforms.RandomCrop(32),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(
    #             (
    #                 0.49139969,
    #                 0.48215842,
    #                 0.44653093),
    #             (
    #                 0.24703223,
    #                 0.24348513,
    #                 0.26158784)
    #         )])
    #     base_dataset = torchvision.datasets.CIFAR10(
    #         "/ssd003/home/akaleem/data/CIFAR10", train=True, download=False,
    #         transform=transform)
    #     base_datasettest = torchvision.datasets.CIFAR10(
    #         "/ssd003/home/akaleem/data/CIFAR10", train=False, download=False,
    #         transform=transform)

    #     # Build PyTorchClassifier
    #     victim_ptc = get_image_classifier_pt(dataset="cifar10")

    #     victim_ptc.fit(  # train the victim
    #         x=self.x_train_cifar10,
    #         y=self.y_train_cifar10,
    #         batch_size=64,
    #         nb_epochs=100,
    #     )

        # print("Done training")

        # # Create the thieved classifier
        # thieved_ptc = get_image_classifier_pt(load_init=False)

        # # Create random attack
        # attack = KnockoffNets(
        #     classifier=victim_ptc,
        #     batch_size_fit=BATCH_SIZE,
        #     batch_size_query=BATCH_SIZE,
        #     nb_epochs=NB_EPOCHS,
        #     nb_stolen=NB_STOLEN,
        #     sampling_strategy="random",
        #     verbose=False,
        # )

        # thieved_ptc = attack.extract(x=self.x_train_cifar10, thieved_classifier=thieved_ptc)

        # victim_preds = np.argmax(victim_ptc.predict(x=self.x_test_cifar10), axis=1)
        # thieved_preds = np.argmax(thieved_ptc.predict(x=self.x_test_cifar10), axis=1)
        # acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)
        # count = 0
        # count2 = 0
        # #print("y test", self.y_test_cifar10)
        # for i in range(len(thieved_preds)):
        #     if (base_datasettest.targets[i] == thieved_preds[i]):   # can also use y_test_cifar10 if we use the argmax. 
        #         count += 1
        #     if base_datasettest.targets[i] == victim_preds[i]:
        #         count2 += 1
        # # print("len1", len(victim_preds))
        # # print("len2", len(self.x_train_cifar10))
        # # print("len3", len(self.x_test_cifar10))

        # print("Fidelity Accuracy", acc)
        # print("Standard Accuracy", count/len(thieved_preds))
        # print("Victim Accuracy", count2/len(thieved_preds))
        # #print("victim", count2)
        # #self.assertGreater(acc, 0.3)

        # # Create adaptive attack

        # #self.x_train_cifar10 = np.reshape(self.x_train_cifar10, (self.x_train_cifar10.shape[0], 1, 28, 28)).astype(np.float32)
        # #print("shape", self.x_train_cifar10.shape)
        # thieved_ptc = get_image_classifier_pt(load_init=False)

        # print("Starting adaptive attack")
        # attack = KnockoffNets(
        #     classifier=victim_ptc,
        #     batch_size_fit=BATCH_SIZE,
        #     batch_size_query=BATCH_SIZE,
        #     nb_epochs=NB_EPOCHS,
        #     nb_stolen=NB_STOLEN,
        #     sampling_strategy="adaptive",
        #     reward="all",
        #     verbose=False,
        # )
        # thieved_ptc = attack.extract(x=self.x_train_cifar10, y=self.y_train_cifar10, thieved_classifier=thieved_ptc)

        # victim_preds = np.argmax(victim_ptc.predict(x=self.x_test_cifar10), axis=1)
        # thieved_preds = np.argmax(thieved_ptc.predict(x=self.x_test_cifar10), axis=1) 
        # acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)
        # count = 0
        # count2 = 0
        # for i in range(len(thieved_preds)):
        #     if base_datasettest.targets[i] == thieved_preds[i]:
        #         count += 1
        #     if base_datasettest.targets[i] == victim_preds[i]:
        #         count2 += 1
        # print("Fidelity Accuracy", acc)
        # print("Target Accuracy", count / len(victim_preds))
        # print("Victim Accuracy", count2 / len(victim_preds))
        #print("victim", count2)

        #self.assertGreater(acc, 0.4)

        

    def test_1_classifier_type_check_fail(self):
        backend_test_classifier_type_check_fail(KnockoffNets, [BaseEstimator, ClassifierMixin])


    # def test_4_pytorch_iris(self):
    #     """
    #     Third test for PyTorch.
    #     :return:
    #     """
    #     print("With Iris")
    #     # Build PyTorchClassifier
    #     victim_ptc = get_tabular_classifier_pt()

    #     # Create the thieved classifier
    #     thieved_ptc = get_tabular_classifier_pt(load_init=False)

    #     # Create random attack
    #     attack = KnockoffNets(
    #         classifier=victim_ptc,
    #         batch_size_fit=BATCH_SIZE,
    #         batch_size_query=BATCH_SIZE,
    #         nb_epochs=NB_EPOCHS,
    #         nb_stolen=NB_STOLEN,
    #         sampling_strategy="random",
    #         verbose=False,
    #     )
    #     thieved_ptc = attack.extract(x=self.x_train_iris, thieved_classifier=thieved_ptc)

    #     victim_preds = np.argmax(victim_ptc.predict(x=self.x_train_iris), axis=1)
    #     thieved_preds = np.argmax(thieved_ptc.predict(x=self.x_train_iris), axis=1)
    #     acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)

    #     self.assertGreater(acc, 0.3)

    #     # Create adaptive attack
    #     attack = KnockoffNets(
    #         classifier=victim_ptc,
    #         batch_size_fit=BATCH_SIZE,
    #         batch_size_query=BATCH_SIZE,
    #         nb_epochs=NB_EPOCHS,
    #         nb_stolen=NB_STOLEN,
    #         sampling_strategy="adaptive",
    #         reward="all",
    #         verbose=False,
    #     )
    #     thieved_ptc = attack.extract(x=self.x_train_iris, y=self.y_train_iris, thieved_classifier=thieved_ptc)

    #     victim_preds = np.argmax(victim_ptc.predict(x=self.x_train_iris), axis=1)
    #     thieved_preds = np.argmax(thieved_ptc.predict(x=self.x_train_iris), axis=1)
    #     acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)
    #     print("fidelity acc", acc)

    #     self.assertGreater(acc, 0.4)


# Write this again but with cifar100 to attack cifar10. look into y target and stuff

if __name__ == "__main__":
    unittest.main()
