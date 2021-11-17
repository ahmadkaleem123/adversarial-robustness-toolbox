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
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import logging
import unittest
from unittest.case import skip

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

BATCH_SIZE = 64
NB_EPOCHS = 100  # make 200
NB_STOLEN = 2000
f = open("logs.txt", "w")
global victim_ptc


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

    def test_4_pytorch_classifier(self):
        """
        Third test with the PyTorchClassifier. (using cifar100 on cifar10)
        :return:
        """
        self.x_train_cifar10 = np.reshape(self.x_train_cifar10, (
        self.x_train_cifar10.shape[0], 3, 32, 32)).astype(np.float32)
        self.x_test_cifar10 = np.reshape(self.x_test_cifar10,
                                         (self.x_test_cifar10.shape[0], 3, 32,
                                          32)).astype(
            np.float32)
        self.x_train_cifar100 = np.reshape(self.x_train_cifar100, (
        self.x_train_cifar100.shape[0], 3, 32, 32)).astype(np.float32)
        self.x_test_cifar100 = np.reshape(self.x_test_cifar100,
                                          (self.x_test_cifar100.shape[0], 3, 32,
                                           32)).astype(
            np.float32)
        transform = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (
                    0.49139969,
                    0.48215842,
                    0.44653093),
                (
                    0.24703223,
                    0.24348513,
                    0.26158784)
            )])
        base_datasettest = torchvision.datasets.CIFAR10(
            "/ssd003/home/akaleem/data/CIFAR10", train=False, download=False,
            transform=transform)

        # Build PyTorchClassifier
        victim_ptc = get_image_classifier_pt(dataset="cifar10")
        print("Starting Training")
        f.write("Starting training\n")

        victim_ptc.fit(  # train the victim and save
            x=self.x_train_cifar10,
            y=self.y_train_cifar10,
            batch_size=128,
            nb_epochs=300,
        )
        import torch
        torch.save(victim_ptc.model.state_dict(), "model2.pth.tar")

        count2 = 0
        victim_preds = np.argmax(victim_ptc.predict(x=self.x_test_cifar10),
                                 axis=1)
        # victim_preds = np.argmax(victim_ptc.predict(x=base_datasettest.data.reshape(len(base_datasettest.data), 3,32,32)),
        #                          axis=1)
        for i in range(len(victim_preds)):
            # print("victim pred", victim_preds[i])
            # print("other",self.y_test_cifar10[i])
            if np.argmax(self.y_test_cifar10[i]) == victim_preds[i]:
                count2 += 1
        print("Victim Accuracy", count2 / len(victim_preds))
        f.write(f"Victim Accuracy: {count2 / len(victim_preds)}")

        # Create the thieved classifier
        # thieved_ptc = get_image_classifier_pt(load_init=False, dataset="cifar10")
        #
        # # Create random attack
        # attack = KnockoffNets(
        #     classifier=victim_ptc,
        #     batch_size_fit=BATCH_SIZE,
        #     batch_size_query=BATCH_SIZE,
        #     nb_epochs=50,
        #     nb_stolen=NB_STOLEN,
        #     sampling_strategy="random",
        #     verbose=False,
        # )
        #
        # thieved_ptc = attack.extract(x=self.x_train_cifar100, thieved_classifier=thieved_ptc)
        #
        # thieved_preds = np.argmax(thieved_ptc.predict(x=self.x_test_cifar10), axis=1)
        # acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)
        # count = 0
        # #print("y test", self.y_test_cifar10)
        # for i in range(len(thieved_preds)):
        #     if (base_datasettest.targets[i] == thieved_preds[i]):   # can also use y_test_cifar10 if we use the argmax.
        #         count += 1
        # # print("len1", len(victim_preds))
        # # print("len2", len(self.x_train_cifar10))
        # # print("len3", len(self.x_test_cifar10))
        #
        # print("Fidelity Accuracy", acc)
        # print("Standard Accuracy", count/len(thieved_preds))
        # f.write(f"Random accuracy: {count/len(thieved_preds)}")
        # #print("victim", count2)
        # self.assertGreater(acc, 0.3)

        # Create adaptive attack

        # self.x_train_cifar10 = np.reshape(self.x_train_cifar10, (self.x_train_cifar10.shape[0], 1, 28, 28)).astype(np.float32)
        # print("shape", self.x_train_cifar10.shape)
        # thieved_ptc = get_image_classifier_pt(load_init=False, dataset="cifar10")
        #
        # print("Starting Adaptive attack")
        # attack = KnockoffNets(
        #     classifier=victim_ptc,
        #     batch_size_fit=BATCH_SIZE,
        #     batch_size_query=BATCH_SIZE,
        #     nb_epochs=50,
        #     nb_stolen=NB_STOLEN,
        #     sampling_strategy="adaptive",
        #     reward="all",
        #     verbose=True,
        # )
        # thieved_ptc = attack.extract(x=self.x_train_cifar100, y=self.y_train_cifar100, thieved_classifier=thieved_ptc)
        #
        # thieved_preds = np.argmax(thieved_ptc.predict(x=self.x_test_cifar10), axis=1)
        # acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)
        # count = 0
        # count2 = 0
        # for i in range(len(thieved_preds)):
        #     if base_datasettest.targets[i] == thieved_preds[i]:
        #         count += 1
        # print("Fidelity Accuracy", acc)
        # print("Target Accuracy", count / len(victim_preds))
        # f.write(f"Adaptive accuracy: {count / len(victim_preds)}")
        f.close()

    def test_1_classifier_type_check_fail(self):
        backend_test_classifier_type_check_fail(KnockoffNets, [BaseEstimator,
                                                               ClassifierMixin])


if __name__ == "__main__":
    unittest.main()
