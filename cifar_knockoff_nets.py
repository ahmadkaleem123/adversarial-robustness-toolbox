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

import numpy as np
import torch

from art.attacks.extraction.knockoff_nets import KnockoffNets
from tests.utils import create_image_dataset
from tests.utils import get_image_classifier_pt

logger = logging.getLogger(__name__)

BATCH_SIZE = 128
NB_STOLEN = 5000
f = open("logs.txt", "w")
global victim_ptc


class TestKnockoffNets:

    def __init__(self, train=False, random=True, adaptive=True,
                 dataset='mnist', load_init=True):
        self.train = train
        self.testrandom = random
        self.testadaptive = adaptive
        self.dataset = dataset
        self.load_init = load_init

    def runknockoff(self):
        if self.dataset == 'cifar10':
            """
            Using CIFAR10 as the victim model and CIFAR100 as the attacker's dataset
            :return:
            """
            self.x_train_victim, self.y_train_victim, self.x_test_victim, self.y_test_victim = create_image_dataset(
                n_train=50000, n_test=10000, dataset="cifar10")
            self.x_train_attack, self.y_train_attack, self.x_test_attack, self.y_test_attack = create_image_dataset(
                n_train=50000, n_test=10000, dataset="cifar100")

            self.x_train_victim = np.reshape(self.x_train_victim, (
                self.x_train_victim.shape[0], 3, 32, 32)).astype(np.float32)

            self.x_test_victim = np.reshape(
                self.x_test_victim,
                (self.x_test_victim.shape[0], 3, 32, 32)).astype(np.float32)

            self.x_train_attack = np.reshape(self.x_train_attack, (
                self.x_train_attack.shape[0], 3, 32, 32)).astype(np.float32)

            self.x_test_attack = np.reshape(
                self.x_test_attack,
                (self.x_test_attack.shape[0], 3, 32, 32)).astype(np.float32)

            batch_size = BATCH_SIZE
            nb_epochs = 300

        elif self.dataset == 'mnist':
            self.x_train_victim, self.y_train_victim, self.x_test_victim, self.y_test_victim = create_image_dataset(
                n_train=60000, n_test=10000, dataset="mnist")

            self.x_train_attack, self.y_train_attack, self.x_test_attack, self.y_test_attack = create_image_dataset(
                n_train=60000, n_test=10000, dataset="svhn")

            self.x_train_victim = np.reshape(self.x_train_victim, (
                self.x_train_victim.shape[0], 1, 28, 28)).astype(np.float32)
            self.x_test_victim = np.reshape(self.x_test_victim,
                                            (self.x_test_victim.shape[0], 1, 28,
                                             28)).astype(
                np.float32)

            self.x_train_attack = np.reshape(
                self.x_train_attack, (
                    self.x_train_attack.shape[0], 1, 28, 28)).astype(np.float32)
            self.x_test_attack = np.reshape(
                self.x_test_attack,
                (self.x_test_attack.shape[0], 1, 28, 28)).astype(np.float32)

            batch_size = BATCH_SIZE
            nb_epochs = 20

        victim_ptc = get_image_classifier_pt(dataset=self.dataset,
                                             load_init=self.load_init)
        if self.train:
            print("Starting Training")
            f.write("Starting training\n")
            f.flush()

            victim_ptc.fit(  # train the victim and save
                x=self.x_train_victim,
                y=self.y_train_victim,
                batch_size=batch_size,
                nb_epochs=nb_epochs,
            )
            if hasattr(victim_ptc.model, 'module'):
                model = victim_ptc.model.module
            else:
                model = victim_ptc.model
            torch.save(model.state_dict(),
                       f"model-{self.dataset}.pth.tar")

        victim_preds = np.argmax(
            victim_ptc.predict(x=self.x_test_victim), axis=1)
        y_test_preds = np.argmax(
            self.y_test_victim, axis=1
        )
        count = np.sum((victim_preds == y_test_preds).astype(int))
        print("Victim Accuracy", count / len(victim_preds))

        # Create the thieved classifier
        if self.testrandom:
            print("Start Random Attack.")
            thieved_ptc = get_image_classifier_pt(load_init=False,
                                                  dataset=self.dataset)

            # Create random attack
            attack = KnockoffNets(
                classifier=victim_ptc,
                batch_size_fit=BATCH_SIZE,
                batch_size_query=BATCH_SIZE,
                nb_epochs=100,
                nb_stolen=NB_STOLEN,
                sampling_strategy="random",
                verbose=True,
            )

            thieved_ptc = attack.extract(x=self.x_train_attack,
                                         thieved_classifier=thieved_ptc)

            thieved_preds = np.argmax(
                thieved_ptc.predict(x=self.x_test_victim), axis=1)
            acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)
            count = 0
            for i in range(len(thieved_preds)):
                if np.argmax(self.y_test_victim[i]) == thieved_preds[i]:
                    count += 1
            print("Fidelity Accuracy", acc)
            print("Standard Accuracy", count / len(thieved_preds))
            f.write(f"Random accuracy: {count / len(thieved_preds)}")
        if self.testadaptive:
            # Create adaptive attack

            thieved_ptc = get_image_classifier_pt(load_init=False,
                                                  dataset=self.dataset)

            print("Starting Adaptive attack")
            attack = KnockoffNets(
                classifier=victim_ptc,
                batch_size_fit=BATCH_SIZE,
                batch_size_query=BATCH_SIZE,
                nb_epochs=100,
                nb_stolen=NB_STOLEN,
                sampling_strategy="adaptive",
                reward="all",
                verbose=True,
            )
            thieved_ptc = attack.extract(x=self.x_train_attack,
                                         y=self.y_train_attack,
                                         thieved_classifier=thieved_ptc)

            thieved_preds = np.argmax(
                thieved_ptc.predict(x=self.x_test_victim), axis=1)
            acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)
            count = 0
            for i in range(len(thieved_preds)):
                if np.argmax(self.y_test_victim[i]) == thieved_preds[i]:
                    count += 1
            print("Fidelity Accuracy", acc)
            print("Target Accuracy", count / len(victim_preds))
            f.write(f"Adaptive accuracy: {count / len(victim_preds)}")
        f.close()


if __name__ == "__main__":
    dataset = 'cifar10'

    if dataset == 'cifar10':
        train = True
    elif dataset == 'mnist':
        train = True
    else:
        raise Exception(f"Unknown dataset: {dataset}")

    if train:
        load_init = False
    else:
        load_init = True

    knockoff = TestKnockoffNets(
        train=train,
        random=True,
        adaptive=True,
        dataset=dataset,
        load_init=load_init)
    knockoff.runknockoff()
