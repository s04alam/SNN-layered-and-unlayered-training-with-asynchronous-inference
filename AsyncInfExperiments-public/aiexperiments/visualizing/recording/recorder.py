import matplotlib.pyplot as plt
import pickle
import os

import torch

class BatchResult:
    def __init__(self, loss, accuracy, batch_size, regularization_loss=0):
        if torch.is_tensor(loss):
            self.loss = loss.item()
        else:
            self.loss = loss
        if torch.is_tensor(accuracy):
            self.accuracy = accuracy.item()
        else:
            self.accuracy = accuracy
        self.batch_size = batch_size
        if torch.is_tensor(regularization_loss):
            self.regularization_loss = regularization_loss.item()
        else:
            self.regularization_loss = regularization_loss

class Recorder:
    def __init__(self):
        self.accuracies = []
        self.losses = []
        self.batch_sizes = []
        self.regularization_losses = []

    def record(self, result: BatchResult):
        self.accuracies.append(result.accuracy)
        self.losses.append(result.loss)
        self.batch_sizes.append(result.batch_size)
        self.regularization_losses.append(result.regularization_loss)

    def calculate_weighted_accuracy(self): # useful for calculating the average accuracy over multiple batches (test set)
        return sum([a * b for a, b in zip(self.accuracies, self.batch_sizes)]) / sum(self.batch_sizes)

    def calculate_weighted_loss(self): # useful for calculating the average loss over multiple batches (test set)
        return sum([a * b for a, b in zip(self.losses, self.batch_sizes)]) / sum(self.batch_sizes)
    
    def calculate_weighted_regularization_loss(self):
        return sum([a * b for a, b in zip(self.regularization_losses, self.batch_sizes)]) / sum(self.batch_sizes)

    # Loads a Recorder object from a file. If the file does not exist, returns a new Recorder object.
    def load(file_name):
        if not os.path.exists(file_name):
            return Recorder()
        else:
            with open(file_name, 'rb') as f:
                return pickle.load(f)

    def save(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)