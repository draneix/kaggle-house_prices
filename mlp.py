import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import optuna


# ---------- Initialisation of fixed hyperparameters
num_epochs = 1000
criterion = nn.MSELoss()

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class EarlyStopper:
    """
    From: https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    """
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def define_model(trial, n_features):
    # Number of layers
    n_layers = trial.suggest_int("n_layers", 1, 4)
    layers = []
    n_outputs = None

    for i in range(n_layers):
        # For first layer
        if n_outputs is None:
            n_inputs = n_features
        else:
            n_inputs = n_outputs
        # Get number of nodes for the next layer
        n_outputs = trial.suggest_int(f"n_outputs_{i}", 1, 3000)
        layers.append(nn.Linear(n_inputs, n_outputs))
        # Activation layer
        activation_func_name = trial.suggest_categorical(f"act_func_{i}", ["ReLU", "Tanh", "Sigmoid"])
        layers.append(getattr(nn, activation_func_name)())
        # Include dropout?
        if trial.suggest_categorical(f"do_{i}", [True, False]):
            do_prob = trial.suggest_float(f"do_prob_{i}", 0.1, 0.5)
            layers.append(nn.Dropout(do_prob))
        # Include batchnorm
        if trial.suggest_categorical(f"bn_{i}", [True, False]):
            layers.append(nn.BatchNorm1d(n_outputs))

    # Output layer
    layers.append(nn.Linear(n_outputs, 1))

    return nn.Sequential(*layers)


def objective(trial, tensor_train, x_valid, y_valid):

    # Define model
    model = define_model(trial, x_valid.shape[1]).to(device)

    # Model initialisation
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [2 ** 4, 2 ** 5, 2 ** 6])
    optimiser_name = trial.suggest_categorical("optimiser_name", ["Adam", "SGD"])
    optimiser = getattr(optim, optimiser_name)(model.parameters(), lr=learning_rate)
    loader_train = DataLoader(tensor_train, shuffle=True, batch_size=batch_size)

    for epoch in range(num_epochs):
        # Train model
        model.train()
        for batch, (inputs_train, targets_train) in enumerate(loader_train):
            inputs_train = inputs_train.to(device)
            targets_train = targets_train.to(device)
            outputs_train = model(inputs_train)
            optimiser.zero_grad()

            loss = criterion(outputs_train, targets_train)
            loss.backward()
            optimiser.step()

        # Validate model for early stopping
        model.eval()
        with torch.no_grad():
            x_valid = x_valid.to(device)
            y_valid = y_valid.to(device)
            outputs_valid = model(x_valid)
            loss_valid = criterion(outputs_valid, y_valid)

        # Trial pruning
        trial.report(loss_valid, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # Early stopping
        early_stopper = EarlyStopper(patience=10, min_delta=0.05)
        if early_stopper.early_stop(loss_valid):
            break

    return loss_valid
