import torch.nn as nn

def pointwise_feedforward_network(d_model, dff):
    model = nn.Sequential(
        nn.Linear(d_model, dff),
        nn.ReLU(),
        nn.Linear(dff, d_model)
    )

    return model