import torch, numpy as np, pandas as pd
from torch import tensor
from fastai.data.transforms import RandomSplitter

# Read file and adjust data to scale down bigger values
df = pd.read_csv("discs.csv")
df["DIAMETER (cm)"] = df["DIAMETER (cm)"]/10
df["INSIDE RIM DIAMETER (cm)"] = df["INSIDE RIM DIAMETER (cm)"]/10

# Independant and dependant variables (turn and fade are rather based on side profile)
t_dep = tensor(df[["SPEED", "GLIDE"]].values, dtype=torch.float)
indep_cols = df.drop(columns = ["MOLD", "DISC TYPE", "SPEED", "GLIDE", "TURN", "FADE", "STABILITY"])
t_indep = tensor(indep_cols.values, dtype = torch.float)

# Generate coefficients and bias
n_inputs = t_indep.shape[1]
n_outputs = t_dep.shape[1]
coeffs = (torch.rand(n_inputs, n_outputs) - 0.5).requires_grad_()
bias = (torch.rand(n_outputs) - 0.5).requires_grad_()

def calc_preds(coeffs, indeps, bias):
    return indeps @ coeffs + bias

def calc_loss(coeffs, indeps, deps, bias):
    return torch.abs(calc_preds(coeffs, indeps, bias) - deps).mean() 

# Split into training and validation sets
trn_split, val_split = RandomSplitter()(df)
trn_indep, val_indep = t_indep[trn_split], t_indep[val_split]
trn_dep, val_dep = t_dep[trn_split], t_dep[val_split]

def update_params(coeffs, bias, lr):
    coeffs.sub_(coeffs.grad * lr)
    bias.sub_(bias.grad * lr)

# Run through one cycle of training
def one_epoch(coeffs, bias, lr):
    coeffs.grad = None
    bias.grad = None
    loss = calc_loss(coeffs, trn_indep, trn_dep, bias)
    loss.backward()
    with torch.no_grad():
        update_params(coeffs, bias, lr)
        
def init_params():
    return (torch.rand(n_inputs, n_outputs)-0.5).requires_grad_(), (torch.rand(n_outputs) - 0.5).requires_grad_()

# Train model for x epochs and y learning rate
def train_model(epochs, lr):
    coeffs, bias = init_params()
    for i in range(epochs):
        one_epoch(coeffs, bias, lr)
    return coeffs, bias

coeffs, bias = train_model(500, 0.04)

# User inputted values for a disc
diameter = float(input("Diameter(cm):"))/10
height = float(input("Height(cm):"))
rim_depth = float(input("Rim_depth(cm):"))
rim_thickness = float(input("Rim_thickness(cm):"))
inside_rim_diameter = float(input("Inside_rim_diameter:"))/10

# Generate a new disc tensor based on user input
new_disc = torch.tensor([[diameter, height, rim_depth, rim_thickness, inside_rim_diameter, height/diameter]], dtype=torch.float)

with torch.no_grad():
    preds = calc_preds(coeffs, new_disc, bias)

# Predict speed and glide of inputted disc
print(f"SPEED: {preds[0,0]:.3f}, GLIDE: {preds[0,1]:.3f}")

