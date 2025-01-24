import numpy as np
import pandas as pd

def save(network):
    weights = {}
    for layer in network:
        if isinstance(layer,Dense):
            weights[layer.name] = layer.weights
    print(weights)
    for weight in weights:
        
        df = pd.DataFrame(weights[weight])
        filename = f"Model_Weights/{weight}_weights.csv"
        df.to_csv(filename, index=False, header=False)  # Save without headers for simplicity
        print(f"Saved {weight} to {filename}")