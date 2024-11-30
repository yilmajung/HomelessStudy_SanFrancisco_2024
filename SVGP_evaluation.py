import json
import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error

def load_model_for_evaluation(filepath, config):
    checkpoint = torch.load(filepath)
    num_areas = df['bboxid'].nunique()  # Calculate number of unique areas from your data
    model = GPModel(inducing_points=torch.rand((config['num_inducing_points'], config['embedding_dim'] + 7)).to(device), num_areas=num_areas)
    model = model.to(device)
    likelihood = PoissonLikelihood().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    likelihood.eval()
    return model, likelihood

model, likelihood = load_model_for_evaluation('model.pth', config)

    # Forecasting (prediction) with validation set
    model.eval()
    likelihood.eval()

    # Validation set
    # Create a bboxid tensor
    bboxid_tensor_val = torch.tensor(df_val['bboxid'].astype('category').cat.codes.values, dtype=torch.long)

    # Create a temperature tensor
    temperature_tensor_val = torch.tensor(df_val[['max','min','precipitation']].values, dtype=torch.float32)

    # Create a demographic tensor
    demographic_tensor_val = torch.tensor(df_val[['total_population','white_ratio','black_ratio','hh_median_income']].values, dtype=torch.float32)

    # Create a ground truth tensor
    target_tensor_val = torch.tensor(df_val['ground_truth'].values, dtype=torch.float32)

    # Concatenate the bboxid tensor, temperature tensor, and demographic tensor into a single tensor
    val_x = torch.cat([bboxid_tensor_val.unsqueeze(-1), temperature_tensor_val, demographic_tensor_val], dim=-1)

    # Make predictions
    val_loader = DataLoader(TensorDataset(val_x, target_tensor_val), batch_size=batch_size, shuffle=False)
    # all_predictions = []

    # model.eval()
    # likelihood.eval()

    # with torch.no_grad(), gpytorch.settings.fast_pred_var():
    #     for batch_x in val_loader:
    #         batch_x = batch_x[0].to(device)
    #         batch_predictions = likelihood(model(batch_x))
    #         all_predictions.append(batch_predictions.mean.cpu().numpy())
    
    # all_predictions = np.concatenate(all_predictions, axis=0)
    
    # Evaluate performance
    total_mse = 0
    total_points = 0

    with torch.no_grad:

        for batch_x, batch_y in val_loader:
            batch_x = batch_x[0].to(device)
            batch_predictions = likelihood(model(batch_x)).mean.cpu().numpy()
            batch_predictions = batch_predictions.squeeze()
            batch_y = batch_y.squeeze().cpu().numpy()
            
            # Compute batch-wise MSE
            batch_mse = mean_squared_error(batch_y, batch_predictions)
            total_mse += batch_mse * len(batch_y)
            total_points += len(batch_y)

    # Calculate RMSE
    rmse = np.sqrt(total_mse / total_points)
    print(f'Validation RMSE: {rmse}')


if __name__=="__main__":
    with open('config.json') as file:
        config = json.load(file)

    main(config)



# Need to separate df_train into training and validation data before predicting them
# Add cholesky_jitter to the model