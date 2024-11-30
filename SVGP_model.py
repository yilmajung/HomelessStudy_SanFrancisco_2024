import json
import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error

def main(config):
    print("batch_size: ", config["batch_size"], "  ",
    "num_inducing_points: ", config["num_inducing_points"], "  ",
    "embedding_dim: ", config["embedding_dim"], "  ",
    "training_iterations: ", config["training_iterations"], "  ",
    "learning_rate: ", config["learning_rate"])

    # Move the model to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the data
    df = pd.read_csv('~/GroundingDINO/svgp/df_cleaned_10292024.csv')

    # Separate df into training and testing data
    df_train_val = df[df['ground_truth'].notnull()]
    df_test = df[df['ground_truth'].isnull()]

    # Shuffle the data
    df_train_val = df_train_val.sample(frac=1, random_state=42)
    df_train_val = df_train_val.reset_index(drop=True)

    # Define the Latent Embedding Kernel and combine with other features
    class LatentEmbeddingAndFeatureKernel(gpytorch.kernels.Kernel):
        def __init__(self, num_areas, embedding_dim, feature_kernel, **kwargs):
            super(LatentEmbeddingAndFeatureKernel, self).__init__(**kwargs)
            
            # Learnable embeddings for each area (discrete)
            self.embedding = torch.nn.Embedding(num_areas, embedding_dim)
            
            # Kernel for continuous features
            self.feature_kernel = feature_kernel
        
        def forward(self, x1, x2, diag=False, **params):
            # Split x1 and x2 into bboxid and continuous features
            bboxid_x1, features_x1 = x1[:, 0].long(), x1[:, 1:]
            bboxid_x2, features_x2 = x2[:, 0].long(), x2[:, 1:]

            # Compute the embeddings for bboxid
            embed_x1 = self.embedding(bboxid_x1)
            embed_x2 = self.embedding(bboxid_x2)

            # Standardize the continuous features
            features_x1 = (features_x1 - features_x1.mean(dim=0)) / features_x1.std(dim=0)
            features_x2 = (features_x2 - features_x2.mean(dim=0)) / features_x2.std(dim=0)

            # Combine embeddings with the continuous features
            combined_x1 = torch.cat([embed_x1, features_x1], dim=-1)
            combined_x2 = torch.cat([embed_x2, features_x2], dim=-1)

            # Apply the RBF kernel to the combined inputs
            if diag:
                # Return only the diagonal elements when requested
                return self.feature_kernel(combined_x1, combined_x2, diag=True)
            else:
                # Return the full covariance matrix when diagonal is not requested
                return self.feature_kernel(combined_x1, combined_x2)

        def diag(self, x):
            # Handle diagonal requests by passing `diag=True` to the forward method
            return self.forward(x, x, diag=True)
            
            # # Apply a standard kernel to the combined inputs (embeddings + continuous features)
            # return self.feature_kernel(combined_x1, combined_x2)

    # Define the GP Model with Poisson Likelihood
    class GPModel(gpytorch.models.ApproximateGP):
        def __init__(self, inducing_points, num_areas):
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
            variational_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
            super(GPModel, self).__init__(variational_strategy)

            # Define the kernel: latent embedding for area codes + kernel for temperature and demographic features
            feature_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=embedding_dim + 7) # 7  additional features
            self.covar_module = LatentEmbeddingAndFeatureKernel(num_areas, embedding_dim, feature_kernel)
            self.mean_module = gpytorch.means.ConstantMean()

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        
    # Define the Poisson likelihood for count data (number of tents)
    class PoissonLikelihood(gpytorch.likelihoods.Likelihood):
        def forward(self, function_samples, *args, **kwargs):
            # Apply the log-link function (exponentiate the latent GP)
            return torch.distributions.Poisson(rate=function_samples.exp())


    # Set up k-fold cross-validation (k=5)
    fold_size = len(df_train_val) // 5
    scores = []

    for i in range(5):
        start = i * fold_size
        end = (i + 1) * fold_size

        # Separate df into training and validation data
        df_train = pd.concat([df_train_val.iloc[:start], df_train_val.iloc[end:]])
        df_val = df_train_val.iloc[start:end]
    
    # # Separate df_train into training and validation data
    # df_train_val = df_train_val.sample(frac=1, random_state=42)
    # df_train_val = df_train_val.reset_index(drop=True)
    # df_train = df_train_val.iloc[:int(0.8 * len(df_train_val))]
    # df_val = df_train_val.iloc[int(0.8 * len(df_train_val)):]
    # df_val = df_val.reset_index(drop=True)

        # Get the number of unique areas and days
        num_areas = df_train['bboxid'].nunique()
        num_days = df_train['timestamp'].nunique()
        embedding_dim = config['embedding_dim']
        batch_size = config["batch_size"]

        # Create a bboxid tensor
        bboxid_tensor = torch.tensor(df_train['bboxid'].astype('category').cat.codes.values, dtype=torch.long).to(device)

        # Create a temperature tensor
        temperature_tensor = torch.tensor(df_train[['max','min','precipitation']].values, dtype=torch.float32).to(device)

        # Create a demographic tensor
        demographic_tensor = torch.tensor(df_train[['total_population','white_ratio','black_ratio','hh_median_income']].values, dtype=torch.float32).to(device)

        # Create a ground truth tensor
        target_tensor = torch.tensor(df_train['ground_truth'].values, dtype=torch.float32).to(device)

        # Prepare the dataset
        train_dataset = TensorDataset(torch.cat([bboxid_tensor.unsqueeze(-1), temperature_tensor, demographic_tensor], dim=-1), target_tensor)

        # Set up a DataLoader for mini-batching
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Concatenate the bboxid tensor, temperature tensor, and demographic tensor into a single tensor
        train_x = torch.cat([bboxid_tensor.unsqueeze(-1), temperature_tensor, demographic_tensor], dim=-1)

        # Define inducing points for the sparse approximation (subset of training data)
        num_inducing_points = config["num_inducing_points"]
        inducing_points = train_x[torch.randperm(train_x.size(0))[:num_inducing_points]] # Randomly select inducing points

        # Initialize model and likelihood
        model = GPModel(inducing_points, num_areas=len(bboxid_tensor.unique()))
        model = model.to(device)
        likelihood = PoissonLikelihood()
        likelihood = likelihood.to(device)

        # Training the model
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': likelihood.parameters()},
        ], lr=config['learning_rate'])

        # Loss function: Variational ELBO
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(train_dataset))

        # Training loop
        training_iterations = config['training_iterations']
        for j in range(training_iterations):
            model.train()
            likelihood.train()

            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                output = model(batch_x)
                loss = -mll(output, batch_y)
                loss.backward()
                optimizer.step()

            if j % 10 == 0:
                print(f'Iteration {j}/{training_iterations}, Fold: {i} - Loss: {loss.item()}')

        # # Save the model
        # checkpoint = {
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'loss': loss.item(),
        #     'config': config
        # }
        # torch.save(checkpoint, f'checkpoint_indpts_{num_inducing_points}_embdim_{embedding_dim}.pth')
        # print("Model saved successfully!")

    # print(f"train_x shape: {train_x.shape}")
    # print(f"inducing_points shape: {inducing_points.shape}")
    # print(f"batch_x shape: {batch_x.shape}")
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
        val_dataset = TensorDataset(torch.cat([bboxid_tensor_val.unsqueeze(-1), temperature_tensor_val, demographic_tensor_val], dim=-1), target_tensor_val) 
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Concatenate the bboxid tensor, temperature tensor, and demographic tensor into a single tensor
        val_x = torch.cat([bboxid_tensor_val.unsqueeze(-1), temperature_tensor_val, demographic_tensor_val], dim=-1)

        # Define inducing points for the sparse approximation (subset of training data)
        inducing_points = val_x[torch.randperm(val_x.size(0))[:num_inducing_points]] # Randomly select inducing points

        # print(f"val_x shape: {val_x.shape}")
        # print(f"inducing_points_val shape: {inducing_points.shape}")

    
        # with torch.no_grad(), gpytorch.settings.fast_pred_var():
        #     for batch_x in val_loader:
        #         batch_x = batch_x[0].to(device)
        #         batch_predictions = likelihood(model(batch_x))
        #         all_predictions.append(batch_predictions.mean.cpu().numpy())
        
        # all_predictions = np.concatenate(all_predictions, axis=0)
        
        # Evaluate performance
        total_mse = 0
        total_points = 0

        # Forecasting (prediction) with validation set
        model.eval()
        likelihood.eval()

        for batch_x_val, batch_y_val in val_loader:
            batch_x_val = batch_x_val.to(device)
            #print(f"batch_x_val shape: {batch_x_val.shape}")
            batch_predictions = likelihood(model(batch_x_val)).mean.detach().cpu().numpy()
            batch_predictions = batch_predictions.mean(axis=0)
            batch_y_val = batch_y_val.squeeze().cpu().numpy()

            # print(f"batch_predictions shape: {batch_predictions.shape}")
            # print(f"batch_y_val shape: {batch_y_val.shape}")

            batch_mse = mean_squared_error(batch_y_val, batch_predictions)
            total_mse += batch_mse * len(batch_y_val)
            total_points += len(batch_y_val)

        # Calculate RMSE
        rmse = np.sqrt(total_mse / total_points)
        print(f'Validation RMSE: {rmse}')
        scores.append(rmse)
    
    print(f"Average RMSE: {np.mean(scores)}")


if __name__=="__main__":
    with open('config.json') as file:
        config = json.load(file)

    main(config)



# Need to separate df_train into training and validation data before predicting them Yilma0210!#
# Add cholesky_jitter to the model