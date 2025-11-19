import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

import torch
import torch.nn  as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F




def create_directory(directory_path):
    if not os.path.isdir(directory_path):
        print('Directory doesnt exist')
        os.makedirs(directory_path)

def load_data(file_name):

    # folder_path = "/home/jabdullayev/Codes/UCRArchive_2018/"
    folder_path = "/home/jabdullayev/phd/datasets/UCRArchive_2018/"
    folder_path += file_name + "/"

    train_path = folder_path + file_name + "_TRAIN.tsv"
    test_path = folder_path + file_name + "_TEST.tsv"

    if os.path.exists(test_path) <= 0:
        print("File not found")
        return None, None, None, None

    train = np.loadtxt(train_path, dtype=np.float64)
    test = np.loadtxt(test_path, dtype=np.float64)

    ytrain = train[:, 0]
    ytest = test[:, 0]

    xtrain = np.delete(train, 0, axis=1)
    xtest = np.delete(test, 0, axis=1)

    return xtrain, ytrain, xtest, ytest

def znormalisation(x):

    stds = np.std(x, axis=1, keepdims=True)
    if len(stds[stds == 0.0]) > 0:
        stds[stds == 0.0] = 1.0
        return (x - x.mean(axis=1, keepdims=True)) / stds
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True))

def encode_labels(y):

    labenc = LabelEncoder()

    return labenc.fit_transform(y)

def preprocess_data(data, target, mini_batch_size=64, shuffle=True,):

    data = znormalisation(data)
    data = np.expand_dims(data, axis=1)

    target = encode_labels(target)

    data, target = torch.from_numpy(data), torch.from_numpy(target)

    torch.manual_seed(42)    
    dataloader = DataLoader(
        TensorDataset(data, target),
        batch_size=mini_batch_size,
        shuffle=shuffle,
    )

    return dataloader

def plot_loss_and_acc_curves(training_losses, val_losses, training_accuracies, val_accuracies, out_dir):
    plt.plot(training_losses, label='train_loss')
    plt.plot(val_losses, label='val_loss')
    plt.savefig(out_dir + 'losses.png')
    plt.clf()
    plt.cla()
    plt.close()
    plt.plot(training_accuracies, label='train_acc')
    plt.plot(val_accuracies, label='val_acc')
    plt.savefig(out_dir + 'accuracies.png')
    plt.clf()
    plt.cla()
    plt.close()


# Plotly Codes
def update_fig_layout(fig):

    plot_bg_color='rgba(255, 255,255, 0.8)'

    fig.update_layout(xaxis_title="",)
    fig.update_layout(xaxis_title="Dim1",  yaxis_title="Dim2",plot_bgcolor=plot_bg_color,)
    fig.update_yaxes( showline=True, linewidth=2, linecolor='white', mirror=True, showgrid=True, gridwidth=0.1, gridcolor='white', tickprefix='', ticksuffix=' ') 

    fig.update_xaxes(visible=True, showticklabels=True)
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgb(230, 230, 230)')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgb(230, 230, 230)')

    fig.update_layout(autosize=False, width=1000, height=500)

    return fig

def compute_redundant_mask(feature_map, cosine_threshold=0.95):
    batch_size, num_channels, ts_length = feature_map.shape
    normalized = F.normalize(feature_map, p=2, dim=2, eps=1e-8) 
    cosine_sim = torch.bmm(normalized, normalized.transpose(1, 2))
    eye = torch.eye(num_channels, device=cosine_sim.device).unsqueeze(0)  
    cosine_sim = cosine_sim - eye * 2.0
    
    lower_mask = torch.tril(torch.ones_like(cosine_sim), diagonal=-1) 
    masked_cosine = cosine_sim * lower_mask  
    
    max_sim, _ = masked_cosine.max(dim=2)
    keep_mask = max_sim <= cosine_threshold
    
    keep_mask[:, 0] = True
    expanded_mask = keep_mask.unsqueeze(-1).float() 
    return expanded_mask

def compute_non_activated_mask(feature_map):
    mean_activation = feature_map.mean(axis=2)
    sample_thresholds = mean_activation.mean(axis=1)  
    active_filters_mask = mean_activation > sample_thresholds.view(-1, 1)
    batch_size, num_filters, ts_length = feature_map.shape
    return active_filters_mask.view(batch_size, num_filters, 1)

def orthogonality_loss_feature(feature_map):
    """
    Compute orthogonality loss on feature maps.

    Args:
        feature_map (torch.Tensor): Shape (batch_size, num_filters, time_series_length)

    Returns:
        loss (torch.Tensor): Scalar value, encouraging orthogonality of feature representations.
    """
    batch_size, num_filters, ts_length = feature_map.shape
    
    # Reshape feature map: Treat each filter (channel) as a vector
    feature_matrix = feature_map.view(batch_size, num_filters, -1)  # Shape: (batch_size, num_filters, ts_length)
    
    # Normalize each filter along the time dimension
    feature_matrix = F.normalize(feature_matrix, p=2, dim=2)  # Normalize across time axis

    # Compute Gram matrix: measures similarity between feature channels
    gram_matrix = torch.bmm(feature_matrix, feature_matrix.transpose(1, 2))  # (batch_size, num_filters, num_filters)

    # Create identity matrix for comparison (encouraging orthogonality)
    identity = torch.eye(num_filters, device=feature_map.device).unsqueeze(0)  # Shape: (1, num_filters, num_filters)

    # Compute mean squared error loss between Gram matrix and identity matrix
    loss = F.mse_loss(gram_matrix, identity.expand_as(gram_matrix))

    return loss


class MFDLoss(torch.nn.Module):
    def __init__(self):
        super(MFDLoss, self).__init__()

    def forward(self, feature_map):
        B, C, T = feature_map.shape 

        feature_map_centered = feature_map - feature_map.mean(dim=-1, keepdim=True)

        std_dev = feature_map_centered.std(dim=-1, unbiased=True, keepdim=True) + 1e-6
        feature_map_normalized = feature_map_centered / std_dev

        corr_matrices = torch.bmm(feature_map_normalized, feature_map_normalized.transpose(1, 2)) / (T - 1)

        avg_corr_matrix = torch.abs(corr_matrices).mean(dim=0)  

        triu_indices = torch.triu_indices(C, C, offset=1)
        redundancy_scores = torch.pow(avg_corr_matrix[triu_indices[0], triu_indices[1]], 2)

        loss = redundancy_scores.mean()
        
        avg_corr_matrix_new = avg_corr_matrix
        loss_new = torch.norm(avg_corr_matrix_new, p='fro') ** 2 / (C * C)


        return loss + loss_new

class MFDLoss_2(torch.nn.Module):
    """
    Multi-Stage Feature Decorrelation (MFD) Loss with Batch-Averaged Correlation.
    """
    def __init__(self):
        super(MFDLoss_2, self).__init__()

    def forward(self, feature_map):
        """
        Compute the MFD loss.

        Args:
        - feature_map (torch.Tensor): Shape (B, C, T), where:
            B = Batch size,
            C = Number of channels,
            T = Time series length.

        Returns:
        - loss (torch.Tensor): Scalar MFD loss value.
        """
        B, C, T = feature_map.shape  # Extract dimensions

        # Step 1: Mean-center each channel per sample
        feature_map_centered = feature_map - feature_map.mean(dim=-1, keepdim=True)

        # Step 2: Compute standard deviation for normalization (Avoid division by zero)
        std_dev = feature_map_centered.std(dim=-1, unbiased=True, keepdim=True) + 1e-6
        feature_map_normalized = feature_map_centered / std_dev

        # Step 3: Compute Pearson Correlation Matrices per sample
        corr_matrices = torch.bmm(feature_map_normalized, feature_map_normalized.transpose(1, 2)) / (T - 1)

        # Step 4: Average correlation matrix over batch
        avg_corr_matrix = corr_matrices.mean(dim=0)  # Shape: (C, C)

        # Step 5: Extract upper triangular values (excluding diagonal)
        triu_indices = torch.triu_indices(C, C, offset=1)
        redundancy_scores = torch.pow(avg_corr_matrix[triu_indices[0], triu_indices[1]], 2)  # Squared correlation

        # Step 6: Compute overall MFD loss
        loss = redundancy_scores.mean()

        return loss

class CovarianceLoss(torch.nn.Module):
    """
    Covariance Regularization Loss.
    Encourages feature independence by minimizing covariance between feature channels.
    """
    def __init__(self):
        super(CovarianceLoss, self).__init__()

    def forward(self, feature_map):
        """
        Compute the covariance minimization loss.

        Args:
        - feature_map (torch.Tensor): Shape (B, C, T), where:
            B = Batch size,
            C = Number of channels,
            T = Time series length.

        Returns:
        - loss (torch.Tensor): Scalar loss value.
        """
        B, C, T = feature_map.shape

        # Step 1: Mean-center each channel
        # feature_map_centered = feature_map - feature_map.mean(dim=-1, keepdim=True)
        mean_centered = feature_map - feature_map.mean(dim=-1, keepdims=True)        
        std_dev = torch.std(feature_map, dim=-1, unbiased=False, keepdim=True)
        std_dev = torch.where(std_dev == 0, torch.tensor(1e-6, device=std_dev.device), std_dev)
        
        norm_feature_maps = mean_centered / std_dev
        
        corr_matrix = torch.bmm(norm_feature_maps, norm_feature_maps.transpose(1, 2)) / (T - 1)
        corr_matrix = (corr_matrix + 1) / 2

        norm_per_instance = torch.norm(corr_matrix, p=2, dim=-1)
        
        # print('norm per instance shape: ', norm_per_instance.shape)
        # norm_per_channel = torch.norm(norm_per_instance, p=1, dim=1)
        # norm_per_channel = torch.sum(norm_per_instance, dim=1) / C
        norm_per_channel = torch.norm(norm_per_instance, p=1, dim=-1)
                
        # Sum over all channels and batch
        cov_loss = torch.sum(norm_per_channel)



        print('corr_matrix: ', corr_matrix[0])
        # Step 2: Compute Covariance Matrix
        # cov_matrices = torch.bmm(feature_map_centered, feature_map_centered.transpose(1, 2)) / (T - 1)

        # Step 3: Average covariance over batch
        # avg_cov_matrix = cov_matrices.mean(dim=0)  # Shape: (C, C)

        # Step 4: Compute Frobenius norm of covariance matrix
        # loss = torch.norm(avg_cov_matrix, p='fro') ** 2 

        return cov_loss  
    
class InstanceFeatureSparseLoss(torch.nn.Module):
    """
    Instance-Wise Feature Sparse Regularization Loss using L2,1-norm.
    Encourages instance-dependent sparsity in feature maps to reduce redundancy.
    """
    def __init__(self, lambda_reg=0.01):
        """
        Args:
        - lambda_reg (float): Regularization coefficient to control sparsity.
        """
        super(InstanceFeatureSparseLoss, self).__init__()
        self.lambda_reg = lambda_reg

    def forward(self, feature_maps):
        """
        Compute the instance feature sparsity loss.

        Args:
        - feature_maps (torch.Tensor): Shape (B, C, T)
          where:
            B = Batch size,
            C = Number of feature channels,
            T = Time series length.
        
        Returns:
        - loss (torch.Tensor): Scalar sparsity loss value.
        """
        B, C, T = feature_maps.shape  # Extract batch size, channels, and time series length
        
        # Compute L2-norm along the time dimension (T)
        # norm_per_instance = torch.sqrt(torch.sum(feature_maps**2, dim=2) + 1e-6)
        norm_per_instance = torch.norm(feature_maps, p=2, dim=-1)
        
        # print('norm per instance shape: ', norm_per_instance.shape)
        # norm_per_channel = torch.norm(norm_per_instance, p=1, dim=1)
        # norm_per_channel = torch.sum(norm_per_instance, dim=1) / C
        norm_per_channel = torch.norm(norm_per_instance, p=1, dim=-1)
                
        # Sum over all channels and batch
        sparsity_loss = torch.sum(norm_per_channel)
        
        return self.lambda_reg * sparsity_loss

class CovarianceLoss2(torch.nn.Module):
    """
    Covariance Regularization Loss.
    Encourages feature independence by minimizing covariance between feature channels.
    """
    def __init__(self, threshold):
        super(CovarianceLoss2, self).__init__()
        self.threshold = threshold
        
    def forward(self, feature_map):
        b, c, ts_len = feature_map.shape
        activation_strength = torch.norm(feature_map, p=1, dim=-1) #(B, C)
        mean_activation = activation_strength.mean(dim=-1, keepdims=True) #(B, 1)
        activation_mask = (activation_strength > mean_activation).float()
        
        mean_centered = feature_map - feature_map.mean(dim=-1, keepdims=True)
        
        std_dev = torch.std(feature_map, dim=-1, unbiased=False, keepdim=True)
        std_dev = torch.where(std_dev == 0, torch.tensor(1e-6, device=std_dev.device), std_dev)
        
        norm_feature_maps = mean_centered / std_dev
        
        corr_matrix = torch.bmm(norm_feature_maps, norm_feature_maps.transpose(1, 2)) / (ts_len - 1)
        corr_matrix = (corr_matrix + 1) / 2
        corr_matrix *= activation_mask.unsqueeze(-1)
        
        similarity_mask = corr_matrix > self.threshold
        
        corr_matrix = corr_matrix * similarity_mask                
        
        identity_matrix = torch.eye(c, device=corr_matrix.device).unsqueeze(0).expand(b, -1, -1)
        corr_matrix = corr_matrix * (1 - identity_matrix)
        # print('\n\n aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \n\n')
        # print('correlation matrix shape: ', corr_matrix.shape)
        
        norm_per_instance = torch.norm(corr_matrix, p=2, dim=-1)
        
        # print('norm per instance shape: ', norm_per_instance.shape)
        # norm_per_channel = torch.norm(norm_per_instance, p=1, dim=1)
        # norm_per_channel = torch.sum(norm_per_instance, dim=1) / C
        norm_per_channel = torch.norm(norm_per_instance, p=1, dim=-1)
                
        # Sum over all channels and batch
        corr_loss = torch.sum(norm_per_channel)
        print(corr_matrix)
        return corr_loss

class InstanceWiseSparsityInference(nn.Module):
    
    def __init__(self, alpha=0.2, beta=0.5):
        super(InstanceWiseSparsityInference, self).__init__()
        self.alpha = alpha
        self.beta = beta
        
    def compute_feature_energy(self, feature_maps):
        # print('111 feature maps shape: ', feature_maps.shape)
        return torch.sqrt(torch.sum(feature_maps ** 2, dim=2) + 1e-6)
          
    def compute_cv(self, feature_energy):
        mean_energy = feature_energy.mean(dim=1, keepdim=True)
        std_energy = feature_energy.std(dim=1, unbiased=True, keepdim=True) 
        cv = std_energy / (mean_energy + 1e-6)  
        return cv.mean()
    
    def forward(self, feature_maps):
        if feature_maps.shape[1] > 96:        
            feature_energy = self.compute_feature_energy(feature_maps[:96])
            mean_energy = feature_energy.mean(dim=1, keepdim=True) 
            pruning_mask = (feature_energy > self.beta * mean_energy).float()  
            pruned_feature_maps = feature_maps[:96] * pruning_mask.unsqueeze(-1)
            # pruned_feature_maps = feature_maps * pruning_mask.unsqueeze(-1)

        else:
            feature_energy = self.compute_feature_energy(feature_maps)
            mean_energy = feature_energy.mean(dim=1, keepdim=True) 
            pruning_mask = (feature_energy > self.beta * mean_energy).float()  
            pruned_feature_maps = feature_maps * pruning_mask.unsqueeze(-1)

        # def mask_gradients(grad):
        #   return grad * pruning_mask.unsqueeze(-1)

        # pruned_feature_maps.hook_register(mask_gradients)
            
        return pruned_feature_maps, pruning_mask


class FeatureMapPruner:
    
    def __init__(self, threshold=0.8):
        self.threshold = threshold
       
        
    def compute_covariance(self, feature_maps):
        b, c, ts_len = feature_maps.shape
        print('\n\n\n\n')
        print('Feature map shape: ', feature_maps.shape)
        activation_strength = torch.norm(feature_maps, p=1, dim=-1) #(B, C)
        print('acivation_strength shape: ', activation_strength.shape)
        
        mean_activation = activation_strength.mean(dim=-1, keepdims=True) #(B, 1)
        print('mean_activation shape: ', mean_activation.shape)
        
        activation_mask = (activation_strength > mean_activation).float()
        
        # masked_feature_maps = feature_maps * activation_mask.unsqueeze(-1)  

        mean_centered = feature_maps - feature_maps.mean(dim=-1, keepdims=True)
        print('mean centered shape: ', mean_centered.shape)
        
        std_dev = torch.std(feature_maps, dim=-1, unbiased=False, keepdim=True)
        std_dev = torch.where(std_dev == 0, torch.tensor(1e-6, device=std_dev.device), std_dev)
        
        norm_feature_maps = mean_centered / std_dev
        
        corr_matrix = torch.bmm(norm_feature_maps, norm_feature_maps.transpose(1, 2)) / (ts_len - 1)
        corr_matrix = (corr_matrix + 1) / 2
        corr_matrix *= activation_mask.unsqueeze(-1)
        print('activation mask shape: ', activation_mask.unsqueeze(-1).shape)
        print('activation_mask: ', activation_mask[0])
        
        # # 1️⃣ Create similarity mask (B, C, C) where 1 = correlated, 0 = not
        # similarity_mask = (corr_matrix >= self.threshold).float()

        # # 2️⃣ Ensure we only consider features with high activation
        # activation_mask = (activation_strength > self.threshold).float()  # (B, C)
        # activation_mask = activation_mask.unsqueeze(-1) * activation_mask.unsqueeze(1)  # (B, C, C)

        # # 3️⃣ Combine masks: Only features that are correlated & have high activation
        # similarity_mask = similarity_mask * activation_mask
        
        
        return corr_matrix

    
def plot_1v1_perf(res_df, p_value, x_y_lim=0, acc_base=1, co_col='ens_co', base_col='base_ens', 
                  xlabel='base perf.', ylabel='CoTrain perf.', legend_base='Base', legend_co='CoTrain', 
                  title='Title', show_p_value=True, file_name=None):
    # Define the points for the diagonal line
    x_line = [x_y_lim, acc_base]
    y_line = [x_y_lim, acc_base]

    # Define points for scatter plot
    x_scatter = res_df[base_col].tolist() 
    y_scatter = res_df[co_col].tolist() 

    x_above = np.array([x for x, y in zip(x_scatter, y_scatter) if y > x])
    y_above = np.array([y for x, y in zip(x_scatter, y_scatter) if y > x])

    x_same = np.array([x for x, y in zip(x_scatter, y_scatter) if y == x])
    y_same = np.array([y for x, y in zip(x_scatter, y_scatter) if y == x])

    x_below = np.array([x for x, y in zip(x_scatter, y_scatter) if y < x])
    y_below = np.array([y for x, y in zip(x_scatter, y_scatter) if y < x])

    # Plot the diagonal line
    plt.plot(x_line, y_line,  color='blue')
    num_wins = res_df[res_df[co_col] > res_df[base_col]].shape[0]
    num_ties = res_df[res_df[co_col] == res_df[base_col]].shape[0]
    num_losses = res_df[res_df[co_col] < res_df[base_col]].shape[0]

    # Add p-value to the legend
    if show_p_value:
        plt.scatter([], [], label=f'p-value: {p_value}', color='none')  # Invisible point for legend entry

    # Plot the scatter points
    plt.scatter(x_below, y_below, label=f'{legend_base} Wins - ' + str(num_losses), color='green')
    plt.scatter(x_same, y_same, label='Equal - ' + str(num_ties), color='orange')
    plt.scatter(x_above, y_above, label=f'{legend_co} Wins - ' + str(num_wins), color='red')

    # # Set axis limits
    plt.xlim(x_y_lim, acc_base)
    plt.ylim(x_y_lim, acc_base)

    plt.xlabel(xlabel, fontsize=14)  
    plt.ylabel(ylabel, fontsize=14)  
    plt.title(title, fontsize=16)    


    # Add a legend
    plt.legend()
    if file_name:
        plt.savefig(f'{file_name}.pdf', bbox_inches='tight', pad_inches=0)

    plt.show()
    
    
    