import torch
import torch.nn as nn
import numpy as np
from torch_geometric.graphgym.config import cfg

# Metrics Config
SMOOTH = 1e-8
l1_loss = nn.L1Loss(reduction="none")
mse_loss = nn.MSELoss(reduction="none")

def compute_loss(pred, true):
    MAE = l1_loss(pred, true)
    MSE = mse_loss(pred, true)
    return MAE, MSE

def get_volume(A):
    return ((4.0 / 3.0) * np.pi * torch.sqrt(torch.det(A)))

def get_error_volume(pred, true):
    volume_true = get_volume(pred)
    volume_pred = get_volume(true)
    return (torch.abs(volume_true - volume_pred)/(volume_true+SMOOTH))

def get_KL(pred, true):
    p = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(true.shape[0],3, device=true.device), true)
    q = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(pred.shape[0],3, device=true.device), pred)

    return torch.distributions.kl.kl_divergence(p, q)

def get_similarity_index(pred, true):
    r12_num = 2 ** (3 / 2) * torch.linalg.det(torch.linalg.inv(true) @ torch.linalg.inv(pred)) ** (1 / 4)
    r12_den = torch.linalg.det(torch.linalg.inv(true) + torch.linalg.inv(pred)) ** (1 / 2)
    return 100*(1-r12_num/r12_den)

def iou_pytorch3D(outputs: torch.Tensor, labels: torch.Tensor):

    intersection = (outputs & labels).float().sum((1, 2, 3))  # Will be zero if Truth=0 or Prediction=0
    
    union = (outputs | labels).float().sum((1, 2, 3))         # Will be zero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our division to avoid 0/0
    
    return iou

def get_ellipsoids(covariance_matrices):
    num_points = 64
    device = covariance_matrices.device
    num_matrices = covariance_matrices.shape[0]

    # Generate grid of points (x, y) centered at (0, 0)
    x = torch.linspace(-1, 1, num_points, device = device)
    y = torch.linspace(-1, 1, num_points, device = device)
    z = torch.linspace(-1, 1, num_points, device = device)
    x_grid, y_grid, z_grid = torch.meshgrid(x, y, z)
    points = torch.stack((x_grid, y_grid, z_grid), axis=-1)

    
    # Inverse of the covariance matrices for all points in the batch
    covariance_invs = torch.linalg.inv(covariance_matrices)
    
    # Reshape the points to a 2D array for matrix multiplication
    reshaped_points = points.reshape(-1, 3)
    
    mult = reshaped_points.unsqueeze(0) @ covariance_invs.unsqueeze(1)
    mahalanobis_distances = torch.sqrt(torch.sum(mult.squeeze(1) * reshaped_points.unsqueeze(0).repeat(num_matrices,1,1), axis = -1))


    # Reshape the resulting distances back to the grid shape for each matrix in the batch
    mahalanobis_maps = mahalanobis_distances.reshape(num_matrices, num_points, num_points, num_points)
    ellipsoid_mask = (mahalanobis_maps < 1)

    return ellipsoid_mask


def compute_3D_IoU(pred,true):
    device = true.device
    matrix_norm_pred = torch.linalg.matrix_norm(pred)
    matrix_norm_true = torch.linalg.matrix_norm(true)

    matrix_norm = torch.where(matrix_norm_pred>matrix_norm_true, matrix_norm_pred, matrix_norm_true).unsqueeze(-1).unsqueeze(-1)

    true_norm = true/matrix_norm
    pred_norm = pred/matrix_norm

    # Generate the ellipsoid mesh
    ellipsoid_mesh_true = get_ellipsoids(true_norm)
    ellipsoid_mesh_pred = get_ellipsoids(pred_norm)
    
        

    iou = iou_pytorch3D(ellipsoid_mesh_pred,ellipsoid_mesh_true)

    return iou

def compute_metrics_and_logging(pred, true, mae, mse, loss, lr, time_used, logger, test_metrics=False):
    
    if cfg.dataset.name == "ADP":
        if test_metrics:
            logger.update_stats(true = true.to("cpu"),
                                pred = pred.to("cpu"),
                                loss = loss.mean().item(),
                                MAE = mae.mean().item(),
                                MSE = mse.mean().item(),
                                lr = lr,
                                time_used = time_used,
                                params = cfg.params_count,
                                dataset_name = cfg.dataset.name,
                                volume_percentage_error = get_error_volume(pred,true).mean().item(),
                                iou = compute_3D_IoU(pred,true).mean().item(),
                                similarity_index = get_similarity_index(pred, true).mean().item(),
                            )
        else:
            logger.update_stats(true = true.to("cpu"),
                            pred = pred.to("cpu"),
                            loss = loss.mean().item(),
                            MAE = mae.mean().item(),
                            MSE = mse.mean().item(),
                            lr = lr,
                            volume_percentage_error = get_error_volume(pred,true).mean().item(),
                            similarity_index = get_similarity_index(pred, true).mean().item(),
                            time_used = time_used,
                            params = cfg.params_count,
                            dataset_name = cfg.dataset.name,
                        )
    else:
        logger.update_stats(true = true.to("cpu"),
                            pred = pred.to("cpu"),
                            loss = loss.mean().item(),
                            MAE = mae.mean().item(),
                            MSE = mse.mean().item(),
                            lr = lr,
                            time_used = time_used,
                            params = cfg.params_count,
                            dataset_name = cfg.dataset.name
                        )