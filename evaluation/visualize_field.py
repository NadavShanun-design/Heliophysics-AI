"""
visualize_field.py
-----------------
Field visualization and evaluation metrics for solar magnetic fields.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import jax.numpy as jnp

def compute_mse(pred: np.ndarray, true: np.ndarray) -> float:
    """Compute Mean Squared Error."""
    return np.mean((pred - true) ** 2)

def compute_ssim(pred: np.ndarray, true: np.ndarray) -> float:
    """Compute Structural Similarity Index."""
    # Simplified SSIM implementation
    mu_pred = np.mean(pred)
    mu_true = np.mean(true)
    
    sigma_pred = np.var(pred)
    sigma_true = np.var(true)
    sigma_pred_true = np.mean((pred - mu_pred) * (true - mu_true))
    
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    ssim = ((2 * mu_pred * mu_true + c1) * (2 * sigma_pred_true + c2)) / \
           ((mu_pred ** 2 + mu_true ** 2 + c1) * (sigma_pred + sigma_true + c2))
    
    return ssim

def compute_psnr(pred: np.ndarray, true: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    mse = compute_mse(pred, true)
    if mse == 0:
        return float('inf')
    
    max_val = np.max(true)
    psnr = 20 * np.log10(max_val / np.sqrt(mse))
    return psnr

def compute_relative_l2_error(pred: np.ndarray, true: np.ndarray) -> float:
    """Compute relative L2 error."""
    numerator = np.sqrt(np.sum((pred - true) ** 2))
    denominator = np.sqrt(np.sum(true ** 2))
    return numerator / (denominator + 1e-8)

def visualize_magnetogram(magnetogram: np.ndarray, 
                         title: str = "Magnetogram",
                         save_path: Optional[str] = None) -> None:
    """Visualize magnetogram components."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    components = ['Bx', 'By', 'Bz']
    for i, (ax, component) in enumerate(zip(axes, components)):
        im = ax.imshow(magnetogram[i], cmap='RdBu_r', aspect='equal')
        ax.set_title(f'{component} Component')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def visualize_field_lines(B_field: np.ndarray, 
                         coords: np.ndarray,
                         n_lines: int = 10,
                         save_path: Optional[str] = None) -> None:
    """Visualize magnetic field lines."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Sample starting points
    nx, ny, nz = B_field.shape[:3]
    start_points = []
    
    for _ in range(n_lines):
        x = np.random.uniform(-2, 2)
        y = np.random.uniform(-2, 2)
        z = np.random.uniform(0, 2)
        start_points.append([x, y, z])
    
    # Trace field lines (simplified)
    for start_point in start_points:
        x, y, z = start_point
        line_x, line_y, line_z = [x], [y], [z]
        
        # Simple field line tracing
        for _ in range(50):
            # Find nearest grid point
            idx_x = int((x + 2) / 4 * (nx - 1))
            idx_y = int((y + 2) / 4 * (ny - 1))
            idx_z = int(z / 4 * (nz - 1))
            
            idx_x = max(0, min(idx_x, nx - 1))
            idx_y = max(0, min(idx_y, ny - 1))
            idx_z = max(0, min(idx_z, nz - 1))
            
            # Get field direction
            Bx = B_field[idx_x, idx_y, idx_z, 0]
            By = B_field[idx_x, idx_y, idx_z, 1]
            Bz = B_field[idx_x, idx_y, idx_z, 2]
            
            # Normalize
            B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
            if B_mag > 1e-8:
                Bx /= B_mag
                By /= B_mag
                Bz /= B_mag
            
            # Step along field line
            step_size = 0.1
            x += step_size * Bx
            y += step_size * By
            z += step_size * Bz
            
            line_x.append(x)
            line_y.append(y)
            line_z.append(z)
            
            # Stop if too far
            if np.sqrt(x**2 + y**2 + z**2) > 5:
                break
        
        ax.plot(line_x, line_y, line_z, alpha=0.7)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Magnetic Field Lines')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def plot_training_history(history: Dict[str, list], 
                         save_path: Optional[str] = None) -> None:
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    if 'train_loss' in history and 'val_loss' in history:
        axes[0].plot(history['train_loss'], label='Train Loss')
        axes[0].plot(history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True)
    
    # Metrics plot
    if 'train_metrics' in history and 'val_metrics' in history:
        metrics = list(history['train_metrics'][0].keys()) if history['train_metrics'] else []
        for metric in metrics:
            if metric != 'total_loss':
                train_vals = [m[metric] for m in history['train_metrics']]
                val_vals = [m[metric] for m in history['val_metrics']]
                axes[1].plot(train_vals, label=f'Train {metric}')
                axes[1].plot(val_vals, label=f'Val {metric}')
        
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Metric Value')
        axes[1].set_title('Training Metrics')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def compare_predictions(pred: np.ndarray, 
                       true: np.ndarray,
                       save_path: Optional[str] = None) -> None:
    """Compare predicted vs true magnetic fields."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    components = ['Bx', 'By', 'Bz']
    
    for i, component in enumerate(components):
        # True field
        im1 = axes[0, i].imshow(true[..., i], cmap='RdBu_r', aspect='equal')
        axes[0, i].set_title(f'True {component}')
        axes[0, i].set_xlabel('X')
        axes[0, i].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[0, i])
        
        # Predicted field
        im2 = axes[1, i].imshow(pred[..., i], cmap='RdBu_r', aspect='equal')
        axes[1, i].set_title(f'Predicted {component}')
        axes[1, i].set_xlabel('X')
        axes[1, i].set_ylabel('Y')
        plt.colorbar(im2, ax=axes[1, i])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def test_visualization():
    """Test visualization functions."""
    print("Testing visualization functions...")
    
    # Create test data
    nx, ny, nz = 32, 32, 16
    magnetogram = np.random.randn(3, nx, ny)
    B_field = np.random.randn(nx, ny, nz, 3)
    coords = np.random.randn(nx, ny, nz, 3)
    
    # Test metrics
    pred = np.random.randn(64, 64, 3)
    true = np.random.randn(64, 64, 3)
    
    mse = compute_mse(pred, true)
    ssim = compute_ssim(pred, true)
    psnr = compute_psnr(pred, true)
    rel_error = compute_relative_l2_error(pred, true)
    
    print(f"MSE: {mse:.6f}")
    print(f"SSIM: {ssim:.6f}")
    print(f"PSNR: {psnr:.6f}")
    print(f"Relative L2 Error: {rel_error:.6f}")
    
    # Test visualization
    visualize_magnetogram(magnetogram, save_path='test_magnetogram.png')
    print("Magnetogram visualization saved as 'test_magnetogram.png'")
    
    print("Visualization test completed successfully!")
    return True

if __name__ == "__main__":
    test_visualization() 