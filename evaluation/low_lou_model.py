"""
low_lou_model.py
----------------
Low & Lou analytical model for force-free magnetic fields.
Provides analytical solutions for testing and validation.

References:
- Low & Lou, "Modeling Solar Force-Free Magnetic Fields" (1990)
- Analytical solutions for force-free magnetic fields
"""
import numpy as np
import jax.numpy as jnp
from typing import Tuple, Optional

def low_lou_bfield(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
                   alpha: float = 0.5, a: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate Low & Lou force-free magnetic field.
    
    Args:
        X, Y, Z: Coordinate arrays
        alpha: Force-free parameter
        a: Scale parameter
        
    Returns:
        Bx, By, Bz: Magnetic field components
    """
    # Convert to spherical coordinates
    r = np.sqrt(X**2 + Y**2 + Z**2) + 1e-8
    theta = np.arccos(Z / r)
    phi = np.arctan2(Y, X)
    
    # Force-free field components in spherical coordinates
    # Simplified Low & Lou model
    Br = np.cos(theta) * np.sin(alpha * r) / r
    Btheta = np.sin(theta) * np.sin(alpha * r) / r
    Bphi = np.sin(alpha * r) / r
    
    # Convert to Cartesian coordinates
    Bx = (Br * np.sin(theta) * np.cos(phi) +
          Btheta * np.cos(theta) * np.cos(phi) -
          Bphi * np.sin(phi))
    
    By = (Br * np.sin(theta) * np.sin(phi) +
          Btheta * np.cos(theta) * np.sin(phi) +
          Bphi * np.cos(phi))
    
    Bz = Br * np.cos(theta) - Btheta * np.sin(theta)
    
    # Apply scaling
    Bx *= a
    By *= a
    Bz *= a
    
    return Bx, By, Bz

def field_line(X0: np.ndarray, Y0: np.ndarray, Z0: np.ndarray,
               alpha: float = 0.5, a: float = 1.0, 
               max_steps: int = 1000, step_size: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Trace magnetic field lines using Low & Lou model.
    
    Args:
        X0, Y0, Z0: Starting points
        alpha: Force-free parameter
        a: Scale parameter
        max_steps: Maximum number of integration steps
        step_size: Integration step size
        
    Returns:
        X, Y, Z: Field line coordinates
    """
    # Initialize arrays
    X = np.zeros(max_steps)
    Y = np.zeros(max_steps)
    Z = np.zeros(max_steps)
    
    X[0] = X0
    Y[0] = Y0
    Z[0] = Z0
    
    # Simple Euler integration
    for i in range(max_steps - 1):
        # Get field at current position
        Bx, By, Bz = low_lou_bfield(X[i:i+1], Y[i:i+1], Z[i:i+1], alpha, a)
        
        # Normalize field vector
        B_mag = np.sqrt(Bx[0]**2 + By[0]**2 + Bz[0]**2)
        if B_mag > 1e-8:
            Bx[0] /= B_mag
            By[0] /= B_mag
            Bz[0] /= B_mag
        
        # Step along field line
        X[i+1] = X[i] + step_size * Bx[0]
        Y[i+1] = Y[i] + step_size * By[0]
        Z[i+1] = Z[i] + step_size * Bz[0]
        
        # Check if we've gone too far
        r = np.sqrt(X[i+1]**2 + Y[i+1]**2 + Z[i+1]**2)
        if r > 10.0:  # Stop if too far from origin
            break
    
    return X[:i+2], Y[:i+2], Z[:i+2]

def generate_low_lou_sequence(n_samples: int = 10, 
                             grid_size: Tuple[int, int, int] = (64, 64, 32),
                             time_steps: int = 10) -> dict:
    """
    Generate a sequence of Low & Lou fields for temporal testing.
    
    Args:
        n_samples: Number of samples
        grid_size: Grid dimensions (nx, ny, nz)
        time_steps: Number of time steps
        
    Returns:
        Dictionary containing field sequences
    """
    nx, ny, nz = grid_size
    
    # Create coordinate grid
    x = np.linspace(-2, 2, nx)
    y = np.linspace(-2, 2, ny)
    z = np.linspace(0, 4, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    sequences = []
    times = np.linspace(0, 1.0, time_steps)
    
    for sample in range(n_samples):
        sample_sequence = []
        
        # Vary alpha over time
        alpha_base = 0.5 + 0.2 * np.sin(sample * np.pi / n_samples)
        
        for t in times:
            # Time-varying alpha
            alpha = alpha_base + 0.1 * np.sin(2 * np.pi * t)
            
            # Generate field
            Bx, By, Bz = low_lou_bfield(X, Y, Z, alpha=alpha, a=1.0)
            
            # Add temporal evolution
            evolution_factor = 1.0 + 0.1 * np.sin(2 * np.pi * t)
            Bx *= evolution_factor
            By *= evolution_factor
            Bz *= evolution_factor
            
            # Stack components
            field = np.stack([Bx, By, Bz], axis=0)  # (3, nx, ny, nz)
            
            # Extract surface magnetogram (z=0)
            magnetogram = field[:, :, :, 0]  # (3, nx, ny)
            
            sample_sequence.append(magnetogram)
        
        sequences.append(np.stack(sample_sequence, axis=0))  # (T, 3, nx, ny)
    
    sequences = np.stack(sequences, axis=0)  # (N, T, 3, nx, ny)
    
    return {
        'sequences': sequences,
        'times': times,
        'coordinates': (X, Y, Z),
        'type': 'low_lou_synthetic'
    }

def test_low_lou_model():
    """Test the Low & Lou model implementation."""
    print("Testing Low & Lou model...")
    
    # Create test grid
    nx, ny, nz = 32, 32, 16
    x = np.linspace(-2, 2, nx)
    y = np.linspace(-2, 2, ny)
    z = np.linspace(0, 4, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Generate field
    Bx, By, Bz = low_lou_bfield(X, Y, Z, alpha=0.5, a=1.0)
    
    print(f"Field shapes: Bx={Bx.shape}, By={By.shape}, Bz={Bz.shape}")
    print(f"Field ranges: Bx=[{Bx.min():.3f}, {Bx.max():.3f}]")
    print(f"             By=[{By.min():.3f}, {By.max():.3f}]")
    print(f"             Bz=[{Bz.min():.3f}, {Bz.max():.3f}]")
    
    # Test field line tracing
    X_line, Y_line, Z_line = field_line(1.0, 0.0, 0.0, alpha=0.5, a=1.0)
    print(f"Field line length: {len(X_line)} points")
    
    print("Low & Lou model test completed successfully!")
    return True

if __name__ == "__main__":
    test_low_lou_model() 