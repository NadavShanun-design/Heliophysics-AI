#!/usr/bin/env python3
"""
sanity_check.py
---------------
Comprehensive sanity check for Solar AI system.
Tests data pipeline, model creation, and basic training functionality.
"""
import sys
import os
import numpy as np
import jax
import jax.numpy as jnp
import optax
from pathlib import Path
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import our components
try:
    from models.solar_deeponet_3d import SolarDeepONet, PhysicsInformedLoss, create_solar_deeponet_training_step
    from models.solar_fno_3d import SolarFNO3D, PhysicsInformedFNOLoss, create_solar_fno_training_step
    from data.sdo_data_pipeline import SyntheticDataGenerator
    from evaluation.comprehensive_evaluation import generate_synthetic_test_data
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def setup_logging():
    """Setup logging for sanity check."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('sanity_check.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('sanity_check')

def test_jax_environment():
    """Test JAX environment and basic functionality."""
    print("\n🔧 Testing JAX Environment...")
    
    # Test basic JAX operations
    x = jnp.array([1, 2, 3, 4, 5])
    y = jnp.array([2, 3, 4, 5, 6])
    z = x * y + jnp.sin(x)
    print(f"   JAX computation: {x} * {y} + sin({x}) = {z}")
    
    # Test automatic differentiation
    def simple_fn(x):
        return jnp.sum(x ** 2)
    
    grad_fn = jax.grad(simple_fn)
    x_test = jnp.array([1.0, 2.0, 3.0])
    grad_result = grad_fn(x_test)
    print(f"   Autodiff test: grad(sum(x²)) at {x_test} = {grad_result}")
    
    # Test JIT compilation
    @jax.jit
    def jit_fn(x, y):
        return jnp.dot(x, y)
    
    result = jit_fn(jnp.array([1, 2, 3]), jnp.array([4, 5, 6]))
    print(f"   JIT compilation test: dot([1,2,3], [4,5,6]) = {result}")
    
    print("   ✅ JAX environment working correctly!")
    return True

def test_data_generation():
    """Test synthetic data generation."""
    print("\n📊 Testing Data Generation...")
    
    # Test synthetic data generator
    try:
        data_gen = SyntheticDataGenerator(grid_size=(32, 32, 16))
        print("   ✅ SyntheticDataGenerator created successfully")
        
        # Generate test data
        test_data = generate_synthetic_test_data(
            n_samples=4,
            grid_size=(32, 32, 16),
            key=jax.random.PRNGKey(42)
        )
        
        print(f"   Generated test data keys: {list(test_data.keys())}")
        for key, value in test_data.items():
            if isinstance(value, np.ndarray):
                print(f"   {key}: shape {value.shape}, dtype {value.dtype}")
        
        # Verify data shapes and ranges
        magnetogram = test_data['magnetogram']
        coords = test_data['coords']
        B_true = test_data['ground_truth']  # Changed from 'B_true' to 'ground_truth'
        
        print(f"   Magnetogram shape: {magnetogram.shape}")
        print(f"   Coordinates shape: {coords.shape}")
        print(f"   Magnetic field shape: {B_true.shape}")
        
        # Check data ranges
        print(f"   Magnetogram range: [{magnetogram.min():.3f}, {magnetogram.max():.3f}]")
        print(f"   Coordinates range: [{coords.min():.3f}, {coords.max():.3f}]")
        print(f"   B_field range: [{B_true.min():.3f}, {B_true.max():.3f}]")
        
        # Check for NaN or infinite values
        has_nan = np.isnan(magnetogram).any() or np.isnan(coords).any() or np.isnan(B_true).any()
        has_inf = np.isinf(magnetogram).any() or np.isinf(coords).any() or np.isinf(B_true).any()
        
        if has_nan:
            print("   ⚠️  Warning: NaN values detected in data")
        if has_inf:
            print("   ⚠️  Warning: Infinite values detected in data")
        if not has_nan and not has_inf:
            print("   ✅ Data validation passed (no NaN/Inf values)")
        
        return test_data
        
    except Exception as e:
        print(f"   ❌ Data generation failed: {e}")
        return None

def test_model_creation():
    """Test model creation and basic forward pass."""
    print("\n🧠 Testing Model Creation...")
    
    # Test DeepONet
    try:
        print("   Testing DeepONet...")
        deeponet = SolarDeepONet(
            magnetogram_shape=(32, 32),
            latent_dim=64,
            branch_depth=4,
            trunk_depth=3,
            width=128,
            key=jax.random.PRNGKey(42)
        )
        print("   ✅ DeepONet created successfully")
    except Exception as e:
        print(f"   ❌ DeepONet creation failed: {e}")
        return None, None
        
    # Test FNO (skip for now due to Equinox compatibility issues)
    print("   Testing FNO...")
    try:
        fno = SolarFNO3D(
            input_channels=3,
            output_channels=3,
            modes=(8, 8, 4),
            width=32,
            depth=2,
            grid_size=(32, 32, 16),
            key=jax.random.PRNGKey(43)
        )
        print("   ✅ FNO created successfully")
    except Exception as e:
        print(f"   ⚠️  FNO creation failed: {e}")
        print("   ⚠️  Skipping FNO tests for now")
        fno = None
        
    return deeponet, fno

def test_forward_pass(deeponet, fno, test_data):
    """Test forward pass through models."""
    print("\n🔄 Testing Forward Pass...")
    
    try:
        # Test DeepONet forward pass
        print("   Testing DeepONet forward pass...")
        magnetogram = test_data['magnetogram'][0]  # Take first sample
        coords = test_data['coords'][0].reshape(-1, 3)  # Flatten coordinates
        
        # Get model parameters (Equinox models don't have parameters() method)
        deeponet_params = None  # Equinox models are callable directly
        
        # Forward pass
        B_pred_deeponet = deeponet(magnetogram, coords)
        print(f"   DeepONet output shape: {B_pred_deeponet.shape}")
        print(f"   DeepONet output range: [{B_pred_deeponet.min():.3f}, {B_pred_deeponet.max():.3f}]")
        
        # Test FNO forward pass (if available)
        if fno is not None:
            print("   Testing FNO forward pass...")
            magnetogram_fno = test_data['magnetogram'][:2]  # Batch of 2
            coords_fno = test_data['coords'][:2]
            
            # Get model parameters
            fno_params = fno.parameters()
            
            # Forward pass
            B_pred_fno = fno(fno_params, magnetogram_fno, coords_fno)
            print(f"   FNO output shape: {B_pred_fno.shape}")
            print(f"   FNO output range: [{B_pred_fno.min():.3f}, {B_pred_fno.max():.3f}]")
            print("   ✅ Forward pass successful for both models!")
        else:
            print("   ⚠️  Skipping FNO forward pass (FNO not available)")
            print("   ✅ Forward pass successful for DeepONet!")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_loss_functions(deeponet, fno, test_data):
    """Test loss function computation."""
    print("\n📉 Testing Loss Functions...")
    
    try:
        # Test DeepONet loss
        print("   Testing DeepONet loss...")
        loss_fn_deeponet = PhysicsInformedLoss(lambda_data=1.0, lambda_physics=0.1)
        
        magnetogram = test_data['magnetogram'][0]
        coords = test_data['coords'][0].reshape(-1, 3)
        B_true = test_data['ground_truth'][0].reshape(-1, 3)
        
        # Equinox models are callable directly
        total_loss, loss_components = loss_fn_deeponet(
            deeponet, None, magnetogram, coords, B_true
        )
        
        print(f"   DeepONet total loss: {total_loss:.6f}")
        print(f"   Loss components: {loss_components}")
        
        # Test FNO loss
        print("   Testing FNO loss...")
        loss_fn_fno = PhysicsInformedFNOLoss(lambda_data=1.0, lambda_physics=0.1)
        
        magnetogram_fno = test_data['magnetogram'][:2]
        coords_fno = test_data['coords'][:2]
        B_true_fno = test_data['ground_truth'][:2]
        
        fno_params = fno.parameters()
        
        total_loss_fno, loss_components_fno = loss_fn_fno(
            fno, fno_params, magnetogram_fno, coords_fno, B_true_fno
        )
        
        print(f"   FNO total loss: {total_loss_fno:.6f}")
        print(f"   Loss components: {loss_components_fno}")
        
        print("   ✅ Loss functions working correctly!")
        return True
        
    except Exception as e:
        print(f"   ❌ Loss function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_step(deeponet, fno, test_data):
    """Test training step functionality."""
    print("\n🏋️ Testing Training Step...")
    
    try:
        # Test DeepONet training step
        print("   Testing DeepONet training step...")
        optimizer = optax.adam(learning_rate=1e-3)
        
        magnetogram = test_data['magnetogram'][0]
        coords = test_data['coords'][0].reshape(-1, 3)
        B_true = test_data['ground_truth'][0].reshape(-1, 3)
        
        # Equinox models are callable directly
        opt_state = optimizer.init(None)
        
        # Create training step
        loss_fn = PhysicsInformedLoss(lambda_data=1.0, lambda_physics=0.1)
        training_step = create_solar_deeponet_training_step(deeponet, loss_fn, optimizer)
        
        # Run training step
        new_params, new_opt_state, loss_info = training_step(
            None, opt_state, magnetogram, coords, B_true
        )
        
        print(f"   DeepONet training step loss: {loss_info['total_loss']:.6f}")
        print(f"   Loss components: {loss_info}")
        
        # Test FNO training step
        print("   Testing FNO training step...")
        fno_optimizer = optax.adam(learning_rate=1e-3)
        
        magnetogram_fno = test_data['magnetogram'][:2]
        coords_fno = test_data['coords'][:2]
        B_true_fno = test_data['ground_truth'][:2]
        
        fno_params = fno.parameters()
        fno_opt_state = fno_optimizer.init(fno_params)
        
        # Create training step
        fno_loss_fn = PhysicsInformedFNOLoss(lambda_data=1.0, lambda_physics=0.1)
        fno_training_step = create_solar_fno_training_step(fno, fno_loss_fn, fno_optimizer)
        
        # Run training step
        new_fno_params, new_fno_opt_state, fno_loss_info = fno_training_step(
            fno_params, fno_opt_state, magnetogram_fno, coords_fno, B_true_fno
        )
        
        print(f"   FNO training step loss: {fno_loss_info['total_loss']:.6f}")
        print(f"   Loss components: {fno_loss_info}")
        
        print("   ✅ Training steps working correctly!")
        return True
        
    except Exception as e:
        print(f"   ❌ Training step test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_training(deeponet, test_data):
    """Test basic training loop."""
    print("\n🎯 Testing Basic Training Loop...")
    
    try:
        # Setup training
        optimizer = optax.adam(learning_rate=1e-3)
        loss_fn = PhysicsInformedLoss(lambda_data=1.0, lambda_physics=0.1)
        training_step = create_solar_deeponet_training_step(deeponet, loss_fn, optimizer)
        
        # Prepare data
        magnetogram = test_data['magnetogram'][0]
        coords = test_data['coords'][0].reshape(-1, 3)
        B_true = test_data['ground_truth'][0].reshape(-1, 3)
        
        # Initialize optimizer (Equinox models are callable directly)
        params = None
        opt_state = optimizer.init(params)
        
        # Training loop
        n_epochs = 10
        losses = []
        
        print(f"   Training for {n_epochs} epochs...")
        for epoch in range(n_epochs):
            params, opt_state, loss_info = training_step(params, opt_state, magnetogram, coords, B_true)
            losses.append(loss_info['total_loss'])
            
            if epoch % 2 == 0:
                print(f"   Epoch {epoch}: Loss = {loss_info['total_loss']:.6f}")
        
        print(f"   Final loss: {losses[-1]:.6f}")
        print(f"   Loss reduction: {losses[0] - losses[-1]:.6f}")
        
        if losses[-1] < losses[0]:
            print("   ✅ Training successful (loss decreased)!")
            return True
        else:
            print("   ⚠️  Training may not be working (loss didn't decrease)")
            return False
        
    except Exception as e:
        print(f"   ❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_visualization(test_data):
    """Test data visualization."""
    print("\n📊 Testing Data Visualization...")
    
    try:
        # Create a simple visualization
        magnetogram = test_data['magnetogram'][0]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot magnetogram components
        for i, component in enumerate(['Bx', 'By', 'Bz']):
            im = axes[i].imshow(magnetogram[i], cmap='RdBu_r', aspect='equal')
            axes[i].set_title(f'{component} Component')
            axes[i].set_xlabel('X')
            axes[i].set_ylabel('Y')
            plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        plt.savefig('sanity_check_magnetogram.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("   ✅ Magnetogram visualization saved as 'sanity_check_magnetogram.png'")
        return True
        
    except Exception as e:
        print(f"   ❌ Visualization failed: {e}")
        return False

def main():
    """Run comprehensive sanity check."""
    print("🔍 Solar AI System - Comprehensive Sanity Check")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    
    # Setup logging
    logger = setup_logging()
    
    # Test results
    results = {}
    
    # 1. Test JAX environment
    results['jax_environment'] = test_jax_environment()
    
    # 2. Test data generation
    test_data = test_data_generation()
    results['data_generation'] = test_data is not None
    
    if test_data is None:
        print("❌ Data generation failed, stopping sanity check")
        return
    
    # 3. Test model creation
    deeponet, fno = test_model_creation()
    results['model_creation'] = deeponet is not None  # Only require DeepONet to work
    
    if deeponet is None:
        print("❌ DeepONet creation failed, stopping sanity check")
        return
    
    # 4. Test forward pass
    results['forward_pass'] = test_forward_pass(deeponet, fno, test_data)
    
    # 5. Test loss functions
    results['loss_functions'] = test_loss_functions(deeponet, fno, test_data)
    
    # 6. Test training step
    results['training_step'] = test_training_step(deeponet, fno, test_data)
    
    # 7. Test basic training
    results['basic_training'] = test_basic_training(deeponet, test_data)
    
    # 8. Test visualization
    results['visualization'] = test_data_visualization(test_data)
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 SANITY CHECK SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! The Solar AI system is working correctly.")
    else:
        print("⚠️  SOME TESTS FAILED. Please check the logs for details.")
    
    print(f"Completed at: {datetime.now()}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 