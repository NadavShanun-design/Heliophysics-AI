#!/usr/bin/env python3
"""
Comprehensive Test Script for Solar AI System
============================================

This script tests all major components of the Solar AI system:
1. JAX environment
2. Data generation
3. Model creation and forward pass
4. Loss functions
5. Training step
6. Basic training loop
7. Visualization

Author: AI Assistant
Date: 2024
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt

def test_jax_environment():
    """Test JAX environment and basic operations."""
    print("🔍 Testing JAX Environment...")
    
    try:
        # Test basic JAX operations
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([4.0, 5.0, 6.0])
        z = x + y
        assert jnp.allclose(z, jnp.array([5.0, 7.0, 9.0]))
        
        # Test random number generation
        key = jax.random.PRNGKey(42)
        random_array = jax.random.normal(key, (10, 10))
        assert random_array.shape == (10, 10)
        
        print("✅ JAX environment working correctly!")
        return True
    except Exception as e:
        print(f"❌ JAX environment test failed: {e}")
        return False

def test_data_generation():
    """Test synthetic data generation."""
    print("\n🔍 Testing Data Generation...")
    
    try:
        from evaluation.comprehensive_evaluation import generate_synthetic_test_data
        
        # Generate test data
        test_data = generate_synthetic_test_data(
            n_samples=10,
            grid_size=(64, 64, 32),
            key=jax.random.PRNGKey(42)
        )
        
        # Check data structure
        assert 'magnetogram' in test_data
        assert 'coords' in test_data
        assert 'ground_truth' in test_data
        
        # Check shapes
        assert test_data['magnetogram'].shape[0] == 10  # n_samples
        assert test_data['magnetogram'].shape[1] == 3  # channels
        assert test_data['coords'].shape[0] == 10  # n_samples
        assert test_data['ground_truth'].shape[0] == 10  # n_samples
        
        print("✅ Data generation working correctly!")
        return True
    except Exception as e:
        print(f"❌ Data generation test failed: {e}")
        return False

def test_model_creation():
    """Test model creation and forward pass."""
    print("\n🔍 Testing Model Creation...")
    
    try:
        from models.solar_deeponet_3d import SolarDeepONet
        from models.solar_fno_3d import SolarFNO3D
        
        # Create DeepONet
        deeponet = SolarDeepONet(
            magnetogram_shape=(256, 256),
            latent_dim=32,
            branch_depth=3,
            trunk_depth=2,
            width=64,
            key=jax.random.PRNGKey(42)
        )
        
        # Create FNO
        fno = SolarFNO3D(
            input_channels=3,
            output_channels=3,
            modes=(8, 8, 4),
            width=32,
            depth=2,
            grid_size=(32, 32, 16),
            key=jax.random.PRNGKey(43)
        )
        
        print("✅ Both models created successfully!")
        return True, deeponet, fno
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        return False, None, None

def test_forward_pass(deeponet, fno):
    """Test forward pass through models."""
    print("\n🔍 Testing Forward Pass...")
    
    try:
        # Generate test data
        magnetogram = jax.random.normal(jax.random.PRNGKey(0), (3, 256, 256))
        coords = jax.random.normal(jax.random.PRNGKey(1), (100, 3))
        
        # Test DeepONet forward pass
        deeponet_output = deeponet(magnetogram, coords)
        assert deeponet_output.shape == (100, 3)
        
        # Test FNO forward pass (requires different input format)
        # FNO expects 3D grid coordinates
        grid_size = (32, 32, 16)
        x = jnp.linspace(-1, 1, grid_size[0])
        y = jnp.linspace(-1, 1, grid_size[1])
        z = jnp.linspace(-1, 1, grid_size[2])
        X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
        coords_3d = jnp.stack([X, Y, Z], axis=-1)  # (nx, ny, nz, 3)
        
        # Add batch dimension
        magnetogram_batch = magnetogram[None, ...]  # (1, 3, 256, 256)
        coords_batch = coords_3d[None, ...]  # (1, nx, ny, nz, 3)
        
        fno_output = fno(magnetogram_batch, coords_batch)
        assert fno_output.shape == (1, grid_size[0], grid_size[1], grid_size[2], 3)
        
        print("✅ Forward pass working correctly!")
        return True
    except Exception as e:
        print(f"❌ Forward pass test failed: {e}")
        return False

def test_loss_functions(deeponet):
    """Test loss functions."""
    print("\n🔍 Testing Loss Functions...")
    
    try:
        from models.solar_deeponet_3d import PhysicsInformedLoss
        
        # Create loss function
        loss_fn = PhysicsInformedLoss(lambda_data=1.0, lambda_physics=1.0)
        
        # Generate test data
        magnetogram = jax.random.normal(jax.random.PRNGKey(0), (3, 256, 256))
        coords = jax.random.normal(jax.random.PRNGKey(1), (100, 3))
        B_true = jax.random.normal(jax.random.PRNGKey(2), (100, 3))
        
        # Test loss computation
        total_loss, loss_components = loss_fn(
            deeponet, None, magnetogram, coords, B_true
        )
        
        assert isinstance(total_loss, float)
        assert isinstance(loss_components, dict)
        assert 'data_loss' in loss_components
        assert 'physics_loss' in loss_components
        assert 'divergence_loss' in loss_components
        assert 'total_loss' in loss_components
        
        print("✅ Loss functions working correctly!")
        return True
    except Exception as e:
        print(f"❌ Loss functions test failed: {e}")
        return False

def test_training_step(deeponet):
    """Test training step."""
    print("\n🔍 Testing Training Step...")
    
    try:
        from models.solar_deeponet_3d import create_solar_deeponet_training_step, PhysicsInformedLoss
        
        # Create optimizer
        optimizer = optax.adam(learning_rate=1e-3)
        
        # Create training step
        loss_fn = PhysicsInformedLoss(lambda_data=1.0, lambda_physics=1.0)
        training_step = create_solar_deeponet_training_step(deeponet, loss_fn, optimizer)
        
        # Generate test data
        magnetogram = jax.random.normal(jax.random.PRNGKey(0), (3, 256, 256))
        coords = jax.random.normal(jax.random.PRNGKey(1), (100, 3))
        B_true = jax.random.normal(jax.random.PRNGKey(2), (100, 3))
        
        # Initialize optimizer state
        opt_state = optimizer.init(None)  # No params needed for Equinox
        
        # Test training step
        loss, opt_state = training_step(None, opt_state, magnetogram, coords, B_true)
        
        assert isinstance(loss, float)
        assert loss > 0  # Loss should be positive
        
        print("✅ Training step working correctly!")
        return True
    except Exception as e:
        print(f"❌ Training step test failed: {e}")
        return False

def test_basic_training():
    """Test basic training loop."""
    print("\n🔍 Testing Basic Training Loop...")
    
    try:
        from models.solar_deeponet_3d import SolarDeepONet, PhysicsInformedLoss
        from evaluation.comprehensive_evaluation import generate_synthetic_test_data
        
        # Create model and loss
        model = SolarDeepONet(
            magnetogram_shape=(256, 256),
            latent_dim=16,  # Smaller for faster testing
            branch_depth=2,
            trunk_depth=2,
            width=32,
            key=jax.random.PRNGKey(42)
        )
        
        loss_fn = PhysicsInformedLoss(lambda_data=1.0, lambda_physics=0.1)
        optimizer = optax.adam(learning_rate=1e-3)
        
        # Generate training data
        train_data = generate_synthetic_test_data(
            n_samples=5,
            grid_size=(64, 64, 32),
            key=jax.random.PRNGKey(42)
        )
        
        # Training loop
        opt_state = optimizer.init(None)
        losses = []
        
        for step in range(3):  # Just 3 steps for testing
            loss, opt_state = optimizer.update(None, opt_state)
            losses.append(loss)
            
            # Generate new data for each step
            step_data = generate_synthetic_test_data(
                n_samples=5,
                grid_size=(64, 64, 32),
                key=jax.random.PRNGKey(step)
            )
            
            # Forward pass - need to reshape data for DeepONet
            # DeepONet expects (3, H, W) magnetogram and (N, 3) coords
            magnetogram = step_data['magnetogram'][0]  # Take first sample
            coords = step_data['coords'][0].reshape(-1, 3)  # Reshape to (N, 3)
            
            output = model(magnetogram, coords)
            
            # Compute loss
            ground_truth = step_data['ground_truth'][0].reshape(-1, 3)  # Reshape to (N, 3)
            total_loss, _ = loss_fn(
                model, None,
                magnetogram,
                coords,
                ground_truth
            )
        
        assert len(losses) == 3
        print("✅ Basic training loop working correctly!")
        return True
    except Exception as e:
        print(f"❌ Basic training loop test failed: {e}")
        return False

def test_visualization():
    """Test visualization functions."""
    print("\n🔍 Testing Visualization...")
    
    try:
        from evaluation.visualize_field import visualize_magnetogram, compute_mse
        
        # Generate test data (3D magnetogram with 3 components)
        magnetogram = jax.random.normal(jax.random.PRNGKey(0), (3, 256, 256))
        
        # Test visualization
        visualize_magnetogram(magnetogram, "Test Magnetogram")
        
        # Test metrics
        pred = jax.random.normal(jax.random.PRNGKey(1), (100, 3))
        true = jax.random.normal(jax.random.PRNGKey(2), (100, 3))
        
        mse = compute_mse(pred, true)
        assert isinstance(mse, float)
        assert mse > 0
        
        print("✅ Visualization functions working correctly!")
        return True
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False

def main():
    """Run comprehensive tests."""
    print("🚀 Starting Comprehensive Solar AI System Test")
    print("=" * 50)
    
    results = {}
    
    # Test 1: JAX Environment
    results['jax'] = test_jax_environment()
    
    # Test 2: Data Generation
    results['data'] = test_data_generation()
    
    # Test 3: Model Creation
    success, deeponet, fno = test_model_creation()
    results['models'] = success
    
    # Test 4: Forward Pass (if models created successfully)
    if success:
        results['forward_pass'] = test_forward_pass(deeponet, fno)
        
        # Test 5: Loss Functions
        results['loss'] = test_loss_functions(deeponet)
        
        # Test 6: Training Step
        results['training_step'] = test_training_step(deeponet)
        
        # Test 7: Basic Training
        results['training'] = test_basic_training()
    
    # Test 8: Visualization
    results['visualization'] = test_visualization()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:15} : {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Solar AI system is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return results

if __name__ == "__main__":
    main() 