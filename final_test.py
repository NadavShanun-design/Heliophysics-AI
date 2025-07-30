#!/usr/bin/env python3
"""
Final Test Script for Solar AI System
=====================================

This script performs a final verification that the core components are working:
1. JAX environment
2. Model creation
3. Forward pass
4. Data generation
5. Basic training

Author: AI Assistant
Date: 2024
"""

import jax
import jax.numpy as jnp
import numpy as np

def test_core_functionality():
    """Test core functionality of the Solar AI system."""
    print("🚀 Final Core Functionality Test")
    print("=" * 40)
    
    # Test 1: JAX Environment
    print("🔍 Testing JAX Environment...")
    try:
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([4.0, 5.0, 6.0])
        z = x + y
        assert jnp.allclose(z, jnp.array([5.0, 7.0, 9.0]))
        print("✅ JAX environment working!")
    except Exception as e:
        print(f"❌ JAX environment failed: {e}")
        return False
    
    # Test 2: Model Creation
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
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False
    
    # Test 3: Forward Pass
    print("\n🔍 Testing Forward Pass...")
    try:
        # Generate test data
        magnetogram = jax.random.normal(jax.random.PRNGKey(0), (3, 256, 256))
        coords = jax.random.normal(jax.random.PRNGKey(1), (100, 3))
        
        # Test DeepONet forward pass
        deeponet_output = deeponet(magnetogram, coords)
        assert deeponet_output.shape == (100, 3)
        print("✅ DeepONet forward pass working!")
        
        # Test FNO forward pass
        grid_size = (32, 32, 16)
        x = jnp.linspace(-1, 1, grid_size[0])
        y = jnp.linspace(-1, 1, grid_size[1])
        z = jnp.linspace(-1, 1, grid_size[2])
        X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
        coords_3d = jnp.stack([X, Y, Z], axis=-1)
        
        magnetogram_batch = magnetogram[None, ...]
        coords_batch = coords_3d[None, ...]
        
        fno_output = fno(magnetogram_batch, coords_batch)
        assert fno_output.shape == (1, grid_size[0], grid_size[1], grid_size[2], 3)
        print("✅ FNO forward pass working!")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    # Test 4: Data Generation
    print("\n🔍 Testing Data Generation...")
    try:
        from evaluation.comprehensive_evaluation import generate_synthetic_test_data
        
        test_data = generate_synthetic_test_data(
            n_samples=5,
            grid_size=(64, 64, 32),
            key=jax.random.PRNGKey(42)
        )
        
        assert 'magnetogram' in test_data
        assert 'coords' in test_data
        assert 'ground_truth' in test_data
        assert test_data['magnetogram'].shape[0] == 5
        print("✅ Data generation working!")
        
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return False
    
    # Test 5: Loss Function
    print("\n🔍 Testing Loss Function...")
    try:
        from models.solar_deeponet_3d import PhysicsInformedLoss
        
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
        print("✅ Loss function working!")
        
    except Exception as e:
        print(f"❌ Loss function failed: {e}")
        return False
    
    # Test 6: Basic Training Step
    print("\n🔍 Testing Basic Training Step...")
    try:
        import optax
        from models.solar_deeponet_3d import create_solar_deeponet_training_step
        
        optimizer = optax.adam(learning_rate=1e-3)
        training_step = create_solar_deeponet_training_step(deeponet, loss_fn, optimizer)
        
        # Generate test data
        magnetogram = jax.random.normal(jax.random.PRNGKey(0), (3, 256, 256))
        coords = jax.random.normal(jax.random.PRNGKey(1), (100, 3))
        B_true = jax.random.normal(jax.random.PRNGKey(2), (100, 3))
        
        # Initialize optimizer state
        opt_state = optimizer.init(None)
        
        # Test training step
        loss, opt_state = training_step(None, opt_state, magnetogram, coords, B_true)
        
        assert isinstance(loss, float)
        assert loss > 0
        print("✅ Training step working!")
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        return False
    
    print("\n" + "=" * 40)
    print("🎉 ALL TESTS PASSED!")
    print("✅ Solar AI system is working correctly!")
    print("=" * 40)
    
    return True

if __name__ == "__main__":
    success = test_core_functionality()
    if not success:
        print("\n❌ Some tests failed. Please check the errors above.")
        exit(1) 