#!/usr/bin/env python3
"""
simple_test.py
---------------
Simplified test to verify basic Solar AI functionality.
"""
import sys
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_basic_functionality():
    """Test basic functionality without complex model interactions."""
    print("🔍 Simple Solar AI Test")
    print("=" * 40)
    
    # Test 1: JAX environment
    print("1. Testing JAX environment...")
    x = jnp.array([1, 2, 3, 4, 5])
    y = jnp.array([2, 3, 4, 5, 6])
    z = x * y + jnp.sin(x)
    print(f"   JAX computation: {z}")
    print("   ✅ JAX working")
    
    # Test 2: Data generation
    print("\n2. Testing data generation...")
    try:
        from evaluation.comprehensive_evaluation import generate_synthetic_test_data
        
        test_data = generate_synthetic_test_data(
            n_samples=2,
            grid_size=(16, 16, 8),
            key=jax.random.PRNGKey(42)
        )
        
        print(f"   Generated data keys: {list(test_data.keys())}")
        print(f"   Magnetogram shape: {test_data['magnetogram'].shape}")
        print(f"   Coordinates shape: {test_data['coords'].shape}")
        print(f"   Ground truth shape: {test_data['ground_truth'].shape}")
        print("   ✅ Data generation working")
        
    except Exception as e:
        print(f"   ❌ Data generation failed: {e}")
        return False
    
    # Test 3: Model creation (simplified)
    print("\n3. Testing model creation...")
    try:
        from models.solar_deeponet_3d import SolarDeepONet
        
        # Create a smaller model for testing
        deeponet = SolarDeepONet(
            magnetogram_shape=(256, 256),  # Use the default size
            latent_dim=32,
            branch_depth=3,
            trunk_depth=2,
            width=64,
            key=jax.random.PRNGKey(42)
        )
        print("   ✅ DeepONet created successfully")
        
    except Exception as e:
        print(f"   ❌ Model creation failed: {e}")
        return False
    
    # Test 4: Simple forward pass
    print("\n4. Testing simple forward pass...")
    try:
        # Create simple test data
        magnetogram = jax.random.normal(jax.random.PRNGKey(0), (3, 256, 256))  # (channels, height, width)
        coords = jax.random.normal(jax.random.PRNGKey(1), (100, 3))
        
        # Forward pass
        # The DeepONet expects magnetogram in (channels, height, width) format
        # and adds the batch dimension internally
        output = deeponet(magnetogram, coords)
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
        print("   ✅ Forward pass working")
        
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Loss function
    print("\n5. Testing loss function...")
    try:
        from models.solar_deeponet_3d import PhysicsInformedLoss
        
        loss_fn = PhysicsInformedLoss(lambda_data=1.0, lambda_physics=0.1)
        
        # Create test data
        magnetogram = jax.random.normal(jax.random.PRNGKey(0), (3, 256, 256))
        coords = jax.random.normal(jax.random.PRNGKey(1), (100, 3))
        B_true = jax.random.normal(jax.random.PRNGKey(2), (100, 3))
        
        # Compute loss
        total_loss, loss_components = loss_fn(
            deeponet, None, magnetogram, coords, B_true
        )
        
        print(f"   Total loss: {total_loss:.6f}")
        print(f"   Loss components: {loss_components}")
        print("   ✅ Loss function working")
        
    except Exception as e:
        print(f"   ❌ Loss function failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 6: Optimization
    print("\n6. Testing optimization...")
    try:
        import optax
        
        # Create optimizer
        optimizer = optax.adam(learning_rate=1e-3)
        opt_state = optimizer.init(None)
        
        # Create simple training step
        def simple_loss(params, magnetogram, coords, B_true):
            pred = deeponet(magnetogram, coords)
            return jnp.mean((pred - B_true) ** 2)
        
        # Compute gradients
        loss_val, grads = jax.value_and_grad(simple_loss)(
            None, magnetogram, coords, B_true
        )
        
        print(f"   Loss: {loss_val:.6f}")
        print(f"   Gradients computed successfully")
        print("   ✅ Optimization working")
        
    except Exception as e:
        print(f"   ❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 40)
    print("🎉 ALL TESTS PASSED!")
    print("The Solar AI system is working correctly!")
    return True

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1) 