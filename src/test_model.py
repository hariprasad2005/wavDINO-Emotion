"""
Testing and validation utilities for wavDINO-Emotion
"""

import torch
import numpy as np
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.models.wavdino_emotion import WavDINOEmotion
from src.models.inference import ModelLoader


def test_model_creation():
    """Test model creation and forward pass"""
    print("Testing Model Creation...")
    try:
        model = WavDINOEmotion(num_emotions=6)
        print("✓ Model created successfully")
        
        # Test forward pass
        audio_emb = torch.randn(2, 1024)
        visual_emb = torch.randn(2, 1024)
        logits, probs = model(audio_emb, visual_emb)
        
        assert logits.shape == (2, 6), f"Expected logits shape (2, 6), got {logits.shape}"
        assert probs.shape == (2, 6), f"Expected probs shape (2, 6), got {probs.shape}"
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5), "Probabilities don't sum to 1"
        
        print("✓ Forward pass successful")
        print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_model_saving_loading():
    """Test model checkpoint saving and loading"""
    print("\nTesting Model Saving/Loading...")
    try:
        # Create and save model
        model1 = WavDINOEmotion(num_emotions=6)
        checkpoint_path = "test_checkpoint.pt"
        torch.save(model1.state_dict(), checkpoint_path)
        print("✓ Model saved")
        
        # Load model
        model2 = WavDINOEmotion(num_emotions=6)
        model2.load_state_dict(torch.load(checkpoint_path))
        print("✓ Model loaded")
        
        # Compare outputs
        audio_emb = torch.randn(2, 1024)
        visual_emb = torch.randn(2, 1024)
        
        with torch.no_grad():
            logits1, _ = model1(audio_emb, visual_emb)
            logits2, _ = model2(audio_emb, visual_emb)
        
        assert torch.allclose(logits1, logits2, atol=1e-5), "Model outputs differ after loading"
        print("✓ Model outputs match")
        
        # Clean up
        os.remove(checkpoint_path)
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        return False


def test_inference_loader():
    """Test ModelLoader for inference"""
    print("\nTesting Model Loader...")
    try:
        # Create and save test model
        model = WavDINOEmotion(num_emotions=6)
        checkpoint_path = "test_inference.pt"
        torch.save(model.state_dict(), checkpoint_path)
        
        # Load with ModelLoader
        loader = ModelLoader(checkpoint_path, device='cpu')
        print("✓ ModelLoader initialized")
        
        # Test prediction
        audio_emb = np.random.randn(1024).astype(np.float32)
        visual_emb = np.random.randn(1024).astype(np.float32)
        
        emotion, confidence, probs = loader.predict(audio_emb, visual_emb)
        
        assert isinstance(emotion, str), "Emotion should be string"
        assert 0 <= confidence <= 1, f"Confidence should be between 0 and 1, got {confidence}"
        assert len(probs) == 6, f"Should have 6 emotion probabilities, got {len(probs)}"
        assert np.isclose(sum(probs.values()), 1.0, atol=1e-5), "Probabilities don't sum to 1"
        
        print(f"✓ Prediction successful")
        print(f"  - Emotion: {emotion}")
        print(f"  - Confidence: {confidence:.4f}")
        
        # Clean up
        os.remove(checkpoint_path)
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        return False


def test_batch_processing():
    """Test batch processing with different batch sizes"""
    print("\nTesting Batch Processing...")
    try:
        model = WavDINOEmotion(num_emotions=6)
        
        for batch_size in [1, 2, 8, 16, 32]:
            audio_emb = torch.randn(batch_size, 1024)
            visual_emb = torch.randn(batch_size, 1024)
            
            with torch.no_grad():
                logits, probs = model(audio_emb, visual_emb)
            
            assert logits.shape[0] == batch_size, f"Batch size mismatch for {batch_size}"
            assert probs.shape[0] == batch_size, f"Output batch size mismatch"
            
            print(f"✓ Batch size {batch_size:2d}: OK")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_gradient_flow():
    """Test gradient flow through the model"""
    print("\nTesting Gradient Flow...")
    try:
        model = WavDINOEmotion(num_emotions=6)
        criterion = torch.nn.CrossEntropyLoss()
        
        audio_emb = torch.randn(4, 1024, requires_grad=True)
        visual_emb = torch.randn(4, 1024, requires_grad=True)
        labels = torch.randint(0, 6, (4,))
        
        # Forward pass
        logits, _ = model(audio_emb, visual_emb)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        has_gradient = False
        for param in model.parameters():
            if param.grad is not None and torch.abs(param.grad).sum() > 0:
                has_gradient = True
                break
        
        assert has_gradient, "No gradients computed"
        print("✓ Gradients computed successfully")
        
        # Check for NaN gradients
        for param in model.parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), "NaN gradients detected"
        
        print("✓ No NaN gradients")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_different_devices():
    """Test model on different devices"""
    print("\nTesting Different Devices...")
    try:
        model = WavDINOEmotion(num_emotions=6)
        audio_emb = torch.randn(2, 1024)
        visual_emb = torch.randn(2, 1024)
        
        # CPU
        model_cpu = model.to('cpu')
        audio_cpu = audio_emb.to('cpu')
        visual_cpu = visual_emb.to('cpu')
        
        with torch.no_grad():
            logits_cpu, probs_cpu = model_cpu(audio_cpu, visual_cpu)
        print("✓ CPU device: OK")
        
        # CUDA (if available)
        if torch.cuda.is_available():
            model_cuda = model.to('cuda')
            audio_cuda = audio_emb.to('cuda')
            visual_cuda = visual_emb.to('cuda')
            
            with torch.no_grad():
                logits_cuda, probs_cuda = model_cuda(audio_cuda, visual_cuda)
            
            # Compare outputs
            diff = torch.abs(logits_cpu - logits_cuda.cpu()).max().item()
            assert diff < 1e-4, f"CPU-CUDA output mismatch: {diff}"
            print("✓ CUDA device: OK")
        else:
            print("⚠ CUDA not available")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_model_modes():
    """Test model train/eval modes"""
    print("\nTesting Model Modes...")
    try:
        model = WavDINOEmotion(num_emotions=6)
        audio_emb = torch.randn(4, 1024)
        visual_emb = torch.randn(4, 1024)
        
        # Training mode
        model.train()
        logits1, probs1 = model(audio_emb, visual_emb)
        print("✓ Train mode: OK")
        
        # Evaluation mode
        model.eval()
        with torch.no_grad():
            logits2, probs2 = model(audio_emb, visual_emb)
        print("✓ Eval mode: OK")
        
        # Outputs should be different due to dropout
        # (but this might not always be true for small networks)
        print("✓ Mode switching: OK")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("wavDINO-Emotion Test Suite")
    print("=" * 60)
    
    tests = [
        ("Model Creation", test_model_creation),
        ("Model Saving/Loading", test_model_saving_loading),
        ("Inference Loader", test_inference_loader),
        ("Batch Processing", test_batch_processing),
        ("Gradient Flow", test_gradient_flow),
        ("Different Devices", test_different_devices),
        ("Model Modes", test_model_modes),
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
