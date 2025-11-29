"""
Test script to verify all pipeline components work correctly
Run this before compiling to YAML to catch any errors
"""
import sys
import os

def test_import_components():
    """Test that all components can be imported"""
    print("="*60)
    print("Testing Component Imports")
    print("="*60)
    
    try:
        from src.pipeline_components import (
            data_extraction,
            data_preprocessing,
            model_training,
            model_evaluation,
            COMPONENT_METADATA
        )
        print("✓ All components imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_component_metadata():
    """Test that component metadata is properly defined"""
    print("\n" + "="*60)
    print("Testing Component Metadata")
    print("="*60)
    
    try:
        from src.pipeline_components import COMPONENT_METADATA
        
        expected_components = [
            'data_extraction',
            'data_preprocessing', 
            'model_training',
            'model_evaluation'
        ]
        
        for component_name in expected_components:
            if component_name in COMPONENT_METADATA:
                metadata = COMPONENT_METADATA[component_name]
                print(f"\n{component_name}:")
                print(f"  Name: {metadata['name']}")
                print(f"  Inputs: {metadata['inputs']}")
                print(f"  Outputs: {metadata['outputs']}")
                print(f"  ✓ Metadata complete")
            else:
                print(f"✗ Missing metadata for {component_name}")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Metadata error: {e}")
        return False

def test_component_signatures():
    """Test that components have correct function signatures"""
    print("\n" + "="*60)
    print("Testing Component Signatures")
    print("="*60)
    
    try:
        from src.pipeline_components import (
            data_extraction,
            data_preprocessing,
            model_training,
            model_evaluation
        )
        import inspect
        
        # Test data_extraction
        sig = inspect.signature(data_extraction.python_func)
        params = list(sig.parameters.keys())
        print(f"\ndata_extraction parameters: {params}")
        assert 'data_path' in params, "Missing data_path parameter"
        assert 'dvc_remote' in params, "Missing dvc_remote parameter"
        assert 'output_data' in params, "Missing output_data parameter"
        print("✓ data_extraction signature correct")
        
        # Test data_preprocessing
        sig = inspect.signature(data_preprocessing.python_func)
        params = list(sig.parameters.keys())
        print(f"\ndata_preprocessing parameters: {params}")
        assert 'input_data' in params, "Missing input_data parameter"
        assert 'train_data' in params, "Missing train_data parameter"
        assert 'test_data' in params, "Missing test_data parameter"
        print("✓ data_preprocessing signature correct")
        
        # Test model_training
        sig = inspect.signature(model_training.python_func)
        params = list(sig.parameters.keys())
        print(f"\nmodel_training parameters: {params}")
        assert 'train_data' in params, "Missing train_data parameter"
        assert 'model_output' in params, "Missing model_output parameter"
        assert 'n_estimators' in params, "Missing n_estimators parameter"
        print("✓ model_training signature correct")
        
        # Test model_evaluation
        sig = inspect.signature(model_evaluation.python_func)
        params = list(sig.parameters.keys())
        print(f"\nmodel_evaluation parameters: {params}")
        assert 'test_data' in params, "Missing test_data parameter"
        assert 'model' in params, "Missing model parameter"
        assert 'metrics_output' in params, "Missing metrics_output parameter"
        print("✓ model_evaluation signature correct")
        
        return True
    except Exception as e:
        print(f"✗ Signature error: {e}")
        return False

def test_dependencies():
    """Test that all required dependencies are installed"""
    print("\n" + "="*60)
    print("Testing Dependencies")
    print("="*60)
    
    required_packages = [
        'kfp',
        'pandas',
        'sklearn',
        'numpy',
        'joblib'
    ]
    
    all_installed = True
    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            print(f"✗ {package} NOT installed")
            all_installed = False
    
    return all_installed

def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "="*60)
    print("RUNNING COMPONENT TESTS")
    print("="*60 + "\n")
    
    tests = [
        ("Import Test", test_import_components),
        ("Metadata Test", test_component_metadata),
        ("Signature Test", test_component_signatures),
        ("Dependencies Test", test_dependencies)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED! ✓")
        print("You can now compile components to YAML")
    else:
        print("SOME TESTS FAILED! ✗")
        print("Please fix errors before compiling")
    print("="*60 + "\n")
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)