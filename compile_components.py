# """
# Script to compile MLflow/KFP components to YAML files
# This creates reusable component definitions that can be shared and versioned
# """
# import os
# from kfp import compiler
# from src.pipeline_components import (
#     data_extraction,
#     data_preprocessing,
#     model_training,
#     model_evaluation
# )

# def compile_components():
#     """
#     Compile all pipeline components to YAML files
#     """
#     # Create components directory if it doesn't exist
#     os.makedirs('components', exist_ok=True)
    
#     print("="*60)
#     print("COMPILING PIPELINE COMPONENTS TO YAML")
#     print("="*60)
    
#     # List of components to compile
#     components = [
#         (data_extraction, 'data_extraction.yaml'),
#         (data_preprocessing, 'data_preprocessing.yaml'),
#         (model_training, 'model_training.yaml'),
#         (model_evaluation, 'model_evaluation.yaml')
#     ]
    
#     # Compile each component
#     for component_func, output_file in components:
#         output_path = os.path.join('components', output_file)
        
#         print(f"\nCompiling: {component_func.__name__}")
#         print(f"Output: {output_path}")
        
#         try:
#             # Use the compiler to create YAML
#             compiler.Compiler().compile(
#                 pipeline_func=component_func,
#                 package_path=output_path
#             )
#             print(f"✓ Successfully compiled {component_func.__name__}")
            
#         except Exception as e:
#             print(f"✗ Error compiling {component_func.__name__}: {str(e)}")
    
#     print("\n" + "="*60)
#     print("COMPILATION COMPLETE")
#     print("="*60)
    
#     # List all compiled files
#     print("\nCompiled component files:")
#     for file in os.listdir('components'):
#         if file.endswith('.yaml'):
#             file_path = os.path.join('components', file)
#             file_size = os.path.getsize(file_path)
#             print(f"  • {file} ({file_size} bytes)")
    
#     print("\n" + "="*60)

# if __name__ == "__main__":
#     compile_components()

"""
Script to compile MLflow/KFP components to YAML files
This creates reusable component definitions that can be shared and versioned
"""
import os
from kfp import compiler
from src.pipeline_components import (
    data_extraction,
    data_preprocessing,
    model_training,
    model_evaluation
)

def compile_components():
    """
    Compile all pipeline components to YAML files
    """
    # Create components directory if it doesn't exist
    os.makedirs('components', exist_ok=True)
    
    print("="*60)
    print("COMPILING PIPELINE COMPONENTS TO YAML")
    print("="*60)
    
    # List of components to compile with their names
    components = [
        (data_extraction, 'data_extraction', 'data_extraction.yaml'),
        (data_preprocessing, 'data_preprocessing', 'data_preprocessing.yaml'),
        (model_training, 'model_training', 'model_training.yaml'),
        (model_evaluation, 'model_evaluation', 'model_evaluation.yaml')
    ]
    
    # Compile each component
    compiled_count = 0
    for component_func, component_name, output_file in components:
        output_path = os.path.join('components', output_file)
        
        print(f"\nCompiling: {component_name}")
        print(f"Output: {output_path}")
        
        try:
            # Use the compiler to create YAML
            compiler.Compiler().compile(
                pipeline_func=component_func,
                package_path=output_path
            )
            print(f"✓ Successfully compiled {component_name}")
            compiled_count += 1
            
        except Exception as e:
            print(f"✗ Error compiling {component_name}: {str(e)}")
            print(f"   Error type: {type(e).__name__}")
    
    print("\n" + "="*60)
    print(f"COMPILATION COMPLETE - {compiled_count}/4 components compiled")
    print("="*60)
    
    # List all compiled files
    if os.path.exists('components'):
        yaml_files = [f for f in os.listdir('components') if f.endswith('.yaml')]
        
        if yaml_files:
            print("\nCompiled component files:")
            for file in yaml_files:
                file_path = os.path.join('components', file)
                file_size = os.path.getsize(file_path)
                print(f"  • {file} ({file_size:,} bytes)")
        else:
            print("\n⚠ Warning: No YAML files found in components/ directory")
    
    print("\n" + "="*60)
    
    return compiled_count == 4

if __name__ == "__main__":
    import sys
    success = compile_components()
    
    if success:
        print("\n✓ All components compiled successfully!")
        sys.exit(0)
    else:
        print("\n✗ Some components failed to compile. Check errors above.")
        sys.exit(1)