import sys
import os
from kfp import dsl, compiler

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', 'components'))

from data_ingestion import data_ingestion_component
from preprocessing import preprocessing_component
from train_svm import svm_ga_train_component
from train_rf import rf_train_component
from train_nn import nn_train_component
from evaluate import evaluation_component

@dsl.pipeline(name='Ultimate-Master-Pipeline', description='Fully parameterized pipeline meeting all Tasks')
def master_pipeline(
    model_type: str = 'RF',         # Model selection
    random_seed: int = 42,          # Task 7: Reproducibility
    test_size: float = 0.2,         # Task 7: Data splits (e.g., 0.2 or 0.3)
    scaler_type: str = 'standard',  # Task 6: Preprocessing ('standard' or 'minmax')
    rf_n_estimators: int = 100,     # Task 4: RF Hyperparameter
    nn_max_iter: int = 500          # Task 4: NN Hyperparameter
):
    ingest = data_ingestion_component()
    
    preprocess = preprocessing_component(
        input_dataset=ingest.outputs['output_dataset'],
        scaler_type=scaler_type,
        test_size=test_size,
        random_seed=random_seed
    )
    
    with dsl.If(model_type == 'SVM'):
        train_svm = svm_ga_train_component(train_data=preprocess.outputs['output_train'])
        evaluate_svm = evaluation_component(
            test_data=preprocess.outputs['output_test'], 
            model_in=train_svm.outputs['model_artifact']
        )
        
    with dsl.If(model_type == 'RF'):
        train_rf = rf_train_component(
            train_data=preprocess.outputs['output_train'],
            n_estimators=rf_n_estimators,
            random_seed=random_seed
        )
        evaluate_rf = evaluation_component(
            test_data=preprocess.outputs['output_test'], 
            model_in=train_rf.outputs['model_artifact']
        )
        
    with dsl.If(model_type == 'NN'):
        train_nn = nn_train_component(
            train_data=preprocess.outputs['output_train'],
            max_iter=nn_max_iter,
            random_seed=random_seed
        )
        evaluate_nn = evaluation_component(
            test_data=preprocess.outputs['output_test'], 
            model_in=train_nn.outputs['model_artifact']
        )

if __name__ == '__main__':
    compiler.Compiler().compile(master_pipeline, 'ultimate_master_pipeline.yaml')
    print("✅ Ultimate Master Pipeline compiled!")
