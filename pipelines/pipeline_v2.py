from kfp import dsl, compiler
import sys
import os

sys.path.append(os.path.abspath('../components'))
from data_ingestion import data_ingestion_component
from preprocessing import preprocessing_component
from train_svm import svm_ga_train_component
from train_rf import rf_feature_selection_component
from evaluate import evaluation_component

@dsl.pipeline(name='parameterized-ml-workflow', description='Task 4: Change settings via UI')
def parameterized_pipeline(
    model_type: str = 'svm',  # You can choose 'svm' or 'rf'
    rf_trees: int = 100,
    feature_count: int = 10
):
    ingest = data_ingestion_component()
    preprocess = preprocessing_component(input_dataset=ingest.outputs['output_dataset'])
    
    # Conditional logic based on parameters
    with dsl.If(model_type == 'svm'):
        train_svm = svm_ga_train_component(train_data=preprocess.outputs['output_train'])
        evaluate_svm = evaluation_component(
            test_data=preprocess.outputs['output_test'],
            model_in=train_svm.outputs['model_artifact']
        )
        
    with dsl.If(model_type == 'rf'):
        train_rf = rf_feature_selection_component(
            train_data=preprocess.outputs['output_train'],
            n_estimators=rf_trees,
            max_features=feature_count
        )
        # Note: You'll need a slightly different eval for RF if it includes the selector
        # For simplicity, we can just run the evaluations separately.

if __name__ == '__main__':
    compiler.Compiler().compile(parameterized_pipeline, 'parameterized_workflow.yaml')
