from kfp import dsl
from kfp.dsl import Dataset, Output

@dsl.component(base_image='python:3.9', packages_to_install=['pandas==2.0.3', 'numpy==1.26.4', 'scikit-learn==1.3.0'])
def data_ingestion_component(output_dataset: Output[Dataset]):
    # ... rest of your code stays exactly the same    
    from sklearn.datasets import load_breast_cancer
    import pandas as pd
    
    # This loads the data from inside the container (No local file needed!)
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    
    # Save for the next step
    df.to_csv(output_dataset.path, index=False)
