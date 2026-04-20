from kfp import dsl
from kfp.dsl import Dataset, Input, Output

@dsl.component(base_image='python:3.9', packages_to_install=['pandas==2.0.3', 'numpy==1.26.4', 'scikit-learn==1.3.0'])
def preprocessing_component(
    input_dataset: Input[Dataset],
    output_train: Output[Dataset],
    output_test: Output[Dataset],
    scaler_type: str = 'standard',  # NEW PARAMETER
    test_size: float = 0.2,         # NEW PARAMETER
    random_seed: int = 42           # NEW PARAMETER
):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    df = pd.read_csv(input_dataset.path)
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Task 6: Experiment with different preprocessing techniques
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
        
    X_scaled = scaler.fit_transform(X)
    processed_df = pd.DataFrame(X_scaled, columns=X.columns)
    processed_df['target'] = y.values

    # Task 7: Slight data splits
    train_df, test_df = train_test_split(processed_df, test_size=test_size, random_state=random_seed)
    
    train_df.to_csv(output_train.path, index=False)
    test_df.to_csv(output_test.path, index=False)
