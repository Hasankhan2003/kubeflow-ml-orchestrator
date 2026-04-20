from kfp import dsl
from kfp.dsl import Dataset, Model, Input, Output

@dsl.component(base_image='python:3.9', packages_to_install=['pandas==2.0.3', 'numpy==1.26.4', 'scikit-learn==1.3.0', 'joblib==1.3.1'])
def nn_train_component(
    train_data: Input[Dataset], 
    model_artifact: Output[Model],
    max_iter: int = 500,     # NEW HYPERPARAMETER
    random_seed: int = 42    # NEW HYPERPARAMETER
):
    import pandas as pd
    import joblib
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import RandomizedSearchCV

    df = pd.read_csv(train_data.path)
    X, y = df.drop('target', axis=1), df['target']

    mlp = MLPClassifier(max_iter=max_iter, random_state=random_seed)
    param_dist = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.05]
    }
    
    search = RandomizedSearchCV(mlp, param_distributions=param_dist, n_iter=5, cv=3, random_state=random_seed)
    search.fit(X, y)
    
    joblib.dump(search.best_estimator_, model_artifact.path)
