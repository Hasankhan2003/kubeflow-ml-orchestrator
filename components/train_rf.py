from kfp import dsl
from kfp.dsl import Dataset, Model, Input, Output

@dsl.component(base_image='python:3.9', packages_to_install=['pandas==2.0.3', 'numpy==1.26.4', 'scikit-learn==1.3.0', 'joblib==1.3.1'])
def rf_train_component(
    train_data: Input[Dataset], 
    model_artifact: Output[Model],
    n_estimators: int = 100,  # NEW HYPERPARAMETER
    random_seed: int = 42     # NEW HYPERPARAMETER
):
    import pandas as pd
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectFromModel
    from sklearn.pipeline import Pipeline

    df = pd.read_csv(train_data.path)
    X, y = df.drop('target', axis=1), df['target']

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_seed)
    pipeline = Pipeline([
        ('feature_selection', SelectFromModel(clf)),
        ('classification', RandomForestClassifier(n_estimators=n_estimators, random_state=random_seed))
    ])
    
    pipeline.fit(X, y)
    joblib.dump(pipeline, model_artifact.path)
