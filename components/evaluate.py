from kfp import dsl
from kfp.dsl import Dataset, Input, Model, Metrics, ClassificationMetrics, Output

@dsl.component(base_image='python:3.9', packages_to_install=['pandas==2.0.3', 'numpy==1.26.4', 'scikit-learn==1.3.0', 'matplotlib==3.7.2', 'seaborn==0.12.2'])
def evaluation_component(
    test_data: Input[Dataset],
    model_in: Input[Model],
    metrics: Output[Metrics],
    plots: Output[ClassificationMetrics]
):
    import pandas as pd
    import joblib
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    df = pd.read_csv(test_data.path)
    X_test, y_test = df.drop('target', axis=1), df['target']
    
    model = joblib.load(model_in.path)
    y_pred = model.predict(X_test)

    # Log all required metrics
    metrics.log_metric("accuracy", float(accuracy_score(y_test, y_pred)))
    metrics.log_metric("precision", float(precision_score(y_test, y_pred, average='weighted')))
    metrics.log_metric("recall", float(recall_score(y_test, y_pred, average='weighted')))
    metrics.log_metric("f1_score", float(f1_score(y_test, y_pred, average='weighted')))

    cm = confusion_matrix(y_test, y_pred)
    plots.log_confusion_matrix(['Benign', 'Malignant'], cm.tolist())
