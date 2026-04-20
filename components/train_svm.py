from kfp import dsl
from kfp.dsl import Dataset, Model, Input, Output

@dsl.component(base_image='python:3.9', packages_to_install=['pandas==2.0.3', 'numpy==1.26.4', 'scikit-learn==1.3.0', 'deap==1.4.1', 'joblib==1.3.1'])
def svm_ga_train_component(
# ... rest of your code
train_data: Input[Dataset], model_artifact: Output[Model]):
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.svm import SVC
    from deap import base, creator, tools, algorithms

    df = pd.read_csv(train_data.path)
    X, y = df.drop('target', axis=1), df['target']

    # GA to optimize C and Gamma
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, 0.1, 10)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=2)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(ind):
        model = SVC(C=max(0.1, ind[0]), gamma=max(0.001, ind[1]))
        model.fit(X, y)
        return (model.score(X, y),)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=10)
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=5, verbose=False)
    
    best = tools.selBest(pop, 1)[0]
    final_model = SVC(C=max(0.1, best[0]), gamma=max(0.001, best[1]))
    final_model.fit(X, y)
    joblib.dump(final_model, model_artifact.path)
