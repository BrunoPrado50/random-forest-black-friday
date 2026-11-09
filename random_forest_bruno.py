from sklearn.ensemble import RandomForestRegressor
from services.metricas import metricas


def random_forest_treino(x_train, y_train, parametros=None):
    # Ordem no TXT: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features.
    if parametros is None:
        parametros = [80, 0, 2, 1, 1.0]
    if len(parametros) != 5:
        raise ValueError("Random Forest: informe os 5 valores na ordem indicada.")

    arvores, profundidade, min_divisao, min_folha, fracao = parametros
    modelo = RandomForestRegressor(
        n_estimators=arvores,
        max_depth=None if profundidade == 0 else profundidade,  # 0 = sem limite
        min_samples_split=min_divisao,
        min_samples_leaf=min_folha,
        max_features=float(fracao),  # 1 significa usar 100% dos atributos
        random_state=42,
        n_jobs=2,
    )
    return modelo.fit(x_train, y_train)


def random_forest_teste(x_teste, y_teste, modelo):
    previsoes = modelo.predict(x_teste)
    return metricas("RANDOM FOREST", y_teste, previsoes)
