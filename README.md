
# Getting Started
Firecannon é uma ferramenta para treinamento automatizado de modelos de machine learning. Com ela, é possível:
- Encontrar automaticamente o melhor modelo para determinado conjunto de dados;
- Gerar relatórios de métricas para cada algoritmo treinado;

## Como instalar
* Clonar o repositório
```bash
git clone https://github.com/Malware-Hunter/firecannon
```
* Instalar a biblioteca
```bash
cd firecannon
sh distribute.sh
```
* Também é possível utilizar através do Docker (certifique-se que está instalado na máquina)
```bash
cd firecannon
docker image build -t firecannon .
docker container run firecannon
```

### Como utilizar
Firecannon implementa duas abstrações de estimadores, denominados `Classifier` e `Regressor`, para problemas de classificação e regressão, respectivamente. Ambas as classes podem ser acessadas de maneira demonstrada pelo exemplo abaixo:

```python
from sklearn.datasets import make_classification # dataset de exemplo
from firecannon.estimators import Classifier, Regressor

X_c, y_c = make_classification(n_classes=2, n_features=5, n_samples=100)

my_classifier = Classifier()
my_classifier.fit(X_c, y_c)
print(my_classifier.best_model) # por exemplo, KNeighborsClassifier()
```

### Personalizando a execução
Firecannon foi construído de forma a fornecer uma abstração de alto nível de treinamento de modelos, portanto, não requer que o usuário interaja com detalhes de baixo nível, como definição de algoritmos ou ajuste de hiper-parâmetros. Entretanto, é possível modificar esses detalhes, através de argumentos passados às classes `Classifier` e `Regressor`. Abaixo estão elencados os possíveis atributos a serem modificados:

* **model_settings: dict, default=None**: Especifica um dicionário de modelos personalizados, a qual deve conter instâncias de modelos como chaves e um dicionário de hiper-parâmetros **válidos** como valores. A implementação de modelos personalizados deve seguir o contrato definido na [documentação](https://scikit-learn.org/stable/developers/develop.html) do scikit-learn. A estrutura de dicionário esperada por `model_settings` deve ser parecida com o exemplo abaixo:
	```python
	{
	    MyKNeighborsInstance(): {
			'n_neighbors': [3, 5, 7],
			'leaf_size': [15, 30, 45, 60],
			'weights': ['uniform', 'distance']
	    },
	    MyRandomForestClassifierInstance(): {
			'n_estimators': [50, 100, 150],
			'criterion': ['gini', 'entropy'],
			'max_features': ['auto', 'log2', 'sqrt'],
	    },
	    MyAdaBoostClassifierInstance(): {
			'n_estimators': [50, 100, 150],
			'learning_rate': [1, 0.1, 0.5]
	    }
	}
	```

### Encontrando os melhores hiperparâmetros
O teste para obtenção dos melhores hiperparâmetros é feito pela classe BestParamsTestSuite, que recebe como parâmetro no construtor um dicionário, com as configurações do teste:

* **verbose**: se for True, exibe no console os melhores parâmetros encontrados para cada modelo;
* **output_path**: o caminho de um arquivo json que, se fornecido, irá armazenar a saída da busca pelos melhores hiperparâmetros.
* **n_splits**: número de folds do cross validation: padrão=4
* **n_jobs**: número de jobs rodando em paralelo: padrão=-1

O teste é realizado através do método run( ), que possui os seguintes parâmetros:

* **x**: implementação de uma matriz contendo as features  
* **y**: implementação de uma matriz contendo os rótulos (labels)
* **model_parameters**:  um dicionário contendo como chave um objeto instanciado e como valor um outro dicionário, que por sua vez contém o nome do hiperparâmetro como chave e uma lista de valores a serem testados como valor. Exemplo:
```python
model_parameters = {
	KNeighborsClassifier(): {  
		'n_neighbors': [3, 5],  
		'p': [1, 2]
	},  
	RandomForestClassifier(): {  
		'n_estimators': [100, 200],  
		'max_features': ['auto', 'sqrt']
	}
}
```
---
**--|> Nota:**
Neste exemplo, os modelos utilizados são de implementações do scikit-learn. Porém, o programa aceita qualquer implementação de modelo, desde que forneça os seguintes métodos, de acordo com a categoria do modelo:

**Estimator**: objeto base, implementa um método um dos seguintes métodos fit
```python
estimator.fit(data, targets) # aprendizado supervisionado
estimator.fit(data) # aprendizado não-supervisionado
```
**Predictor**: preditor, para problemas de aprendizado supervisionado
```python
predictor.predict(data) ## problemas de resultados contínuos
predictor.predict_proba(data) ## problemas de resultados discretos
```
**Transformer**: filtragem e transformação de dados
```python
transformer.transform(data)
transformer.fit_transform(data)
```
**Model**: um modelo que pode fornecer índices de qualidade e adaptação aos dados, considerando alguma medida.
```python
model.score(data)
```
É recomendado que quaisquer implementações além das fornecidas pelo scikit-learn implementem, pelo menos, os métodos de Estimator, Predictor e Model.

---
**test_parameters**: um dicionário contendo os parâmetros utilizados pelo teste, propriamente dito. Pode receber os seguintes valores:

## Testando os modelos
O teste dos modelos é feito através da classe ParameterizedTestSuite, que recebe como parâmetro no construtor um dicionário, com as configurações do teste:
* **stratify**: se True, obtém amostras estratificadas de cada classe no vetor de features. Recomendado em problemas de classificação.
* **n_splits**: número de folds do cross validation: padrão=4
* **float_precision**: precisão do resultado, em casas decimais: padrão=3
* **problem_type**: tipo de problema a qual se aplica o teste (classificação e regressão)

O teste é realizado através do método run( ), que possui os seguintes parâmetros:
* **x**: implementação de uma matriz contendo as features ;
* **y**: implementação de uma matriz contendo os rótulos (labels);
* **models**: uma lista de modelos instanciados;

O método fornece, como resultado, um DataFrame contendo a  média e desvio padrão de acurácia, precisão e revocação (recall) para cada um dos modelos.
