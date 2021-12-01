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

## Como utilizar?
Firecannon implementa duas abstrações de estimadores, denominados `Classifier` e `Regressor`, para problemas de classificação e regressão, respectivamente. Ambas as classes podem ser acessadas de maneira demonstrada pelo exemplo abaixo:

```python
from sklearn.datasets import make_classification # dataset de exemplo
from firecannon.estimators import Classifier, Regressor

X_c, y_c = make_classification(n_classes=2, n_features=5, n_samples=100)

my_classifier = Classifier()
my_classifier.fit(X_c, y_c)
print(my_classifier.best_model) # por exemplo, KNeighborsClassifier(n_neighbors=3)
```

## Como funciona?
Firecannon objetiva identificar um modelo adequado para determinado problema considerando um balanço entre tempo de execução e desempenho. Para isso, internamente, a Firecannon seleciona esse modelo através da definição de uma lista de candidatos possíveis. Estes candidatos podem ser o mesmo algoritmo, porém com configurações de hiper-parâmetros diferentes (por exemplo, várias versões de Random Forest, com número de estimadores diferentes entre si). Esta lista, no entanto, pode ser alterada conforme as [especificações do usuário](#personalizando-a-execução).

### Conceitos-chave
* **NaiveModel**: Estrutura de dados referente a um modelo "ingênuo", isto é, que ainda não teve seus hiper-parâmetros ajustados. É utilizado para a definição de modelos personalizados pelo usuário, em detrimento dos algoritmos-padrão definidos internamente pela ferramenta. Para ser instanciado, necessita das seguintes informações: 
	* name: o nome do algoritmo;
	* estimator: uma instância de algoritmo que será utilizado como candidato a modelo. Consulte [criando modelos personalizados](#criando-modelos-personalizados) ou a [documentação](https://scikit-learn.org/stable/developers/develop.html) do scikit-learn para informações sobre como criar modelos personalizados.

* **Hyperparameter**: Representa um hiper-parâmetro (i.e. valor que controla o processo de aprendizado de um algoritmo). Foi construído para suportar o processo de ajuste de hiper-parâmetros, portanto, armazena os seguintes campos:
	* name: o nome do hiper-parâmetro;
	* data_type: o tipo de dado que o hiper-parâmetro espera receber;
	* min_value: o valor inferior do espaço de busca;
	* max_value: o valor superior do espaço de busca;	 

* **FittedModel**: Estrutura de dados que estende as informações do ```NaiveModel```. Representa um modelo com seus hiper-parâmetros ajustados. É retornada ao final do treinamento como o modelo representante do conjunto de dados. Em adição aos dados de ```NaiveModel```, possui:
	* cv_score: a pontuação (score) obtida pelo modelo;

## Personalizando a execução
Firecannon foi construído de forma a fornecer uma abstração de alto nível de treinamento de modelos, portanto, não requer que o usuário interaja com detalhes de baixo nível, como definição de algoritmos ou ajuste de hiper-parâmetros. Entretanto, é possível modificar esses detalhes, através da especificação dos seguintes parâmetros ao objeto estimador (```Classifier``` ou ```Regressor```):

* **model_settings: dict, default=None**: Especifica um dicionário de modelos personalizados, a qual deve conter instâncias de modelos como chaves e um dicionário de hiper-parâmetros **válidos** como valores. A estrutura de dicionário esperada por `model_settings` espera uma instância de NaiveModel, que representa um modelo com hiper-parâmetros ainda não ajustados, como chave do dicionário, e uma lista de objetos da classe Hyperparameter como instância. Consulte os [conceitos-chave](#conceitos-chave).
 
	```python	
	{
		NaiveModel(name='RandomForest Classifier', estimator=self._models_supplier.get_model('rf-c')): [
			Hyperparameter(name='n_estimators', data_type='int', min_value=10, max_value=60),
			Hyperparameter(name='min_samples_leaf', data_type='int', min_value=2, max_value=64)
		],
		NaiveModel(name='Adaboost Classifier', estimator=self._models_supplier.get_model('ada-c')): [
			Hyperparameter(name='n_estimators', data_type='int', min_value=10, max_value=100),
			Hyperparameter(name='learning_rate', data_type='float', min_value=0.1, max_value=2)
		]
	}
	```

* **report_type: str, default=None**: Caso um valor seja fornecido, define o método de geração de relatórios, que contém os resultados do processo de treinamento e validação dos modelos. Valores possíveis são 'csv', que resulta na criação de um arquivo .csv contendo os resultados de cada modelo em formato tabular, 'plot' que retorna os resultados em formato de um gráfico de barras e 'json', que retorna os resultados em um formato json, semelhante a um dicionário.

* **k_folds: int, default=5**: Controla o número de folds usados para executar o processo de validação cruzada, no ajuste por hiperparâmetros. Valores muito altos (> 10) tendem a fazer com que uma porcentagem muito pequena dos dados sejam utilizados para teste em cada etapa da validação cruzada.

* **n_jobs: int, default=-1**: Define o número de jobs que serão executados em paralelo. O valor default -1 significa que todos os processadores serão utilizados nessa tarefa.

* **random_state: int, default=777**: Define a semente que será utilizada para a geração dos números pseudo-aleatórios empregados no processo de treinamento e seleção de modelos. Mantido com o objetivo de reduzir a aleatoriedade dos experimentos.


## Criando modelos personalizados
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
