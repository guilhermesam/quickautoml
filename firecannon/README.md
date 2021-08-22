
# Getting Started
testsuite é uma ferramenta utilizada em conjunto com outros scripts Python que permite:
* Encontrar os melhores hiperparâmetros para um conjunto de modelos;
* Testar diferentes modelos, através de métricas como acurácia, precisão e recall, dado um conjunto de dados em comum.

## Como utilizar
* Clonar o repositório
```bash
git clone https://github.com/Malware-Hunter/guilherme_samuel
```
* Instalar a biblioteca
```bash
cd guilherme_samuel
sh distribute.sh
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
