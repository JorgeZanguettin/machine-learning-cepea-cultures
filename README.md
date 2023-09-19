<h1 style="text-align: center;">Machine Learning - CEPEA Cultures Regressor</h1>
<h3 style="text-align: center;">Python (Pandas/XGBoost/Requests/BS4)</h3>

&emsp;O objetivo desse projeto e utilizar a grande variedade de indicadores do agronegocio coletados do renomado site do Centro de Estudos Avançados em Economia Aplicada (CEPEA) para o treino e contrucao de modelos matematicos regressores preditivos utilizando a linguegem Python e suas diversas bibliotecas de Machine Learning.

## Avaliação dos modelos

<img src="https://portfolio-resumes.s3.amazonaws.com/CEPEA Model Evaluation.png">

## Instruções

&emsp;Todas as culturas compativeis com o projeto estao listados no arquivo **cultures.json**.

1. Configure seu ambiente, no caso, utilizei o **Python 3.11.4**;<br><br>
2. Certifique-se que instalou todas as dependências listadas no **requirements.txt**;<br><br>
3. Execute o arquivo **main.py --culture { culture_alias } --id { culture_id }**<br>(Obtenha o culture_alias e culture_id no arquivo **cultures.json**);<br><br>
4. Após o treino e armazenamento do modelo treinado, basta utilizar a função **model_prediction** sempre que desejar realizar novas predições;<br><br>
5. Voce pode configurar a quantidade de dias que serao utilizados no treinamento e na predicao modificando a variavel **time_steps** e tambem o numero de iteracoes do modelo matematico atraves da variavel **n_estimators**;<br><br>
5. **ENJOY!!**

<br><br>**Siga meu perfil!**
