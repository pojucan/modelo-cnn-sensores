## Documentação do Projeto: Calibração Automática de Sensor via CNN

### Descrição geral

Este projeto demonstra um processo de calibração automática de sensores usando uma Rede Neural Convolucional (CNN) em 1D.
O objetivo é simular um sensor com ruído progressivamente menor e treinar uma CNN para corrigir as leituras, aproximando-as de uma função de calibração "verdadeira".

Durante o treinamento, o código:

- Gera leituras ruidosas que vão se tornando mais estáveis com o tempo;

- Treina uma CNN de forma incremental (época por época);

- Monitora métricas como MAE (erro absoluto médio) e R²;

- Gera gráficos de desempenho e calibração final.

### Requisitos do sistema

- **Python:** 3.10.12

**Sistema operacional:** Linux, Windows ou macOS

- **Hardware mínimo:**

- CPU com suporte a operações vetoriais

- 4 GB de RAM

- (GPU opcional para acelerar o treinamento TensorFlow)

### Instalação de dependências

Digite no terminal:

```bash
pip install -r requirements.txt
```

### 1. Geração de dados sintéticos

- **Função verdadeira do sensor:**
```python
def true_sensor_calibrated(x):
    return x**1.05 - 0.02*np.sin(5*x)
```
Essa função simula a resposta calibrada ideal de um sensor.

- **Adição de ruído:**

O ruído inicial é grande e diminui exponencialmente a cada época:

```python
noise_factor = noise_level_initial * np.exp(-0.1 * epoch)
```
Isso simula um sensor que vai sendo "ajustado" com o tempo.

### 2. Modelo de Rede Neural (CNN)

O modelo usa camadas convolucionais 1D para aprender padrões locais das leituras:

```python
model = Sequential([
    InputLayer(input_shape=(1,1)),
    Conv1D(16, kernel_size=3, activation='relu', padding='same'),
    Conv1D(16, kernel_size=3, activation='relu', padding='same'),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(1)
])
```

- **Entrada:** Leituras do sensor com shape (n amostras, 1, 1)

- **Saída:** Valor calibrado previsto

- **Função de perda:** MSE (erro quadrático médio)

- **Otimizador:** Adam

### 3. Treinamento Incremental

O código treina a CNN por 1 época de cada vez, com novas leituras a cada passo:

```python
for epoch in range(1, n_epochs+1):
    model.fit(X, y_current, epochs=1, verbose=0)
    ...
    y_current = generate_sensor_data(epoch).reshape(-1,1)
```

Durante o processo, são registradas:

- Média das intensidades previstas

- Número de anomalias (>5% de diferença do valor bruto)

- MAE e $R^{2}$

### 4. Métricas e Visualizações

O script gera quatro figuras principais:

**Média de intensidade ao longo das épocas**
Mostra a evolução da saída média da CNN.
Arquivos: media_intensidade_sensor.png / .pdf

**Número de anomalias detectadas**
Mede quantas previsões desviam significativamente.
Arquivos: anomalias_detectadas.png / .pdf

**Curva de calibração final**
Compara a saída da CNN com a linha ideal.
Arquivos: curva_calibracao_cnn.png / .pdf

**Métricas de desempenho (MAE e R²)**
Mostra a evolução das métricas ao longo do treino.
Arquivos: metricas_treinamento.png / .pdf

### 5. Saídas Geradas

MAE final: <#valor>\
R² final: <#valor>

Figuras salvas como:
- media_intensidade_sensor.png/.pdf
- anomalias_detectadas.png/.pdf
- curva_calibracao_cnn.png/.pdf
- metricas_treinamento.png/.pdf

### 6. Possíveis Extensões

- Adicionar dropout ou regularização para evitar overfitting.

- Testar arquiteturas diferentes, como redes totalmente conectadas ou LSTM.

- Implementar callback customizado para salvar checkpoints e métricas em CSV.

- Aprimorar visualizações, incluindo barras de erro e gráficos interativos.

- Refatorar o código de forma que ele utilize dados reais de sensoreds físicos

- Inseri-lo em uma estrutura de versionamento de dados e fluxo de ajuste de hiperparâmetros (MinIO, LakeFS e MLflow)

### 7. Licença e Créditos

**Autor:** Pojucan, M.M.S\
**Ano:** 2025