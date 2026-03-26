## Documentação do Projeto: Calibração Automática de Sensor via CNN

### Descrição geral

Este projeto demonstra um processo de calibração automática de sensores usando uma Rede Neural Convolucional (CNN) em 1D.
O objetivo é simular um sensor com ruído progressivamente menor e treinar uma CNN para corrigir as leituras, aproximando-as de uma função de calibração "verdadeira".

Durante o treinamento, o código:

- Gera leituras ruidosas que vão se tornando mais estáveis com o tempo;

- Treina uma CNN de forma incremental (época por época);

- Monitora métricas como MAE (erro absoluto médio) e R²;

- Gera gráficos de desempenho e calibração final.

### Metodologia

O pipeline segue os seguintes passos:

**1. Geração de dados sintéticos**

- Simula leituras de sensor com ruído decrescente ao longo do tempo

**2. Função de calibração real**

- Função não linear com perturbação senoidal

**3. Treinamento incremental**

- O modelo é treinado época a época

- O ruído diminui progressivamente (simulando estabilização do sensor)

**4. Monitoramento de métricas**

- MAE (Mean Absolute Error)

- $R^{2}$ (Coeficiente de determinação)

- Intensidade média

- Número de anomalias

### Estrutura do projeto

```bash
.
├── LICENSE.md
├── README.md
├── requirements.txt
├── src
│   └── model-cnn-sensor.py
└── visualization
    ├── graphics
    │   ├── anomalias_detectadas.png
    │   ├── curva_calibracao_cnn.png
    │   ├── media_intensidade_sensor.png
    │   └── metricas_treinamento.png
    ├── model-cnn-sensor.ipynb
    └── reports
        ├── anomalias_detectadas.pdf
        ├── curva_calibracao_cnn.pdf
        ├── media_intensidade_sensor.pdf
        └── metricas_treinamento.pdf
```

### 5. Requisitos do sistema

#### Tecnologias utilizadas

- Python 3.x
- TensorFlow / Keras
- NumPy
- Scikit-learn
- Matplotlib
- Linux, Windows ou macOS

#### Hardware mínimo:

- CPU com suporte a operações vetoriais

- 4 GB de RAM

- (GPU opcional para acelerar o treinamento TensorFlow)

### 6. Como executar

#### 6.1. Clonar repositório

```bash
git clone git@github.com:pojucan/modelo-cnn-sensores.git

cd modelo-cnn-sensores
```

#### 6.2. Criar ambiente virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

#### 6.3. Instalação de dependências

Digite no terminal:

```bash
pip install -r requirements.txt
```

#### 6.4. Executar o modelo

```bash
python3 python src/model-cnn-sensor.py
```

### 7. O que o programa faz?

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

- **Modelo de Rede Neural (CNN)**

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

- **Treinamento Incremental**

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

- **Métricas e Visualizações**

O script gera quatro figuras principais:

**Média de intensidade ao longo das épocas:**

- Mostra a evolução da saída média da CNN.
- Arquivos: media_intensidade_sensor.png / .pdf

**Número de anomalias detectadas:**

- Mede quantas previsões desviam 
- significativamente.
- Arquivos: anomalias_detectadas.png / .pdf

**Curva de calibração final**
Compara a saída da CNN com a linha ideal.
Arquivos: curva_calibracao_cnn.png / .pdf

**Métricas de desempenho (MAE e R²)**
Mostra a evolução das métricas ao longo do treino.
Arquivos: metricas_treinamento.png / .pdf

### 8. Saídas Geradas

MAE final: <#valor>\
R² final: <#valor>

#### 5.1. Os gráficos são salvos em: 

```bash
visualization/graphics/
visualization/reports/
```

#### Figuras salvas como:
- media_intensidade_sensor.png/.pdf
- anomalias_detectadas.png/.pdf
- curva_calibracao_cnn.png/.pdf
- metricas_treinamento.png/.pdf

### 9. Possíveis Extensões

- Adicionar dropout ou regularização para evitar overfitting.

- Testar arquiteturas diferentes, como redes totalmente conectadas ou LSTM.

- Implementar callback customizado para salvar checkpoints e métricas em CSV.

- Aprimorar visualizações, incluindo barras de erro e gráficos interativos.

- Refatorar o código de forma que ele utilize dados reais de sensoreds físicos

- Inseri-lo em uma estrutura de versionamento de dados e fluxo de ajuste de hiperparâmetros (MinIO, LakeFS e MLflow)

### 10. Licença e Créditos

**Autor:** Pojucan, M.M.S\
**Ano:** 2025