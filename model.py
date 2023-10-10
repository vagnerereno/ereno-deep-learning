import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import Callback

class BatchLogger(Callback):
    def on_train_begin(self, logs={}):
        self.epoch_losses = []
        self.epoch_accuracies = []
        self.losses = []
        self.accuracies = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('accuracy'))

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_losses.append(self.losses[:])
        self.epoch_accuracies.append(self.accuracies[:])
        self.losses = []
        self.accuracies = []

# Definição do modelo
def model(X_train, y_train, X_test):
    batch_logger = BatchLogger()
    num_classes = len(np.unique(y_train))
    # Definição do modelo
    model = Sequential()  # Modelos sequenciais são uma pilha linear de camadas
    model.add(
        Dense(
            # Adiciona uma camada densa (ou camada totalmente) ao modelo. Cada neurônio está conectado a cada neurônio na camada anterior e na camada seguinte.
            64,  # A camada tem 64 unidades (neurônios)
            input_dim=X_train.shape[1],  # Epecifica o número de características de entrada
            activation='relu'
            # Usa a função de ativação ReLU (Rectified Linear Unit). Usada para adicionar não-linearidade ao modelo. Substitui valores negativos na matriz de saída por 0.
        )
    )

    model.add(
        Dense(
            # Adiciona outra camada densa. Cada neurônio está conectado a cada neurônio na camada anterior e na camada seguinte.
            # 1,  # Se o problema for binário a camada tem 1 unidade de saída (neurônio)
            # activation='sigmoid'
            # Função de ativação sigmoid. Configuração para problemas de classificação binária. Mapeia valores de entrada para um valor entre 0 e 1, que pode ser interpretado como uma probabilidade.
            num_classes,               # Para problemas multiclass, a saída do modelo deve ser igual o número de classes existentes.
            activation='softmax'       # Para problemas multiclasse, usar a função de ativação softmax.
        )
    )

    # # Compilação do modelo Deep Learning
    # model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'], run_eagerly=False) # Para problemas de classificação binária.
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'], run_eagerly=False) # Para problemas multiclasse usar a função de perda sparse_categorical_crossentropy.
    print(model.summary())

    # Treinamento do modelo
    history = model.fit(X_train, y_train, epochs=1, batch_size=256, callbacks=[batch_logger])

    # Fazer previsões
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    return history, y_pred
