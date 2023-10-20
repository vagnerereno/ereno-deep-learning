import time
import numpy as np
from keras.src.layers import Dropout, BatchNormalization
from keras.src.optimizers.adam import Adam
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import Callback
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

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

def decision_tree(X_train, y_train, X_test):
    # Definição do modelo DecisionTree (usando o algoritmo CART)
    model = DecisionTreeClassifier(criterion='gini')  # criterion='gini' ou 'entropy'

    # Treinamento do modelo DecisionTree
    model.fit(X_train, y_train)

    # Fazer previsões
    y_pred = model.predict(X_test)

    return y_pred

def mlp_classifier(X_train, y_train, X_test):
    model = MLPClassifier(max_iter=5, batch_size=1000, alpha=1e-4, activation='relu', solver='adam', verbose=10,
                          tol=1e-4, random_state=1)

    training_times = []
    testing_times = []
    start_total = time.time()
    num_runs = 1;
    for _ in range(num_runs):
        # Medindo o tempo de treinamento
        start = time.time()
        history = model.fit(X_train, y_train)
        end = time.time()
        training_times.append(end - start)

        # Medindo o tempo de teste
        start_test = time.time()
        y_pred_mlp = model.predict(X_test)
        end_test = time.time()
        testing_times.append(end_test - start_test)

    end_total = time.time()
    total_time = end_total - start_total
    print("Training time: ", training_times[0])

    return history, y_pred_mlp

def ann_mlp_multiclass(X_train, y_train, X_test):
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
            num_classes,               # Para problemas multiclass, a saída do modelo deve ser igual o número de classes existentes.
            activation='softmax'       # Para problemas multiclasse, usar a função de ativação softmax.
        )
    )

    optimizer = Adam(learning_rate=0.001)

    # Para problemas multiclasse usar a função de perda sparse_categorical_crossentropy.
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'], run_eagerly=False)
    print(model.summary())

    # Treinamento do modelo
    history = model.fit(X_train, y_train, epochs=1, batch_size=64)

    # Fazer previsões
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    return history, y_pred

def ann_mlp_binary(X_train, y_train, X_test):

    model = Sequential()  # Modelos sequenciais são uma pilha linear de camadas
    model.add(
        Dense(
            # Adiciona uma camada densa (ou camada totalmente) ao modelo. Cada neurônio está conectado a cada neurônio na camada anterior e na camada seguinte.
            64,  # A camada tem 64 unidades (neurônios)
            input_dim=X_train.shape[1],  # Epecifica o número de características de entrada
            activation='relu' # Função de ativação ReLU (Rectified Linear Unit).
            # Usada para adicionar não-linearidade ao modelo. Substitui valores negativos na matriz de saída por 0.
        )
    )

    model.add(   # Adiciona outra camada
        Dense(   # A camada é densa. Ou seja, cada neurônio está conectado a cada neurônio na camada anterior e na camada seguinte.
            1,  # Se o problema for binário a camada tem 1 unidade de saída (neurônio)
            activation='sigmoid' # Função de ativação sigmoid para problemas de classificação binária.
            # Mapeia valores de entrada para um valor entre 0 e 1, que pode ser interpretado como uma probabilidade.
        )
    )

    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'], run_eagerly=False) # Para problemas de classificação binária.
    print(model.summary())
    print("X_train Head: ", X_train.shape)
    print("y_train Head: ", y_train.dtype)

    # Treinamento do modelo
    history = model.fit(X_train, y_train, epochs=1, batch_size=64)

    # Fazer previsões
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)

    return history, y_pred

def post_process_predictions(y_pred_multiclass, normal_class_label):
    """
    Transforma as previsões multiclasse em binárias.
    Qualquer previsão que não seja "normal" é considerada "ataque".
    """
    y_pred_binary = np.where(y_pred_multiclass != normal_class_label, 4, 5)  # Retornando 4 para "ataque masquerade" e 5 para "normal".
    return y_pred_binary

