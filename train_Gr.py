import tensorflow as tf
import numpy as np
import json
import os

# 載入 Fashion-MNIST 資料集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# 正規化圖片至 [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 建立模型
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 編譯模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 訓練模型
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# 評估模型
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')

# 保存模型為 .h5 格式
model.save('fashion_mnist.h5')

# 提取模型架構
model_arch = []
for layer in model.layers:
    layer_config = {
        'name': layer.name,
        'type': layer.__class__.__name__,
        'config': layer.get_config(),
        'weights': [w.name for w in layer.weights]
    }
    model_arch.append(layer_config)

# 保存架構至 fashion_mnist.json
with open('fashion_mnist.json', 'w') as f:
    json.dump(model_arch, f, indent=2)

# 提取並保存權重至 fashion_mnist.npz
weights_dict = {}
for layer in model.layers:
    for weight in layer.weights:
        weights_dict[weight.name] = weight.numpy()

np.savez('fashion_mnist.npz', **weights_dict)

# 移動檔案至 ./model 資料夾
os.makedirs('model', exist_ok=True)
os.rename('fashion_mnist.json', 'model/fashion_mnist.json')
os.rename('fashion_mnist.npz', 'model/fashion_mnist.npz')