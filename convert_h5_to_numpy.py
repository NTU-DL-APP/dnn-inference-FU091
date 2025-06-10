import tensorflow as tf
import numpy as np
import json

# 載入模型
model = tf.keras.models.load_model("fashion_mnist.h5")

# 轉換架構
model_arch = []
for i, layer in enumerate(model.layers):
    layer_config = layer.get_config()
    layer_name = f"layer_{i}"
    layer_type = type(layer).__name__
    
    if layer_type not in ["Dense", "Flatten"]:
        raise ValueError(f"Unsupported layer: {layer_type}")
    
    # 欄位轉換
    arch_entry = {
        "name": layer_name,
        "type": layer_type,
        "config": {},
        "weights": []
    }

    if layer_type == "Dense":
        arch_entry["config"]["activation"] = layer_config["activation"]
        arch_entry["weights"] = [f"{layer_name}_W", f"{layer_name}_b"]

    model_arch.append(arch_entry)

# 儲存架構 JSON
with open("model/fashion_mnist.json", "w") as f:
    json.dump(model_arch, f)

# 儲存權重 NPZ
weights_dict = {}
for i, layer in enumerate(model.layers):
    if isinstance(layer, tf.keras.layers.Dense):
        W, b = layer.get_weights()
        weights_dict[f"layer_{i}_W"] = W
        weights_dict[f"layer_{i}_b"] = b

np.savez("model/fashion_mnist.npz", **weights_dict)
