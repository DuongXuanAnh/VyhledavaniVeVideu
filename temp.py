from minisom import MiniSom
import numpy as np

# Náhodná trénovací data
data = np.random.rand(100, 5)

print(data)

# Inicializace SOM
som = MiniSom(7, 7, 5, sigma=1.0, learning_rate=0.5)

# Trénování SOM
som.train_random(data, 100)

# Výstup
for (x, y, z, w, v) in data:
    w_x, w_y = som.winner([x, y, z, w, v])
    print(f'x={x:.2f}, y={y:.2f}, z={z:.2f}, w={w:.2f}, v={v:.2f} -> klaster ({w_x}, {w_y})')
