import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk

def euclidiana(p1, p2):  # distancia euclidiana
    return np.sqrt(np.sum((p1 - p2) ** 2))

def obtenerVecinos(X, point_idx, eps):
    vecinos = []
    for i in range(len(X)):
        if euclidiana(X[point_idx], X[i]) <= eps:
            vecinos.append(i)
    return vecinos

def dbscan(X, eps, min_samples):
    etiquetas = np.full(X.shape[0], -1)
    cluster_id = 0
    for i in range(len(X)):
        if etiquetas[i] != -1:  # saltar puntos ya etiquetados
            continue
        
        vecinos = obtenerVecinos(X, i, eps)
        
        if len(vecinos) < min_samples:
            etiquetas[i] = -1  # ruido
        else:
            etiquetas[i] = cluster_id
            etiquetas = expandirCluster(X, etiquetas, i, vecinos, cluster_id, eps, min_samples)
            cluster_id += 1
    return etiquetas

def expandirCluster(X, etiquetas, point_idx, vecinos, cluster_id, eps, min_samples):
    i = 0
    while i < len(vecinos):
        neighbor_idx = vecinos[i]
        if etiquetas[neighbor_idx] == -1:
            etiquetas[neighbor_idx] = cluster_id
        elif etiquetas[neighbor_idx] == -1:
            etiquetas[neighbor_idx] = cluster_id
            nuevo_vecinos = obtenerVecinos(X, neighbor_idx, eps)
            if len(nuevo_vecinos) >= min_samples:
                vecinos += nuevo_vecinos
        i += 1
    return etiquetas

def mostrar_informacion_clusters(unique_etiquetas, colors):
    ventana = tk.Tk()
    ventana.title("Código de color de Clústers")
    
    etiqueta_principal = tk.Label(ventana, text="Código de color de clústers", font=("Arial", 16))
    etiqueta_principal.pack(pady=10)

    canvas = tk.Canvas(ventana, width=400, height=200)
    canvas.pack()

    for idx, (k, col) in enumerate(zip(unique_etiquetas, colors)):
        color_str = f"RGB({int(col[0]*255)}, {int(col[1]*255)}, {int(col[2]*255)})"
        if k == -1:
            info = f"Ruido"
        else:
            info = f"Cluster {k}"   # luego va el circulo de color
        
        x = 50                      
        y = 30 + idx * 30
        canvas.create_oval(x, y, x+20, y+20, fill=f"#{int(col[0]*255):02x}{int(col[1]*255):02x}{int(col[2]*255):02x}")
        canvas.create_text(x + 50, y + 10, text=info, font=("Arial", 12))

    ventana.mainloop()

def verDatos(X, eps, min_samples):
    etiquetas = dbscan(X, eps, min_samples)

    print("Datos generados (en puntos):")
    print(X)
    print("\nEtiquetas (clusters):")
    print(etiquetas)

    unique_etiquetas = set(etiquetas)

    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_etiquetas))]

    plt.figure(figsize=(8, 6))
    for k, col in zip(unique_etiquetas, colors):
        if k == -1:
            col = [0, 0, 0, 1]  # ruido en color negro

        class_member_mask = (etiquetas == k)
        xy = X[class_member_mask]

        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=10)

    plt.title(f'DBSCAN ')
    plt.show()

    mostrar_informacion_clusters(unique_etiquetas, colors)

def ejecutarCorridas():
    print("Corrida 1")
    X1 = generar_datos_sinteticos(centers=3)
    verDatos(X1, eps=0.5, min_samples=5)
    
    print("Corrida 2")
    X2 = generar_datos_sinteticos(centers=4)
    verDatos(X2, eps=0.4, min_samples=4)
    
    print("Corrida 3")
    X3 = generar_datos_sinteticos(centers=5)
    verDatos(X3, eps=0.3, min_samples=6)

def generar_datos_sinteticos(centers): # está función sirve para generar datos aleatorios para ingresarse en el algoritmo
    np.random.seed(0)
    X = []
    for i in range(centers):
        centro = np.random.uniform(-10, 10, 2)
        cluster = np.random.randn(100, 2) * 0.5 + centro
        X.append(cluster)

    X = np.vstack(X)
    return X

ejecutarCorridas()