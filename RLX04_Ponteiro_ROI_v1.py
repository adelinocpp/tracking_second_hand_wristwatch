#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 16:05:00 2025

@author: adelino
"""

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import dill
# from dill import dumps, loads
# -----------------------------------------------------------------------------
# Função para calcular o ângulo entre dois pontos
def calcular_angulo(p1, p2):
    delta_x = p2[0] - p1[0]
    delta_y = p2[1] - p1[1]
    angulo_rad = math.atan2(delta_y, delta_x)
    angulo_deg = math.degrees(angulo_rad)
    return angulo_deg
# -----------------------------------------------------------------------------
# The resize_image function has the function of changing the size while maintaining the aspect ratio, with the longest side being 1000 pixels
def resize_image(img):
    height, width, _ = img.shape
    # Determine the scaling factor to make the longer side 1000 pixels
    scale_factor = 1000 / max(height, width)
    # Resize the image while preserving the aspect ratio
    img = cv2.resize(img, (int(width * scale_factor), int(height * scale_factor)))
    return img
# -----------------------------------------------------------------------------
# Carregar o vídeo
# video_path = '/home/adelino/MEGAsync/Rolex/output_video.mp4'
video_path = '/home/adelino/MEGAsync/Rolex/IMG_5480.MOV'
xa = 85
ya = 720
xb = xa+768
yb = ya+768

cap = cv2.VideoCapture(video_path)
# Verificar se o vídeo foi aberto corretamente
if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

# iniFrame = 94936 + 240
iniFrame = 182
cap.set(cv2.CAP_PROP_POS_FRAMES, iniFrame)

# Ler o primeiro frame
ret, frame = cap.read()
if not ret:
    print("Erro ao ler o primeiro frame.")
    exit()

frame = frame[ya:yb, xa:xb,:]
frame = resize_image(frame)
# cv2.imwrite("teste.jpg", frame)
centro = (502,506)
# Selecionar a ROI (Região de Interesse) do ponteiro no primeiro frame
# roi = cv2.selectROI("Selecione o ponteiro", frame, fromCenter=False, showCrosshair=True)
# cv2.destroyWindow("Selecione o ponteiro")#     main(args.input_dir, args.output_dir)



# Read image



gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# Aplicar um filtro Gaussiano para suavizar a imagem
blur = cv2.GaussianBlur(gray, (5, 5), 0)
# Aplicar um filtro Canny para detectar bordas
edges = cv2.Canny(blur, 50, 150)
# Aplicar uma operação morfológica de dilatação para realçar o ponteiro
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(edges, kernel, iterations=1)

# Inicializar o tracker (usando CSRT como exemplo)
tracker = cv2.TrackerCSRT_create()
# tracker = cv2.TrackerKCF_create()
# tracker.init(dilated, roi)
k = 0
fps = cap.get(cv2.CAP_PROP_FPS)
min_comp_linha = 150  # Comprimento mínimo da linha para ser considerada o ponteiro

cv2.startWindowThread()
listFrames = []
time = []
for i in range(0, int(3*fps)):
    ret, frame = cap.read()
    
    if not ret:
        break
    frame = frame[ya:yb, xa:xb,:]
    frame = resize_image(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Aplicar um filtro Gaussiano para suavizar a imagem
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Aplicar um filtro Canny para detectar bordas
    edges = cv2.Canny(blur, 50, 150)
    # Aplicar uma operação morfológica de dilatação para realçar o ponteiro
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Atualizar o tracker para encontrar a nova posição do ponteiro
    # success, roi = tracker.update(edges)
    success = True
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=150, maxLineGap=10)
    k +=1
    if lines is not None:
        # Filtro para encontrar a linha mais longa (provavelmente o ponteiro)
        linha_ponteiro = None
        max_comp = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            comp = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            # ang_t = calcular_angulo((x1,y1),centro)
            # if (ang_t < 180) or (ang_t > 150):
            #     continue
            if comp > max_comp:
                max_comp = comp
                linha_ponteiro = line[0]
                

    if success and (linha_ponteiro is not None):
        x1, y1, x2, y2 = linha_ponteiro
        # Desenhar a linha do ponteiro
        # cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Calcular o ponto médio da linha (ponta do ponteiro)
        #ponta = ((x1 + x2) // 2, (y1 + y2) // 2)
        ponta = x1, y1
        
        # Desenhar a ROI (retângulo ao redor do ponteiro)
        # (x, y, w, h) = tuple(map(int, roi))
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 128, 0), 2)

        # Calcular o ponto central da ROI (ponta do ponteiro)
        # ponta = (x + w // 2, y + h // 2)

        # Desenhar uma linha do centro do relógio até a ponta do ponteiro
        cv2.line(frame, centro, ponta, (0, 128, 0), 2)
        
        # Calcular o ângulo do ponteiro
        angulo = calcular_angulo(centro, ponta)
        if (angulo > 120) or (angulo < 90):
            continue
        print(f"Ângulo: {angulo:.1f}° {k:d}; C: {centro[0]:.1f},{centro[1]:.1f}; p: {ponta[0]:.1f},{ponta[1]:.1f}")
        listFrames.append(angulo)
        time.append(i/fps)
        # Exibir o ângulo na tela
        cv2.putText(frame, f"Angulo: {angulo:.2f} °", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 127, 127), 2)
        
    # Exibir o frame
    cv2.imshow('Tracking do Ponteiro', frame)

    # Parar o loop se a tecla 'q' for pressionada
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

dictData = {
    "time":time,
    "tic":listFrames
    }

with open("RLX04_data_v1.pth", "wb") as dill_file:
    dill.dump(dictData, dill_file)
    
    
cm = 1/2.54
fig, ax = plt.subplots(layout='constrained',figsize =(15*cm, 10*cm))
plt.plot(time,listFrames,'r.',linewidth=1)
plt.xlabel('tempo (s)')
plt.ylabel('angulo (°)')
plt.ylim([90,120])

import numpy as np
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(np.array(time).reshape(-1, 1), np.array(listFrames).reshape(-1, 1))
reg.score(np.array(time).reshape(-1, 1), np.array(listFrames).reshape(-1, 1))

disp = listFrames - (time*reg.coef_ + reg.intercept_)

fig, ax = plt.subplots(layout='constrained',figsize =(15*cm, 10*cm))
plt.hist(disp[0,:], bins=int(np.sqrt(len(disp[0,:]))))