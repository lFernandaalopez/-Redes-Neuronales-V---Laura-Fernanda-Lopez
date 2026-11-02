# Redes Neuronales V — Reconocimiento facial con redes siamesas

**Especialización en Inteligencia Artificial · Jahir Ávila**

Red neuronal siamesa en **PyTorch** entrenada con **Triplet Loss** y minería de tripletas en línea, y aplicada a
verificación e identificación facial con **zero-shot, one-shot y few-shot learning**.

## Notebooks (Google Colab)

| # | Notebook | Contenido |
|---|---|---|
| 01 | `01_dataset_y_preprocesamiento_facial.ipynb` | Dataset de prueba, detección (Haar), alineación por ojos, recorte 160×160, CLAHE, control de calidad, aumento de datos |
| 02 | `02_red_siamesa_triplet_loss.ipynb` | Red siamesa desde cero (CNN 128-d), Triplet Loss *batch-all / semi-hard / batch-hard*, evaluación en personas no vistas |
| 03 | `03_reconocimiento_facial_one_few_zero_shot.ipynb` | Verificación 1:1, identificación 1:N one/few-shot, conjunto abierto (rechazo de desconocidos), FaceNet y cabeza ajustada con Triplet Loss |

**Ejecutar en Colab:** abra el notebook desde GitHub (Archivo → Abrir cuaderno → GitHub) y, en la primera celda,
reemplace `REPO_URL` por la URL de este repositorio. Orden: 02 → 01 → 03 (el 03 usa el modelo guardado por el 02 y
los rostros procesados por el 01).

## Datos

- **Entrenamiento y prueba:** AT&T / ORL Database of Faces (AT&T Laboratories Cambridge; 40 personas × 10 fotos), base
  pública para uso académico. 30 personas para entrenar y 10 nunca vistas para probar (8 registradas + 2 desconocidas).
- **Fotos propias:** cree `dataset_propio/<persona>/` (fotos propias o de personas que autoricen su uso académico) y en
  el notebook 01 cambie `CARPETA = Path("dataset_propio")`. Esta carpeta está en `.gitignore` por privacidad.
- **FaceNet:** pesos InceptionResnetV1-VGGFace2 (facenet-pytorch); se descargan automáticamente (112 MB, no se suben).

## Resultados

| Experimento | Resultado |
|---|---|
| Red propia, 10 personas no vistas (notebook 02) | AUC 0,951 · EER 10,2 % · 1-shot 74,8 % · 5-shot 82,2 % (azar 10 %) |
| Minería de tripletas (AUC final) | batch-all 0,949 · semi-hard 0,951 · batch-hard 0,943 (colapsa: pérdida = α) |
| Verificación dataset de prueba — red propia | AUC 0,818 · EER 25,2 % |
| Verificación dataset de prueba — FaceNet | AUC 0,988 · EER 1,0 % |
| Identificación one-shot (8 personas) — FaceNet | 100 % · 2 personas desconocidas rechazadas al 100 % |

## Estructura

```
├── src/          preprocesamiento.py · modelos.py · evaluacion.py
├── notebooks/    01 · 02 · 03 (ejecutados, con salidas)
├── datos/publico/att_faces/
├── modelos/      clasificadores Haar · red_siamesa_att.pt
├── figuras/  ·  resultados/
└── requirements.txt
```

## Referencias

- Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A unified embedding for face recognition and clustering. *CVPR*.
- Koch, G., Zemel, R., & Salakhutdinov, R. (2015). Siamese neural networks for one-shot image recognition. *ICML Deep Learning Workshop*.
- Hermans, A., Beyer, L., & Leibe, B. (2017). In defense of the triplet loss for person re-identification. *arXiv:1703.07737*.
- Samaria, F., & Harter, A. (1994). Parameterisation of a stochastic model for human face identification. *IEEE WACV*.
