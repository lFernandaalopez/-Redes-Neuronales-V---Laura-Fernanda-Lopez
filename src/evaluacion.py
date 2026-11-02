"""
evaluacion.py — Protocolos de evaluación para reconocimiento facial basado en embeddings.

  * Verificación (1:1): ¿son la misma persona? -> distancia < τ. Métricas: ROC/AUC, exactitud,
    FAR (falsos aceptados) y FRR (falsos rechazados) al umbral elegido, EER.
  * Identificación (1:N) one-shot / few-shot: galería con k fotos por persona; el prototipo de cada
    persona es la media de sus k embeddings (renormalizada). Se asigna la persona más cercana.
  * Conjunto abierto: si la distancia mínima supera τ, la cara se rechaza como "desconocido".

Autor: Jahir Ávila — Especialización en Inteligencia Artificial.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def pares(etiquetas):
    """Todos los pares (i, j), i < j, y si son de la misma persona."""
    idx = list(combinations(range(len(etiquetas)), 2))
    i, j = np.array(idx).T
    return i, j, (np.asarray(etiquetas)[i] == np.asarray(etiquetas)[j])


def distancias_pares(emb, i, j):
    return np.linalg.norm(emb[i] - emb[j], axis=1)


def verificacion(dist, misma, umbral=None):
    """Métricas de verificación. Si umbral es None se elige el que maximiza la exactitud balanceada."""
    puntaje = -dist  # menor distancia = más probable misma persona
    auc = roc_auc_score(misma, puntaje)
    fpr, tpr, thr = roc_curve(misma, puntaje)
    fnr = 1 - tpr
    k = np.nanargmin(np.abs(fnr - fpr))
    eer, umbral_eer = float((fpr[k] + fnr[k]) / 2), float(-thr[k])
    if umbral is None:
        candidatos = np.unique(dist)
        bal = [((dist[misma] < u).mean() + (dist[~misma] >= u).mean()) / 2 for u in candidatos]
        umbral = float(candidatos[int(np.argmax(bal))])
    acepta = dist < umbral
    far = float(acepta[~misma].mean())          # impostores aceptados
    frr = float((~acepta[misma]).mean())        # genuinos rechazados
    return {"auc": float(auc), "eer": eer, "umbral_eer": umbral_eer, "umbral": float(umbral),
            "exactitud": float((acepta == misma).mean()), "exactitud_balanceada": float(1 - (far + frr) / 2),
            "far": far, "frr": frr, "n_genuinos": int(misma.sum()), "n_impostores": int((~misma).sum()),
            "roc": (fpr, tpr)}


def prototipos(emb, etiquetas, indices_galeria):
    """Embedding medio (renormalizado) por persona usando solo las imágenes de la galería."""
    etiquetas = np.asarray(etiquetas)
    personas = sorted(set(etiquetas[indices_galeria]))
    P = []
    for p in personas:
        m = emb[[i for i in indices_galeria if etiquetas[i] == p]].mean(0)
        P.append(m / np.linalg.norm(m))
    return personas, np.stack(P)


def identificar(emb_consulta, personas, P, umbral=None):
    d = np.linalg.norm(emb_consulta[:, None, :] - P[None, :, :], axis=2)
    k = d.argmin(1)
    pred = np.array(personas, dtype=object)[k]
    dmin = d[np.arange(len(k)), k]
    if umbral is not None:
        pred = np.where(dmin < umbral, pred, "desconocido")
    return pred, dmin, d


def few_shot(emb, etiquetas, k, repeticiones=50, semilla=0, umbral=None):
    """Exactitud de identificación con k imágenes de galería por persona (resto = consultas).

    Se repite con galerías aleatorias y se reporta media y desviación.
    """
    rng = np.random.default_rng(semilla)
    etiquetas = np.asarray(etiquetas)
    accs = []
    for _ in range(repeticiones):
        gal, con = [], []
        for p in sorted(set(etiquetas)):
            idx = rng.permutation(np.where(etiquetas == p)[0])
            if len(idx) <= k:
                continue
            gal += list(idx[:k]); con += list(idx[k:])
        personas, P = prototipos(emb, etiquetas, gal)
        pred, _, _ = identificar(emb[con], personas, P, umbral)
        accs.append(float((pred == etiquetas[con]).mean()))
    return float(np.mean(accs)), float(np.std(accs))
