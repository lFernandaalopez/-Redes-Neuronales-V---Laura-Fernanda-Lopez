"""
modelos.py — Red siamesa (red de embeddings con pesos compartidos) y Triplet Loss en PyTorch.

Idea central de la red siamesa
------------------------------
Una sola red f_θ (la "torre") convierte cada imagen en un vector (embedding) de dimensión d,
normalizado a norma 1. Las ramas de la red siamesa (2 para pares, 3 para tripletas) son
la MISMA red: comparten todos los pesos. Dos rostros son de la misma persona si la distancia
||f(x1) - f(x2)|| es menor que un umbral τ.

Triplet Loss (Schroff et al., 2015, FaceNet)
--------------------------------------------
Para una tripleta (ancla a, positivo p de la misma persona, negativo n de otra persona):
    L = max( ||f(a) - f(p)||² - ||f(a) - f(n)||² + α , 0 )
Obliga a que el negativo quede al menos un margen α más lejos que el positivo.
Las tripletas se minan EN LÍNEA dentro de cada mini-lote (batch-hard o semi-hard).

Autor: Jahir Ávila — Especialización en Inteligencia Artificial.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Torre de embeddings entrenada desde cero
# ---------------------------------------------------------------------------

def _bloque(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2))


class RedEmbedding(nn.Module):
    """CNN compacta: imagen (1 x 96 x 96) -> embedding de `dim` componentes con norma L2 = 1.

    4 bloques conv (32-64-128-256 filtros) + pooling global + capa densa de proyección.
    """

    def __init__(self, dim: int = 128, canales: int = 1, dropout: float = 0.3):
        super().__init__()
        self.extractor = nn.Sequential(_bloque(canales, 32), _bloque(32, 64), _bloque(64, 128), _bloque(128, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proyeccion = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, dim))

    def forward(self, x):
        return F.normalize(self.proyeccion(self.pool(self.extractor(x))), p=2, dim=1)


class RedSiamesa(nn.Module):
    """Envoltura explícita de la arquitectura siamesa: varias entradas, UNA sola torre compartida."""

    def __init__(self, torre: nn.Module):
        super().__init__()
        self.torre = torre

    def forward(self, *entradas):
        return [self.torre(x) for x in entradas]  # mismos pesos para cada rama

    @torch.no_grad()
    def distancia(self, x1, x2):
        e1, e2 = self.forward(x1, x2)
        return (e1 - e2).norm(dim=1)


# ---------------------------------------------------------------------------
# Triplet Loss con minería en línea
# ---------------------------------------------------------------------------

def distancias_cuadradas(e: torch.Tensor) -> torch.Tensor:
    """Matriz D[i, j] = ||e_i - e_j||² para todos los pares del lote."""
    prod = e @ e.T
    norma = prod.diag()
    return (norma[:, None] - 2 * prod + norma[None, :]).clamp_min(0)


class TripletLoss(nn.Module):
    """Triplet Loss con minería en línea.

    estrategia="batch_hard": para cada ancla toma el positivo MÁS lejano y el negativo MÁS cercano
                             del lote (Hermans et al., 2017).
    estrategia="semi_hard":  negativo más cercano que cumpla d(a,p) < d(a,n) < d(a,p) + α
                             (Schroff et al., 2015).
    estrategia="batch_all":  promedia todas las tripletas válidas del lote que violan el margen
                             (más estable al inicio del entrenamiento).
    """

    def __init__(self, margen: float = 0.2, estrategia: str = "batch_hard"):
        super().__init__()
        self.margen, self.estrategia = margen, estrategia

    def forward(self, emb, etiquetas):
        D = distancias_cuadradas(emb)
        misma = etiquetas[:, None] == etiquetas[None, :]
        eye = torch.eye(len(etiquetas), dtype=torch.bool, device=emb.device)
        pos_mask, neg_mask = misma & ~eye, ~misma
        grande = D.max().detach() + 1.0
        if self.estrategia == "batch_hard":
            d_ap = (D * pos_mask).max(dim=1).values
            d_an = (D + grande * (~neg_mask)).min(dim=1).values
            perdidas = F.relu(d_ap - d_an + self.margen)
            validas = pos_mask.any(1)
            return perdidas[validas].mean(), self._stats(perdidas[validas], d_ap[validas], d_an[validas])
        a_idx, p_idx = torch.where(pos_mask)                   # todos los pares (ancla, positivo)
        d_ap = D[a_idx, p_idx]
        d_an_todas = D[a_idx]                                  # (n_pares, B)
        nm = neg_mask[a_idx]
        if self.estrategia == "batch_all":
            # todas las tripletas válidas; se promedian solo las que aún violan el margen
            L = F.relu(d_ap[:, None] - d_an_todas + self.margen) * nm
            activas = (L > 1e-12).float()
            perdida = L.sum() / activas.sum().clamp_min(1.0)
            d_an = (d_an_todas * nm).sum(1) / nm.sum(1)
            return perdida, self._stats(L[nm], d_ap, d_an)
        # semi-hard: negativo más cercano que esté MÁS LEJOS que el positivo pero dentro del margen.
        # Si no existe, se usa el negativo "menos difícil" entre los que están más cerca que el positivo
        # (mismo criterio que la implementación de TensorFlow Addons).
        fuera = nm & (d_an_todas > d_ap[:, None])
        d_fuera = (d_an_todas + grande * (~fuera)).min(1).values
        d_dentro = (d_an_todas * (nm & ~fuera)).max(1).values
        d_an = torch.where(fuera.any(1), d_fuera, d_dentro)
        perdidas = F.relu(d_ap - d_an + self.margen)
        return perdidas.mean(), self._stats(perdidas, d_ap, d_an)

    @staticmethod
    def _stats(perdidas, d_ap, d_an):
        perdidas, d_ap, d_an = perdidas.detach(), d_ap.detach(), d_an.detach()
        return {"frac_activas": float((perdidas > 0).float().mean()),
                "d_ap": float(d_ap.mean()), "d_an": float(d_an.mean())}


# ---------------------------------------------------------------------------
# Torre preentrenada FaceNet (InceptionResnetV1, VGGFace2) para comparar
# ---------------------------------------------------------------------------
URL_VGGFACE2 = "https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt"


def cargar_facenet(ruta_pesos=None):
    """InceptionResnetV1 (facenet-pytorch) con pesos VGGFace2. Entrada RGB 160x160 en [-1, 1]; salida 512-d con norma 1."""
    from pathlib import Path
    from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
    ruta = Path(ruta_pesos) if ruta_pesos else Path(__file__).resolve().parents[1] / "modelos" / "vggface2.pt"
    if not ruta.exists():
        ruta.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(URL_VGGFACE2, str(ruta))
    modelo = InceptionResnetV1(pretrained=None, classify=False, num_classes=8631)
    estado = torch.load(ruta, map_location="cpu")
    modelo.load_state_dict(estado, strict=False)  # las capas 'logits' de clasificación no se usan
    return modelo.eval()


class CabezaProyeccion(nn.Module):
    """Capa de adaptación (512 -> dim) entrenable con Triplet Loss sobre embeddings congelados de FaceNet."""

    def __init__(self, entrada=512, dim=128):
        super().__init__()
        self.lin = nn.Linear(entrada, dim)

    def forward(self, x):
        return F.normalize(self.lin(x), p=2, dim=1)
