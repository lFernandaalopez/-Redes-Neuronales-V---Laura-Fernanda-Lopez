"""
preprocesamiento.py — Pipeline de preprocesamiento de imágenes faciales.

Etapas (en orden):
  1. Lectura y corrección de orientación EXIF (fotos de celular).
  2. Redimensionado de trabajo (lado mayor <= 1024 px) para acelerar la detección y relleno
     de borde del 20 % (replicado): permite detectar rostros que llenan casi todo el encuadre
     o tocan el borde (selfies muy cercanas, imágenes ya recortadas).
  3. Detección del rostro con un clasificador en cascada de Haar (Viola-Jones, OpenCV).
     Si hay varios, se conserva el de mayor área.
  4. Detección de los ojos dentro del rostro y ALINEACIÓN: rotación para que la línea
     entre los ojos quede horizontal (reduce la variación por inclinación de la cabeza).
  5. Recorte con margen y redimensionado al tamaño de entrada del modelo.
  6. Normalización fotométrica: CLAHE sobre la luminancia (compensa iluminación desigual).
  7. Conversión a tensor y estandarización (se hace en los notebooks / modelos).

Autor: Jahir Ávila — Especialización en Inteligencia Artificial.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

RAIZ = Path(__file__).resolve().parents[1]
MODELOS = RAIZ / "modelos"
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".pgm", ".heic"}


@dataclass
class ResultadoRostro:
    ruta: str
    persona: str
    ok: bool
    motivo: str = ""
    rostro: np.ndarray | None = None          # recorte final RGB uint8 (tam x tam)
    caja: tuple | None = None                 # (x, y, w, h) en la imagen de trabajo
    ojos: list = field(default_factory=list)  # centros de ojos detectados
    angulo: float = 0.0                       # rotación aplicada (grados)
    n_rostros: int = 0


def leer_imagen(ruta) -> np.ndarray:
    """Lee una imagen como RGB uint8 respetando la orientación EXIF."""
    ruta = Path(ruta)
    if ruta.suffix.lower() == ".heic":
        try:
            import pillow_heif  # noqa: F401
            pillow_heif.register_heif_opener()
        except ImportError as e:
            raise RuntimeError("Instale pillow-heif para leer fotos .heic") from e
    img = Image.open(ruta)
    img = ImageOps.exif_transpose(img).convert("RGB")
    return np.array(img)


class PreprocesadorRostros:
    """Detecta, alinea, recorta y normaliza rostros."""

    def __init__(self, tam: int = 160, margen: float = 0.25, clahe: bool = True, lado_max: int = 1024, relleno: float = 0.2):
        self.tam, self.margen, self.usar_clahe, self.lado_max, self.relleno = tam, margen, clahe, lado_max, relleno
        self.cascada_rostro = cv2.CascadeClassifier(str(MODELOS / "haarcascade_frontalface_default.xml"))
        self.cascada_ojos = cv2.CascadeClassifier(str(MODELOS / "haarcascade_eye_tree_eyeglasses.xml"))
        if self.cascada_rostro.empty() or self.cascada_ojos.empty():
            raise RuntimeError("No se encontraron los clasificadores Haar en la carpeta modelos/")
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # -- 2. redimensionado de trabajo -------------------------------------------------
    def _redimensionar(self, img):
        h, w = img.shape[:2]
        f = min(1.0, self.lado_max / max(h, w))
        img = cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA) if f < 1 else img
        b = int(self.relleno * max(img.shape[:2]))
        return cv2.copyMakeBorder(img, b, b, b, b, cv2.BORDER_REPLICATE) if b else img

    # -- 3. detección -----------------------------------------------------------------
    def detectar(self, gris):
        lado = min(gris.shape)
        caras = self.cascada_rostro.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=6,
                                                     minSize=(max(40, lado // 8),) * 2)
        return sorted([tuple(map(int, c)) for c in caras], key=lambda c: c[2] * c[3], reverse=True)

    # -- 4. ojos y ángulo -------------------------------------------------------------
    def _ojos(self, gris, caja):
        x, y, w, h = caja
        zona = gris[y:y + int(h * 0.6), x:x + w]  # mitad superior del rostro
        ojos = self.cascada_ojos.detectMultiScale(zona, scaleFactor=1.1, minNeighbors=8, minSize=(w // 10,) * 2)
        centros = sorted([(x + ex + ew / 2, y + ey + eh / 2) for ex, ey, ew, eh in ojos], key=lambda c: c[0])
        if len(centros) >= 2:
            izq, der = centros[0], centros[-1]
            dist = der[0] - izq[0]
            if 0.25 * w < dist < 0.75 * w and abs(der[1] - izq[1]) < 0.25 * h:  # par plausible
                return [izq, der]
        return []

    def _normalizar(self, rgb):
        if not self.usar_clahe:
            return rgb
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        lab[..., 0] = self.clahe.apply(lab[..., 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def procesar(self, img_rgb, ruta="", persona="") -> ResultadoRostro:
        img = self._redimensionar(img_rgb)
        gris = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        caras = self.detectar(gris)
        r = ResultadoRostro(ruta=str(ruta), persona=persona, ok=False, n_rostros=len(caras))
        if not caras:
            r.motivo = "sin rostro detectado"
            return r
        caja = caras[0]
        r.caja = caja
        ojos = self._ojos(gris, caja)
        x, y, w, h = caja
        centro = (x + w / 2, y + h / 2)
        if ojos:
            (x1, y1), (x2, y2) = ojos
            r.ojos, r.angulo = ojos, float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            centro = ((x1 + x2) / 2, (y1 + y2) / 2 + 0.15 * h)  # centro facial aprox. bajo los ojos
        M = cv2.getRotationMatrix2D(centro, r.angulo, 1.0)
        rot = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        lado = int(max(w, h) * (1 + 2 * self.margen))
        cx, cy = int(centro[0]), int(centro[1])
        x0, y0 = cx - lado // 2, cy - lado // 2
        pad = lado  # relleno para recortes que salen de la imagen
        rot_p = cv2.copyMakeBorder(rot, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        recorte = rot_p[y0 + pad:y0 + pad + lado, x0 + pad:x0 + pad + lado]
        recorte = cv2.resize(recorte, (self.tam, self.tam), interpolation=cv2.INTER_AREA)
        r.rostro, r.ok = self._normalizar(recorte), True
        r.motivo = "ok (alineado)" if ojos else "ok (sin ojos: sin rotación)"
        return r

    def procesar_ruta(self, ruta, persona=""):
        try:
            return self.procesar(leer_imagen(ruta), ruta, persona)
        except Exception as e:  # imagen corrupta o formato no soportado
            return ResultadoRostro(ruta=str(ruta), persona=persona, ok=False, motivo=f"error de lectura: {e}")


def listar_dataset(carpeta) -> list[tuple[str, Path]]:
    """Devuelve [(persona, ruta)] a partir de una carpeta con una subcarpeta por persona."""
    carpeta = Path(carpeta)
    pares = []
    for sub in sorted(p for p in carpeta.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))):
        for f in sorted(sub.rglob("*")):
            if f.suffix.lower() in EXTENSIONES and not f.name.startswith("."):
                pares.append((sub.name, f))
    return pares


def aumentar(rostro: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Aumento de datos ligero: espejo, brillo/contraste, rotación ±10° y desplazamiento."""
    img = rostro.copy()
    if rng.random() < 0.5:
        img = img[:, ::-1]
    alfa, beta = rng.uniform(0.75, 1.25), rng.uniform(-25, 25)
    img = np.clip(img.astype(np.float32) * alfa + beta, 0, 255).astype(np.uint8)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-10, 10), rng.uniform(0.92, 1.08))
    M[:, 2] += rng.uniform(-0.05, 0.05, 2) * w
    return cv2.warpAffine(np.ascontiguousarray(img), M, (w, h), borderMode=cv2.BORDER_REFLECT)
