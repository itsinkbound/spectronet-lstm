"""SHAP + LIME explainability, reproducing Section 3.4/4.2 of the paper:

- SHAP: global per-feature importance over the spectrogram, via
  `shap.GradientExplainer` on the trained fusion model.
- LIME: local, per-instance explanations via `lime_image`, perturbing
  super-pixels of a single spectrogram to see which regions drive the
  predicted class.

Both produce heatmaps overlaid on the original spectrogram so a clinician
can compare "what the model looked at" against known diagnostic cues.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from spectronet.config import CLASSES, ExplainConfig


@dataclasses.dataclass
class ExplanationResult:
    class_name: str
    lime_mask: Optional[np.ndarray]
    shap_values: Optional[np.ndarray]


def explain_with_lime(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    image: np.ndarray,
    cfg: ExplainConfig,
    class_index: int,
) -> np.ndarray:
    """Local Interpretable Model-agnostic Explanation for a single spectrogram.

    `predict_fn` must accept a batch of images (N, H, W, 3) uint8/float and
    return class probabilities (N, num_classes) -- i.e. `model.predict`.
    """
    from lime import lime_image
    from skimage.segmentation import mark_boundaries

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image.astype("double"),
        predict_fn,
        top_labels=len(CLASSES),
        hide_color=0,
        num_samples=cfg.n_lime_samples,
    )
    _, mask = explanation.get_image_and_mask(
        class_index, positive_only=True, num_features=10, hide_rest=False
    )
    return mask


def explain_with_shap(
    model, background_images: np.ndarray, images_to_explain: np.ndarray, cfg: ExplainConfig
) -> np.ndarray:
    """Global SHAP feature-importance values for a batch of spectrograms."""
    import shap

    background = background_images[: cfg.n_shap_background]
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(images_to_explain)
    return np.array(shap_values)


def save_explanation_grid(
    images: dict[str, np.ndarray], output_dir: str, filename: str
) -> str:
    """Persists a matplotlib grid of (original, LIME, SHAP) per class, as in
    Fig. 30/31 of the paper."""
    import matplotlib.pyplot as plt

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    n = len(images)
    fig, axes = plt.subplots(n, 1, figsize=(6, 3 * n))
    if n == 1:
        axes = [axes]
    for ax, (title, img) in zip(axes, images.items()):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    full_path = out_path / filename
    fig.savefig(full_path, dpi=150)
    plt.close(fig)
    return str(full_path)
