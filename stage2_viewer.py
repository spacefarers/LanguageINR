from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

# Project imports
from config import device, opt, dtype, TRANSFER_FUNCTION_PATH, VOLUME_DIMS, VOLUME_NAME  # type: ignore
from model import NGP_TCNN, SemanticLayer, SceneAutoencoder  # type: ignore
from render import Camera, render_with_nerfacc, ParaViewTransferFunction  # type: ignore
import stage2  # type: ignore


# ------------------------------
# Constants / Defaults
# ------------------------------

STAGE1_PATH_DEFAULT    = f"./models/stage1_{VOLUME_NAME}.pth"
STAGE2_HEAD_DEFAULT    = f"./models/stage2_{VOLUME_NAME}_semantic_head.pth"
STAGE2_AUTOENCODER_DEFAULT = f"./models/stage2_{VOLUME_NAME}_autoencoder.pth"
TRANSFER_FUNCTION_DEFAULT = TRANSFER_FUNCTION_PATH  # Imported from config.py
CANONICAL_NEGATIVE_PHRASES = ["object", "things", "stuff", "texture"]


# ------------------------------
# Utility helpers
# ------------------------------

def _ensure(cond: bool, msg: str):
    if not cond:
        raise RuntimeError(msg)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _normalize(t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(t, dim=-1, eps=eps)


def _dense_coords_for_inr(D: int, H: int, W: int, dev: torch.device) -> torch.Tensor:
    """
    Generate dense 3D coordinates in [-1, 1]^3 for sampling the INR.

    Args:
        D, H, W: Depth, Height, Width dimensions
        dev: Device to create tensors on

    Returns:
        [D, H, W, 3] tensor with (x, y, z) coordinates
    """
    x = torch.linspace(-1, 1, W, device=dev, dtype=dtype)
    y = torch.linspace(-1, 1, H, device=dev, dtype=dtype)
    z = torch.linspace(-1, 1, D, device=dev, dtype=dtype)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
    return torch.stack([xx, yy, zz], dim=-1)


def relevancy_score(
    features: torch.Tensor,
    query: torch.Tensor,
    canonical_negatives: List[torch.Tensor],
    eps: float = 1e-6,
) -> torch.Tensor:
    query = query.view(1, -1)
    pos_sim = F.cosine_similarity(
        features, query.expand_as(features), dim=-1, eps=eps
    )

    if not canonical_negatives:
        return pos_sim

    # Stack negatives: [K, D]
    neg_stack = torch.stack(canonical_negatives, dim=0)  # [K, D]
    neg_sim = F.cosine_similarity(
        features.unsqueeze(1), neg_stack.unsqueeze(0), dim=-1, eps=eps
    )
    neg_max = neg_sim.max(dim=-1).values  # [N]

    # Contrast: emphasize query over negatives while preserving dynamic range.
    contrasted = (pos_sim - neg_max).clamp(min=0.0)
    if contrasted.max().item() <= 0.0:
        # Fallback: if nothing beats the canonical negatives, return the raw similarity.
        return pos_sim
    return contrasted


# ------------------------------
# Core engine
# ------------------------------

class VolumeSemanticSearcher:
    """
    Loads INR + semantic head + CLIP model, builds a text-conditioned similarity grid
    S(x,y,z), and renders highlight overlays via the project's NerfAcc renderer.

    The semantic head has three hierarchical levels (subpart, part, whole).
    In auto mode, selects the level with the highest maximum activation.
    """

    def __init__(
        self,
        stage1_path: str = STAGE1_PATH_DEFAULT,
        stage2_head_path: str = STAGE2_HEAD_DEFAULT,
        autoencoder_path: str = STAGE2_AUTOENCODER_DEFAULT,
        transfer_fn_path: str = TRANSFER_FUNCTION_DEFAULT,
        default_res_hw: Tuple[int, int] = (512, 512),
        default_samples: int = 16,
        voxel_batch_cap: int = 256 * 256,  # cap per-chunk voxels to a single depth slice by default
    ) -> None:
        self.image_hw = list(default_res_hw)
        self.n_samples = int(default_samples)
        self.voxel_batch_cap = max(1, int(voxel_batch_cap))

        # ---- Load Stage‑1 INR (supports checkpoints with or without 'opt') ----
        _ensure(os.path.exists(stage1_path), f"Missing Stage‑1 model at {stage1_path}")
        ckpt = torch.load(stage1_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model_opt = ckpt.get("opt", opt)
            self.grid_inr = NGP_TCNN(model_opt).to(device)
            self.grid_inr.load_state_dict(ckpt["model_state_dict"])
        else:  # raw state_dict
            self.grid_inr = NGP_TCNN(opt).to(device)
            self.grid_inr.load_state_dict(ckpt)
        self.grid_inr.eval()

        # ---- CLIP model (text/visual encoders share 768‑D space) ----
        self.clip_model, self.clip_preprocess = stage2.load_clip_model()
        self.clip_model.eval()

        # ---- Stage‑2 semantic head (latent) ----
        _ensure(os.path.exists(stage2_head_path), f"Missing Stage‑2 head at {stage2_head_path}")
        head_loaded = torch.load(stage2_head_path, map_location="cpu")

        def _unwrap_state_dict(blob: object) -> Dict[str, torch.Tensor]:
            if not isinstance(blob, dict):
                return blob  # type: ignore[return-value]
            for key in ("model_state_dict", "state_dict", "semantic_head", "module"):
                inner = blob.get(key)
                if isinstance(inner, dict):
                    return inner  # type: ignore[return-value]
            return blob  # type: ignore[return-value]

        head_state = _unwrap_state_dict(head_loaded)
        meta_tensor = head_state.get("_meta") if isinstance(head_state, dict) else None
        if meta_tensor is None and isinstance(head_loaded, dict):
            meta_tensor = head_loaded.get("_meta")
        if meta_tensor is not None:
            self.semantic = SemanticLayer.from_meta(meta_tensor).to(device)
        else:
            # Fallback for checkpoints without embedded metadata.
            self.semantic = SemanticLayer().to(device)
        self.semantic.load_state_dict(head_state)
        self.semantic.eval()
        self.latent_dim = self.semantic.latent_dim

        # ---- Load Autoencoder (if available) ----
        self.autoencoder = None
        if os.path.exists(autoencoder_path):
            self.autoencoder = SceneAutoencoder().to(device)
            self.autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=device))
            self.autoencoder.eval()
        # If autoencoder is not available, we'll work directly with 768-D features

        # ---- Transfer function ----
        self.transfer_fn = ParaViewTransferFunction(transfer_fn_path)

        # ---- Geometry ----
        self._Dv, self._Hv, self._Wv = map(int, self.grid_inr.get_volume_extents())

        # ---- State (built on demand) ----
        self._rgba: Optional[torch.Tensor] = None        # [D,H,W,4] float
        self._S: Optional[torch.Tensor] = None           # [D,H,W] float
        self._argmax_norm: Optional[torch.Tensor] = None # [3] in [-1,1] (x,y,z)
        self._v_norm: Optional[torch.Tensor] = None      # [D,H,W,1] float
        self._S_levels: Dict[str, torch.Tensor] = {}
        self.hierarchy_mode: str = "part"
        self._selected_level: Optional[str] = None
        self._default_use_canonical: bool = False
        self._canonical_embeddings: Optional[List[torch.Tensor]] = None

    def set_canonical_negatives_enabled(self, enabled: bool) -> None:
        """Toggle the use of canonical negatives for contrastive scoring."""
        self._default_use_canonical = bool(enabled)

    def canonical_negatives_enabled(self) -> bool:
        return self._default_use_canonical

    def _project_text_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Map a 768-D CLIP embedding into the latent space expected by the semantic head.
        """
        vec = embedding.to(device=device, dtype=torch.float32)
        if self.autoencoder is not None:
            with torch.no_grad():
                latent = self.autoencoder.encoder(vec.unsqueeze(0)).squeeze(0)
        else:
            latent = vec

        latent = latent.to(device=device, dtype=dtype)
        if latent.shape[-1] != self.latent_dim:
            raise RuntimeError(
                f"Latent dimension mismatch: expected {self.latent_dim}, got {latent.shape[-1]}"
            )
        return latent

    def _get_canonical_embeddings(self) -> List[torch.Tensor]:
        if self._canonical_embeddings is None:
            embeddings: List[torch.Tensor] = []
            for phrase in CANONICAL_NEGATIVE_PHRASES:
                try:
                    clip_z = self.encode_text(phrase).detach()
                    embeddings.append(self._project_text_embedding(clip_z))
                except Exception:
                    # Skip phrases that fail to encode for any reason.
                    continue
            self._canonical_embeddings = embeddings
        return self._canonical_embeddings

    def _ensure_scalar_field(self) -> torch.Tensor:
        """Cache the normalized scalar volume so downstream passes avoid re-sampling the INR."""
        if self._v_norm is not None:
            return self._v_norm

        Dv, Hv, Wv = self._Dv, self._Hv, self._Wv
        coords = _dense_coords_for_inr(Dv, Hv, Wv, device).view(-1, 3)
        v = self.grid_inr(coords).view(Dv, Hv, Wv, 1)
        v_min = v.amin()
        v_max = v.amax()
        v_norm = ((v - v_min) / (v_max - v_min + 1e-8)).clamp(0, 1).contiguous()
        self._v_norm = v_norm
        return self._v_norm

    # --------- Feature encoders ---------

    @torch.no_grad()
    def encode_text(self, phrase: str) -> torch.Tensor:
        """768‑D CLIP text embedding (normalized)."""
        import open_clip
        # Tokenize and encode text
        text = open_clip.tokenize([phrase]).to(device)
        z = self.clip_model.encode_text(text)  # [1, 768]
        z = z.squeeze(0)  # [768]
        return z.to(device=device, dtype=torch.float32)

    @torch.no_grad()
    def _build_rgba(self) -> torch.Tensor:
        """Sample INR once, apply transfer function, cache as [D,H,W,4] on device."""
        if self._rgba is not None:
            return self._rgba

        v_norm = self._ensure_scalar_field()
        Dv, Hv, Wv = v_norm.shape[:3]

        rgb, alpha = self.transfer_fn(v_norm)
        if alpha.dim() == 3:  # ensure channel
            alpha = alpha.unsqueeze(-1)
        rgba = torch.cat([rgb.clamp(0, 1), alpha.clamp(0, 0.999)], dim=-1).contiguous()

        self._rgba = rgba.to(device=device, dtype=torch.float32)
        return self._rgba

    # --------- Similarity grid ---------

    @torch.no_grad()
    def build_similarity_grid(
        self,
        z_text: torch.Tensor,
        aggregation_radius: int = 0,
        *,
        hierarchy_mode: Optional[str] = None,
        use_canonical: Optional[bool] = None,
    ) -> None:
        """Build similarity grids for each hierarchy level and cache the selection."""
        try:
            Dv, Hv, Wv = self._Dv, self._Hv, self._Wv
            mode = (hierarchy_mode or self.hierarchy_mode or "auto").lower()

            # Reset cached similarity to release memory from previous queries.
            self._S = None
            self._S_levels = {}
            self._selected_level = None

            with torch.no_grad():
                z_text = self.autoencoder.encoder(z_text.unsqueeze(0)).squeeze(0)

            x_coords = torch.linspace(-1, 1, Wv, device=device, dtype=torch.float32)
            y_coords = torch.linspace(-1, 1, Hv, device=device, dtype=torch.float32)
            z_coords = torch.linspace(-1, 1, Dv, device=device, dtype=torch.float32)

            voxels_per_slice = Hv * Wv
            max_voxels = max(voxels_per_slice, self.voxel_batch_cap)
            z_per_chunk = max(1, max_voxels // voxels_per_slice)

            pad = max(0, int(aggregation_radius))
            kernel = 2 * pad + 1

            S_levels = {
                "s": torch.empty((Dv, Hv, Wv), device=device, dtype=torch.float32),
                "p": torch.empty((Dv, Hv, Wv), device=device, dtype=torch.float32),
                "w": torch.empty((Dv, Hv, Wv), device=device, dtype=torch.float32),
            }

            yy_base, xx_base = torch.meshgrid(y_coords, x_coords, indexing="ij")
            yy_base = yy_base.unsqueeze(0)
            xx_base = xx_base.unsqueeze(0)

            d_feat = self.latent_dim
            use_canon = self._default_use_canonical if use_canonical is None else bool(use_canonical)
            canonical_vectors = self._get_canonical_embeddings() if use_canon else []

            for start in range(0, Dv, z_per_chunk):
                end = min(start + z_per_chunk, Dv)
                depth = end - start

                z_chunk = z_coords[start:end].view(-1, 1, 1).expand(-1, Hv, Wv)
                yy = yy_base.expand(depth, -1, -1)
                xx = xx_base.expand(depth, -1, -1)
                coords = torch.stack([xx, yy, z_chunk], dim=-1).reshape(-1, 3)

                feat_s, feat_p, feat_w = self.semantic(coords)
                for key, feats in (("s", feat_s), ("p", feat_p), ("w", feat_w)):
                    if feats is None:
                        continue
                    flat = feats.to(dtype=dtype).view(-1, d_feat)

                    # with torch.no_grad():
                    #     flat = self.autoencoder.decoder(flat)
                    scores_flat = relevancy_score(flat, z_text, canonical_vectors)
                    S_levels[key][start:end] = scores_flat.view(depth, Hv, Wv)

            if pad > 0:
                weight = torch.ones((1, 1, kernel, kernel, kernel), device=device, dtype=torch.float32)
                ones_volume = torch.ones((1, 1, Dv, Hv, Wv), device=device, dtype=torch.float32)
                norm = F.conv3d(ones_volume, weight, padding=pad)
                norm = norm.clamp_min(1.0)
                for key in S_levels.keys():
                    vol = S_levels[key].unsqueeze(0).unsqueeze(0)
                    smoothed = F.conv3d(vol, weight, padding=pad) / norm
                    S_levels[key] = smoothed.squeeze(0).squeeze(0).contiguous()
                del weight, ones_volume, norm

            self._S_levels = {k: v.contiguous() for k, v in S_levels.items()}
            self.hierarchy_mode = mode
            self._S, selected = self._resolve_hierarchy_map(mode)
            self._selected_level = selected

            flat_idx = int(torch.argmax(self._S).item())
            d = flat_idx // (Hv * Wv)
            h = (flat_idx % (Hv * Wv)) // Wv
            w = flat_idx % Wv
            self._argmax_norm = torch.stack([x_coords[w], y_coords[h], z_coords[d]])
        except Exception as e:
            print(f"[viewer] Error building similarity grid: {e}")
            raise

    def _resolve_hierarchy_map(self, mode: str) -> Tuple[torch.Tensor, str]:
        """
        Resolve which hierarchy level to use based on the mode.

        In 'auto' mode, selects the level with the highest maximum activation,
        matching the logic from activate_stream in eval/evaluate_iou_loc.py:
        - For each semantic level (s, p, w), find the highest response (max value)
        - Select the level that had the highest maximum score
        """
        if not self._S_levels:
            raise RuntimeError("Similarity grid has not been built yet.")
        mode_l = (mode or "auto").lower()
        mapping = {
            "subpart": "s",
            "s": "s",
            "part": "p",
            "p": "p",
            "whole": "w",
            "w": "w",
        }
        if mode_l in mapping:
            key = mapping[mode_l]
            return self._S_levels[key], key
        if mode_l == "max":
            combined = torch.maximum(
                torch.maximum(self._S_levels["s"], self._S_levels["p"]),
                self._S_levels["w"],
            )
            return combined.contiguous(), "max"
        if mode_l == "sum":
            combined = (self._S_levels["s"] + self._S_levels["p"] + self._S_levels["w"]) / 3.0
            return combined.contiguous(), "sum"
        if mode_l == "auto":
            # Find the maximum activation value within each of the three relevance maps
            score_lvl = torch.zeros((3,), device=device)
            level_keys = ['s', 'p', 'w']

            for i, key in enumerate(level_keys):
                volume = self._S_levels[key]
                if volume.numel() == 0:
                    score_lvl[i] = float("-inf")
                else:
                    # Find the highest response (max value) in the relevance map for that level
                    score_lvl[i] = volume.max()

            # Select the level that had the highest maximum score
            chosen_idx = torch.argmax(score_lvl).item()
            best_key = level_keys[chosen_idx]

            # Print which mask head was chosen
            level_names = {'s': 'subpart', 'p': 'part', 'w': 'whole'}
            print(f"[viewer] Selected mask head: {level_names[best_key]} (max activation: {score_lvl[chosen_idx]:.4f})")
            print(f"  Activation scores - subpart: {score_lvl[0]:.4f}, part: {score_lvl[1]:.4f}, whole: {score_lvl[2]:.4f}")

            return self._S_levels[best_key], best_key
        raise ValueError(f"Unknown hierarchy mode '{mode}'.")

    def set_hierarchy_mode(self, mode: str) -> None:
        """Set the hierarchy mode and re-resolve the similarity map if already computed."""
        self.hierarchy_mode = mode
        if self._S_levels:
            self._S, selected = self._resolve_hierarchy_map(mode)
            self._selected_level = selected
            if mode.lower() not in ["auto"]:  # auto mode already prints
                level_names = {'s': 'subpart', 'p': 'part', 'w': 'whole'}
                display_name = level_names.get(selected, selected)
                print(f"[viewer] Hierarchy mode changed to '{mode}', using {display_name} level")

    def save_similarity_volume(self, path: str, level: Optional[str] = None) -> None:
        if not self._S_levels and self._S is None:
            raise RuntimeError("Similarity grid not computed; call build_similarity_grid first.")
        if level is None:
            sim = self._S
        else:
            sim, _ = self._resolve_hierarchy_map(level)
        np.savez_compressed(path, similarity=sim.detach().cpu().numpy())

    def get_similarity_levels(self) -> Dict[str, torch.Tensor]:
        if not self._S_levels:
            raise RuntimeError("Similarity grid not computed; call build_similarity_grid first.")
        return {k: v.detach().cpu() for k, v in self._S_levels.items()}

    # --------- Rendering ---------

    @torch.no_grad()
    def render_highlight(
        self,
        cam: Camera,
        threshold: float = 0.90,
        blob_radius_vox: float = 0.0,
        bg_value: float = 1.0,
        image_hw: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Apply thresholded mask (optionally modulated by a Gaussian blob around the
        global maximum) and render with the project's NerfAcc volume renderer.
        """
        _ensure(self._S is not None, "You must build the similarity grid first (call build_similarity_grid).")
        if image_hw is None:
            image_hw = tuple(self.image_hw)
        rgba = self._build_rgba()  # [D,H,W,4]
        Dv, Hv, Wv = rgba.shape[:3]
        S = self._S.view(Dv, Hv, Wv)

        s_min = float(S.min().item())
        s_max = float(S.max().item())
        denom = max(s_max - s_min, 1e-6)
        M = ((S - s_min) / denom) > float(_clamp(threshold, 0.0, 1.0))  # [D,H,W] bool
        M = M.to(dtype=rgba.dtype).unsqueeze(-1)  # [D,H,W,1]

        # Optional Gaussian blob centered at global max
        if blob_radius_vox > 0 and self._argmax_norm is not None:
            cx = (self._argmax_norm[0].item() + 1) * 0.5 * (Wv - 1)
            cy = (self._argmax_norm[1].item() + 1) * 0.5 * (Hv - 1)
            cz = (self._argmax_norm[2].item() + 1) * 0.5 * (Dv - 1)
            x_blob = torch.arange(Wv, device=device, dtype=rgba.dtype)
            y_blob = torch.arange(Hv, device=device, dtype=rgba.dtype)
            z_blob = torch.arange(Dv, device=device, dtype=rgba.dtype)
            zz, yy, xx = torch.meshgrid(z_blob, y_blob, x_blob, indexing="ij")
            d2 = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
            blob = torch.exp(-d2 / (2.0 * (blob_radius_vox ** 2 + 1e-12))).unsqueeze(-1)
            M = M * blob

        # Modulate alpha, fade suppressed voxels toward a white background
        rgba_masked = rgba.clone()
        rgba_masked[..., 3:4] = rgba[..., 3:4] * M
        rgba_masked[..., :3] = rgba[..., :3] * M + bg_value * (1.0 - M)

        img = render_with_nerfacc(
            rgba_volume=rgba_masked.to(dtype=dtype),
            camera=cam,
            hw=image_hw,
            spp=None,
            batch_size=8192,
        )
        return (img.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)

    @torch.no_grad()
    def render_base(self, cam: Camera, image_hw: Optional[Tuple[int, int]] = None) -> np.ndarray:
        if image_hw is None:
            image_hw = tuple(self.image_hw)
        rgba = self._build_rgba()
        img = render_with_nerfacc(
            rgba_volume=rgba.to(dtype=dtype),
            camera=cam,
            hw=image_hw,
            spp=None,
            batch_size=8192,
        )
        return (img.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)


# ------------------------------
# Minimal Qt GUI (optional)
# ------------------------------

@dataclass
class Orbit:
    azi_deg: float
    polar_deg: float
    dist: float
    center: Tuple[float, float, float]


def _default_orbit(searcher: VolumeSemanticSearcher) -> Orbit:
    D, H, W = searcher._Dv, searcher._Hv, searcher._Wv
    dist = math.sqrt(D * D + H * H + W * W) * 0.75
    center = (W / 2.0, H / 2.0, D / 2.0)
    return Orbit(azi_deg=20.0, polar_deg=80.0, dist=dist, center=center)


def _build_camera(orbit: Orbit) -> Camera:
    return Camera(azi_deg=orbit.azi_deg, polar_deg=orbit.polar_deg, dist=orbit.dist, center=orbit.center)


def _try_launch_gui(args, searcher: VolumeSemanticSearcher) -> int:
    try:
        from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore
    except Exception as e:  # pragma: no cover - optional path
        print("[viewer] PyQt5 not available, falling back to CLI:", e)
        return 1

    class ImageWidget(QtWidgets.QLabel):
        requestRender = QtCore.pyqtSignal()

        def __init__(self, parent=None, *, orbit_nudge=None, zoom_nudge=None, drag_state=None):
            super().__init__(parent)
            self.setAlignment(QtCore.Qt.AlignCenter)
            self.setMinimumSize(256, 256)
            self._dragging = False
            self._last = None
            self._slow = False
            self._current_qimage = None
            # Callbacks keep camera controls available even if Qt re-parents us.
            self._orbit_cb = orbit_nudge
            self._zoom_cb = zoom_nudge
            self._drag_cb = drag_state

        def mousePressEvent(self, e):
            if e.button() in (QtCore.Qt.LeftButton, QtCore.Qt.RightButton):
                self._dragging = True
                self._slow = (e.button() == QtCore.Qt.RightButton)
                self._last = e.pos()
                if self._drag_cb is not None:
                    self._drag_cb(True)
                e.accept()
            else:
                super().mousePressEvent(e)

        def mouseMoveEvent(self, e):
            if self._dragging and self._last is not None:
                dx = e.x() - self._last.x()
                dy = e.y() - self._last.y()
                factor = 0.3 if not self._slow else 0.08
                if self._orbit_cb is not None:
                    self._orbit_cb(dx * factor, dy * factor)
                else:
                    owner = self.window()
                    if hasattr(owner, "_nudge_orbit"):
                        owner._nudge_orbit(dx * factor, dy * factor)
                self._last = e.pos()
                self.requestRender.emit()
                e.accept()
            else:
                super().mouseMoveEvent(e)

        def mouseReleaseEvent(self, e):
            self._dragging = False
            self._last = None
            if self._drag_cb is not None:
                self._drag_cb(False)
            self.requestRender.emit()
            e.accept()

        def wheelEvent(self, e):
            delta = e.angleDelta().y() / 120.0
            if self._zoom_cb is not None:
                self._zoom_cb(-delta * 0.1)
            else:
                owner = self.window()
                if hasattr(owner, "_nudge_zoom"):
                    owner._nudge_zoom(-delta * 0.1)
            self.requestRender.emit()
            e.accept()

        def update_image(self, qimg: "QtGui.QImage") -> None:
            self._current_qimage = qimg.copy()
            self._apply_pixmap()

        def resizeEvent(self, event):  # noqa: N802 - Qt override
            super().resizeEvent(event)
            if self._current_qimage is not None:
                self._apply_pixmap()

        def _apply_pixmap(self) -> None:
            if self._current_qimage is None:
                return
            pixmap = QtGui.QPixmap.fromImage(self._current_qimage)
            target_size = self.size()
            if target_size.width() > 0 and target_size.height() > 0:
                pixmap = pixmap.scaled(target_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.setPixmap(pixmap)

    class Panel(QtWidgets.QWidget):
        paramsChanged = QtCore.pyqtSignal()
        findRequested = QtCore.pyqtSignal(str)
        def __init__(self):
            super().__init__()
            self.text = QtWidgets.QLineEdit(self)
            self.text.setPlaceholderText("Type a phrase, e.g., 'thin branches'")
            self.find = QtWidgets.QPushButton("Find region", self)

            self.th_label = QtWidgets.QLabel("Visibility threshold")
            self.th_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
            self.th_slider.setRange(0, 100)
            self.th_slider.setValue(int(args.threshold * 100))
            self.th_value = QtWidgets.QLabel(f"{args.threshold:.2f}")

            self.agg_label = QtWidgets.QLabel("Aggregation radius")
            self.agg_spin = QtWidgets.QSpinBox(self)
            self.agg_spin.setRange(0, 10)
            self.agg_spin.setValue(args.agg)

            self.blob_label = QtWidgets.QLabel("Blob radius (voxels)")
            self.blob_spin = QtWidgets.QDoubleSpinBox(self)
            self.blob_spin.setRange(0.0, 64.0)
            self.blob_spin.setSingleStep(1.0)
            self.blob_spin.setValue(args.blob)

            self.res_label = QtWidgets.QLabel("Resolution (px)")
            self.res_spin = QtWidgets.QSpinBox(self)
            self.res_spin.setRange(128, 1024)
            self.res_spin.setSingleStep(32)
            self.res_spin.setValue(args.res)

            self.reset = QtWidgets.QPushButton("Reset view", self)

            form = QtWidgets.QFormLayout()
            form.addRow("Phrase:", self.text)
            form.addRow(self.find)
            form.addRow(self.th_label, self.th_slider); form.addRow(" ", self.th_value)
            form.addRow(self.agg_label, self.agg_spin)
            form.addRow(self.blob_label, self.blob_spin)
            form.addRow(self.res_label, self.res_spin)
            form.addRow(self.reset)
            self.setLayout(form)

            self.th_slider.valueChanged.connect(lambda v: (self.th_value.setText(f"{v/100.0:.2f}"), self.paramsChanged.emit()))
            self.agg_spin.valueChanged.connect(lambda _: self.paramsChanged.emit())
            self.blob_spin.valueChanged.connect(lambda _: self.paramsChanged.emit())
            self.res_spin.valueChanged.connect(lambda _: self.paramsChanged.emit())
            self.find.clicked.connect(lambda: self.findRequested.emit(self.text.text().strip()))

        def threshold(self) -> float: return self.th_slider.value() / 100.0
        def agg(self) -> int: return int(self.agg_spin.value())
        def blob(self) -> float: return float(self.blob_spin.value())
        def res(self) -> int: return int(self.res_spin.value())

    class SimilarityWorker(QtCore.QObject):
        finished = QtCore.pyqtSignal(float)
        failed = QtCore.pyqtSignal(str)

        def __init__(self, searcher: VolumeSemanticSearcher, phrase: str, agg: int):
            super().__init__()
            self._searcher = searcher
            self._phrase = phrase
            self._agg = agg

        @QtCore.pyqtSlot()
        def run(self):
            try:
                t0 = time.time()
                z_text = self._searcher.encode_text(self._phrase)
                self._searcher.build_similarity_grid(z_text, aggregation_radius=int(self._agg))
                self.finished.emit(time.time() - t0)
            except Exception as exc:  # pragma: no cover - GUI-only path
                self.failed.emit(str(exc))

    class Main(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Stage‑2 Semantic Viewer")
            self.searcher = searcher
            self.orbit = _default_orbit(self.searcher)
            self.cam = _build_camera(self.orbit)

            self.image = ImageWidget(
                self,
                orbit_nudge=self._nudge_orbit,
                zoom_nudge=self._nudge_zoom,
                drag_state=self._set_drag_state,
            )
            self.panel = Panel()
            self.image.setStyleSheet("background-color: #222;")

            central = QtWidgets.QWidget(self)
            layout = QtWidgets.QHBoxLayout(central)
            layout.addWidget(self.image, 1)
            layout.addWidget(self.panel, 0)
            self.setCentralWidget(central)
            self.resize(1100, 700)

            # wire signals
            self.image.requestRender.connect(self.render_once)
            self.panel.paramsChanged.connect(self.render_once)
            self.panel.findRequested.connect(self._on_find)
            self.panel.reset.clicked.connect(self._on_reset)

            self._phrase_ready = False
            self._drag_active = False
            self._worker_thread: Optional[QtCore.QThread] = None
            self._worker: Optional[SimilarityWorker] = None
            self._busy_dialog: Optional[QtWidgets.QProgressDialog] = None
            self._pending_phrase: Optional[str] = None
            self.render_once()

        def _on_reset(self):
            self.orbit = _default_orbit(self.searcher)
            self.cam = _build_camera(self.orbit)
            self.render_once()

        def _nudge_orbit(self, dx_deg: float, dy_deg: float):
            self.orbit.azi_deg = (self.orbit.azi_deg - dx_deg) % 360.0
            self.orbit.polar_deg = _clamp(self.orbit.polar_deg - dy_deg, 5.0, 175.0)
            self.cam = _build_camera(self.orbit)

        def _nudge_zoom(self, delta: float):
            self.orbit.dist = _clamp(self.orbit.dist * math.exp(delta), 50.0, 5000.0)
            self.cam = _build_camera(self.orbit)

        def _set_drag_state(self, active: bool):
            if self._drag_active == active:
                return
            self._drag_active = active

        def _show_busy(self, phrase: str) -> None:
            if self._busy_dialog is not None:
                return
            dlg = QtWidgets.QProgressDialog(f"Finding '{phrase}'...", "", 0, 0, self)
            dlg.setWindowTitle("Building similarity grid")
            dlg.setCancelButton(None)
            dlg.setWindowModality(QtCore.Qt.WindowModal)
            dlg.setMinimumDuration(0)
            dlg.setAutoClose(False)
            dlg.setAutoReset(False)
            dlg.show()
            self._busy_dialog = dlg

        def _hide_busy(self) -> None:
            if self._busy_dialog is None:
                return
            self._busy_dialog.hide()
            self._busy_dialog.deleteLater()
            self._busy_dialog = None

        def _cleanup_worker(self) -> None:
            if self._worker_thread is not None:
                if self._worker_thread.isRunning():
                    self._worker_thread.quit()
                    self._worker_thread.wait()
                self._worker_thread.deleteLater()
                self._worker_thread = None
            if self._worker is not None:
                self._worker.deleteLater()
                self._worker = None

        def _on_find(self, phrase: str):
            if not phrase:
                return
            if self._worker_thread is not None:
                return  # Ignore while a job is already running
            try:
                self._phrase_ready = False
                self._pending_phrase = phrase
                self.panel.find.setEnabled(False)
                self._show_busy(phrase)

                self._worker_thread = QtCore.QThread(self)
                self._worker = SimilarityWorker(self.searcher, phrase, self.panel.agg())
                self._worker.moveToThread(self._worker_thread)
                self._worker_thread.started.connect(self._worker.run)
                self._worker.finished.connect(self._on_worker_finished)
                self._worker.failed.connect(self._on_worker_failed)
                self._worker.finished.connect(self._worker_thread.quit)
                self._worker.failed.connect(self._worker_thread.quit)
                self._worker_thread.start()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", str(e))

        def _on_worker_finished(self, elapsed: float) -> None:
            self._phrase_ready = True
            self._hide_busy()
            self.panel.find.setEnabled(True)
            if self._pending_phrase:
                print(f"[viewer] Similarity grid built for '{self._pending_phrase}' in {elapsed:.2f}s")
            else:
                print(f"[viewer] Similarity grid built in {elapsed:.2f}s")
            self._pending_phrase = None
            self.render_once()
            self._cleanup_worker()

        def _on_worker_failed(self, message: str) -> None:
            self._hide_busy()
            self.panel.find.setEnabled(True)
            QtWidgets.QMessageBox.critical(self, "Error", message)
            self._pending_phrase = None
            self._cleanup_worker()

        def closeEvent(self, event):  # noqa: N802 - Qt override
            self._hide_busy()
            self._cleanup_worker()
            super().closeEvent(event)

        def render_once(self):
            res = self.panel.res()
            if self._drag_active:
                # Lower interactive resolution during orbiting to keep redraws responsive.
                res = max(128, res // 2)

            self.searcher.image_hw = [res, res]
            try:
                if self._phrase_ready:
                    img = self.searcher.render_highlight(
                        cam=self.cam,
                        threshold=self.panel.threshold(),
                        blob_radius_vox=self.panel.blob(),
                        image_hw=(res, res),
                    )
                else:
                    img = self.searcher.render_base(self.cam, image_hw=(res, res))
                qimg = QtGui.QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1], QtGui.QImage.Format_RGB888)
                self.image.update_image(qimg)
            except Exception as e:
                import traceback
                print("[viewer] Render error:")
                traceback.print_exc()

    app = QtWidgets.QApplication(sys.argv)
    win = Main()
    win.show()
    return app.exec_()


# ------------------------------
# CLI
# ------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Stage‑2 Semantic Viewer (from scratch)")
    p.add_argument("--stage1", default=STAGE1_PATH_DEFAULT, help="Path to Stage‑1 INR checkpoint")
    p.add_argument("--head", default=STAGE2_HEAD_DEFAULT, help="Path to Stage‑2 semantic head")
    p.add_argument("--ae", default=STAGE2_AUTOENCODER_DEFAULT, help="Path to Stage‑2 autoencoder")
    p.add_argument("--tf", default=TRANSFER_FUNCTION_DEFAULT, help="ParaView transfer function JSON")
    p.add_argument("--res", type=int, default=512, help="Render resolution (square)")
    p.add_argument(
        "--threshold",
        type=float,
        default=0.30,
        help="Similarity threshold tau applied to raw scores (typ. 0.25-0.40)",
    )
    p.add_argument("--agg", type=int, default=0, help="Local feature aggregation radius (voxels)")
    p.add_argument("--blob", type=float, default=0.0, help="Optional Gaussian blob radius around argmax (voxels)")
    p.add_argument("--no-canon", action="store_true", help="Disable canonical negative contrast when scoring")
    p.add_argument("--phrase", type=str, default="", help="Text phrase to search for (CLI mode)")
    p.add_argument("--save", type=str, default="viewer_out.png", help="Path to save rendered image (CLI mode)")
    p.add_argument("--cli", action="store_true", help="Force CLI mode (skip GUI)")
    p.add_argument("--hierarchy", type=str, default="part", choices=["auto", "max", "subpart", "part", "whole", "sum"], help="Hierarchy mode for similarity selection")
    p.add_argument("--save-sim", type=str, default=None, help="Optional path to save the similarity volume (.npz)")
    p.add_argument("--save-sim-level", type=str, default="selected", choices=["selected", "subpart", "part", "whole", "max", "sum"], help="Hierarchy level to save when using --save-sim")
    p.add_argument("--print-level-stats", action="store_true", help="Print per-level summary statistics after building the similarity grid")
    args = p.parse_args(argv)

    searcher = VolumeSemanticSearcher(
        stage1_path=args.stage1,
        stage2_head_path=args.head,
        autoencoder_path=args.ae,
        transfer_fn_path=args.tf,
        default_res_hw=(args.res, args.res),
    )
    searcher.set_canonical_negatives_enabled(not args.no_canon)
    searcher.set_hierarchy_mode(args.hierarchy)

    # Try GUI by default unless --cli is specified
    if not args.cli:
        result = _try_launch_gui(args, searcher)
        if result == 0:  # GUI launched successfully
            return result
        # If GUI failed (returns 1), fall through to CLI mode
        print("[viewer] Falling back to CLI mode...")

    # CLI mode: render one image
    orbit = _default_orbit(searcher)
    cam = _build_camera(orbit)

    if args.phrase:
        z_text = searcher.encode_text(args.phrase)
        searcher.build_similarity_grid(
            z_text,
            aggregation_radius=args.agg,
            hierarchy_mode=args.hierarchy,
            use_canonical=not args.no_canon,
        )
        selected = searcher._selected_level or args.hierarchy
        print(f"[viewer] Selected hierarchy: {selected}")
        if args.print_level_stats:
            levels = searcher.get_similarity_levels()
            for key, vol in levels.items():
                flat = vol.view(-1)
                mean = float(flat.mean().item())
                q95 = float(torch.quantile(flat, 0.95).item()) if flat.numel() > 0 else float('nan')
                mx = float(flat.max().item()) if flat.numel() > 0 else float('nan')
                label = {"s": "subpart", "p": "part", "w": "whole"}.get(key, key)
                print(f"  [{label}] mean={mean:.4f} q95={q95:.4f} max={mx:.4f}")
        if args.save_sim:
            level_to_save = None if args.save_sim_level == "selected" else args.save_sim_level
            os.makedirs(os.path.dirname(args.save_sim) or ".", exist_ok=True)
            searcher.save_similarity_volume(args.save_sim, level=level_to_save)
            print(f"[viewer] Saved similarity volume to {args.save_sim}")
        img = searcher.render_highlight(cam, threshold=args.threshold, blob_radius_vox=args.blob, image_hw=(args.res, args.res))
    else:
        img = searcher.render_base(cam, image_hw=(args.res, args.res))

    import imageio
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    imageio.imwrite(args.save, img)
    print(f"[viewer] Saved {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
