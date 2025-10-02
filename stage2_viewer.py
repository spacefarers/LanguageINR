# stage2_viewer.py
#
# Interactive Stage-2 visualizer:
# - Loads your Stage-1 grid INR (NGP_TCNN) and Stage-2 semantic layer head
# - Lets the user type a phrase; we compute OpenCLIP text features
# - Builds a coarse 3D similarity grid S(x,y,z) = cos(semantic(x), text)
# - Renders views with the same differentiable ray marcher used in Stage-2
# - Highlights the best-matching region consistently across views
#
# Controls (top-right panel):
# - Text box + "Find region": compute similarity grid for the phrase
# - Blob radius: optionally restrict highlight to a Gaussian blob around the global max
# - Visibility threshold: how strict the mask is when hiding non-matching regions
# - Resolution and Samples: render quality
# - Reset view: center camera
#
# Drag on the image to orbit. Mouse wheel zooms. Right-click drag to slow orbit.
#
# Defaults assume the same opts and paths as main.py/stage2.py.
#
# Notes:
# - No changes to your existing files are required.
# - We reuse stage2.differentiable_render_from_inr and render.Camera.
# - OpenCLIP is frozen and used only for the text encoder.
#
# Copyright (c) 2025

import os
import sys
import math
import time
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from PyQt5 import QtCore, QtGui, QtWidgets

# Project imports
from config import device
from model import NGP_TCNN
from render import Camera, render_with_nerfacc
import stage2
from dataio import get_volume_info
from config import opt

# --------- Defaults that match your training scripts --------- #
# Mirrors main.py's volume shape and opt
_volume_info = get_volume_info()
if not _volume_info["exists"]:
    raise FileNotFoundError(
        f"Configured volume file not found: {_volume_info['path']}"
    )
_x, _y, _z = _volume_info["dims"]
D, H, W = int(_z), int(_y), int(_x)
STAGE1_PATH = "./models/stage1_ngp_tcnn.pth"
STAGE2_HEAD_PATH = "./models/stage2_semantic_head.pth"
CLIP_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
TRANSFER_FUNCTION_PATH = "./paraview_tf/bonsai.json"

# ------------------------------------------------------------- #

def _ensure(cond, msg):
    if not cond:
        raise RuntimeError(msg)

def _to_qimage(img_np_uint8_hw3):
    """Convert HxWx3 uint8 RGB numpy array to QImage."""
    h, w, c = img_np_uint8_hw3.shape
    _ensure(c == 3, "Expected RGB image")
    qimg = QtGui.QImage(img_np_uint8_hw3.data, w, h, 3*w, QtGui.QImage.Format_RGB888)
    return qimg

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

@dataclass
class ViewerState:
    azi_deg: float = 20.0
    polar_deg: float = 80.0
    dist: float = math.sqrt(D*D + H*H + W*W) * 1.4
    center: tuple = (W/2, H/2, D/2)

class VolumeSemanticSearcher:
    """
    Loads the grid INR and semantic head, computes text-driven 3D similarity grids,
    and renders highlight overlays into new views.
    """
    def __init__(self,
                 opt: dict,
                 stage1_path: str,
                 stage2_head_path: str,
                 default_image_hw=(256, 256),
                 default_samples: int = 48,
                 transfer_function_path: str = TRANSFER_FUNCTION_PATH,
                 similarity_voxel_batch: int = 262144):
        self.opt = opt
        self.image_hw = list(default_image_hw)
        self.n_samples = default_samples
        # similarity_voxel_batch controls how many voxels are processed per chunk
        # when building the text-driven similarity volume so we can stay within VRAM.
        self.similarity_voxel_batch = max(1, int(similarity_voxel_batch))

        # Load Stage-1 INR
        self.grid_inr = NGP_TCNN(opt).to(device)
        _ensure(os.path.exists(stage1_path),
                f"Missing Stage-1 model at {stage1_path}")
        self.grid_inr.load_state_dict(torch.load(stage1_path, map_location=device))
        self.grid_inr.eval()

        # Load OpenCLIP just to discover embed dim and get text encoder
        try:
            import open_clip
        except Exception as e:
            raise RuntimeError("Please install open-clip-torch: pip install open-clip-torch") from e

        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            CLIP_NAME, pretrained=CLIP_PRETRAINED
        )
        self.clip_model = self.clip_model.to(device).eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False

        # Embed dim from model
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=device)
            z = self.clip_model.encode_image(dummy)
        self.embed_dim = int(z.shape[-1])

        # Build semantic head and load weights
        self.semantic = stage2.SemanticLayer(embed_dim=self.embed_dim, hidden_dim=128, n_hidden=2).to(device)
        _ensure(os.path.exists(stage2_head_path),
                f"Missing Stage-2 head at {stage2_head_path}. Train Stage-2 first.")
        self.semantic.load_state_dict(torch.load(stage2_head_path, map_location=device))
        self.semantic.eval()

        self.transfer_fn = stage2.ParaViewTransferFunction(transfer_function_path)

        # State for text-driven search
        self.S_vol = None           # [1,1,D,H,W] torch float
        self.argmax_norm = None     # [3] tensor in [-1,1] order (x,y,z)

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        import open_clip
        tokens = open_clip.tokenize([text]).to(device)
        z_t = self.clip_model.encode_text(tokens)
        return F.normalize(z_t, dim=-1).squeeze(0)  # [D]

    @torch.no_grad()
    def build_similarity_grid(self, z_text: torch.Tensor):
        """
        Compute S(x,y,z) = cos( semantic([x,y,z,v_norm]), z_text ) over the full
        volume resolution in VRAM-friendly chunks. Stores self.S_vol [1,1,D,H,W]
        on device and caches the argmax for optional blob highlighting.
        """
        Dv, Hv, Wv = self.opt["full_shape"]
        xs = torch.linspace(-1, 1, Wv, device=device)
        ys = torch.linspace(-1, 1, Hv, device=device)
        zs = torch.linspace(-1, 1, Dv, device=device)

        yy_base, xx_base = torch.meshgrid(ys, xs, indexing='ij')
        yy_base = yy_base.unsqueeze(0)
        xx_base = xx_base.unsqueeze(0)

        S_full = torch.empty((Dv, Hv, Wv), device=device, dtype=torch.float32)

        v_min = self.grid_inr.min()
        v_max = self.grid_inr.max()
        denom = (v_max - v_min).clamp_min(1e-8)

        z_text = z_text.to(device=device, dtype=torch.float32)
        z_text = F.normalize(z_text.unsqueeze(0), dim=-1).squeeze(0)

        voxels_per_slice = Hv * Wv
        max_voxels = max(voxels_per_slice, self.similarity_voxel_batch)
        slice_chunk = max(1, max_voxels // voxels_per_slice)

        for start in range(0, Dv, slice_chunk):
            end = min(start + slice_chunk, Dv)
            z_chunk = zs[start:end]
            zz = z_chunk.view(-1, 1, 1).expand(-1, Hv, Wv)
            yy = yy_base.expand(end - start, -1, -1)
            xx = xx_base.expand(end - start, -1, -1)

            coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

            v = self.grid_inr(coords)
            v_norm = ((v - v_min) / denom).clamp(0, 1)

            feats = self.semantic.forward_per_sample(coords, v_norm)
            feats = F.normalize(feats, dim=-1)

            s = (feats @ z_text.view(-1, 1)).squeeze(1)
            S_full[start:end] = s.view(end - start, Hv, Wv)

        self.S_vol = S_full.unsqueeze(0).unsqueeze(0).contiguous()

        # Cache argmax for optional blob highlighting
        flat = S_full.flatten()

        idx = torch.argmax(flat).item()
        d = idx // (Hv * Wv)
        h = (idx % (Hv * Wv)) // Wv
        w = idx % Wv
        self.argmax_norm = torch.stack([xs[w], ys[h], zs[d]])
        print(f"Similarity grid built. Global max at (x,y,z) = {self.argmax_norm.cpu().numpy()}")

    @torch.no_grad()
    def render_with_highlight(self,
                              cam: Camera,
                              overlay_alpha: float = 0.6,
                              blob_radius_vox: float = 0.0,
                              image_hw=None,
                              n_samples=None):
        """
        Apply highlight by masking the volume's alpha before rendering, then call
        the shared nerfacc renderer: render.render_with_nerfacc.
        """
        _ensure(self.S_vol is not None, "Run 'Find region' first to build the similarity grid.")
        if image_hw is None:
            image_hw = tuple(self.image_hw)
        if n_samples is None:
            n_samples = self.n_samples

        # 1) Build base RGBA volume from INR + TF
        rgba = self.build_rgba_volume()                     # [D,H,W,4]
        Dv, Hv, Wv = rgba.shape[:3]

        # 2) Work with the full-resolution similarity volume (already D x H x W)
        S_full = self.S_vol.view(Dv, Hv, Wv)

        max_val = float(S_full.max().item())
        min_val = float(S_full.min().item())
        if not math.isfinite(max_val):
            max_val = 1.0
        if not math.isfinite(min_val):
            min_val = 0.0

        denom = max(max_val - min_val, 1e-5)
        normalized = ((S_full - min_val) / denom).clamp(0.0, 1.0)
        cutoff = float(_clamp(overlay_alpha, 0.0, 1.0))
        M = (normalized > cutoff).float()  # [D,H,W]

        # 4) Optional Gaussian blob around global argmax
        if blob_radius_vox > 0 and self.argmax_norm is not None:
            # convert normalized center to voxel coords
            cx = (self.argmax_norm[0].item() + 1) * 0.5 * (Wv - 1)
            cy = (self.argmax_norm[1].item() + 1) * 0.5 * (Hv - 1)
            cz = (self.argmax_norm[2].item() + 1) * 0.5 * (Dv - 1)
            xs = torch.arange(Wv, device=device)
            ys = torch.arange(Hv, device=device)
            zs = torch.arange(Dv, device=device)
            zz_idx, yy_idx, xx_idx = torch.meshgrid(zs, ys, xs, indexing='ij')
            d2 = (xx_idx - cx)**2 + (yy_idx - cy)**2 + (zz_idx - cz)**2
            sigma2 = (blob_radius_vox**2)
            blob = torch.exp(-d2 / (2.0 * (sigma2 + 1e-12)))
            # combine with thresholded mask
            M = M * blob

        # 5) Visibility threshold: sharpen the mask and control aggressiveness
        #    overlay_alpha in UI acts like a cutoff in [0,1]
        M = M.unsqueeze(-1)       # [D,H,W,1]

        # 6) Modulate volume alpha, set RGB of suppressed voxels to background (white)
        background_rgb = 1.0
        rgba_masked = rgba.clone()
        rgba_masked[..., 3:4] = rgba[..., 3:4] * M
        suppressed = (1.0 - M)
        rgba_masked[..., :3] = rgba[..., :3] * M + background_rgb * suppressed

        # 7) Single shared renderer call
        img = render_with_nerfacc(
            rgba_volume=rgba_masked,   # torch tensor keeps everything on device
            camera=cam,
            hw=image_hw,
            spp=None,
            batch_size=8192
        )  # [H,W,3] in [0,1]
        print(f"Rendered {image_hw[0]}x{image_hw[1]} view with {n_samples} samples/ray.")
        return (img.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)

    def _dense_coords(self, Dv, Hv, Wv):
        z = torch.linspace(-1, 1, Dv, device=device)
        y = torch.linspace(-1, 1, Hv, device=device)
        x = torch.linspace(-1, 1, Wv, device=device)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
        return torch.stack([xx, yy, zz], dim=-1)  # [D,H,W,3]

    @torch.no_grad()
    def build_rgba_volume(self) -> torch.Tensor:
        """
        Samples the INR once, applies the transfer function, returns torch [D,H,W,4] on device.
        This mirrors stage2.differentiable_render_from_inr volume construction.
        """
        Dv, Hv, Wv = self.opt["full_shape"]
        coords = self._dense_coords(Dv, Hv, Wv)                   # [D,H,W,3] in [-1,1]
        v = self.grid_inr(coords.view(-1, 3)).view(Dv, Hv, Wv, 1) # [D,H,W,1]
        v_min = self.grid_inr.min()
        v_max = self.grid_inr.max()
        v_norm = ((v - v_min) / (v_max - v_min + 1e-8)).clamp(0, 1)

        # transfer_function returns (rgb, alpha) in [0,1]
        rgb, alpha = self.transfer_fn(v_norm)
        if alpha.dim() == 3:
            alpha = alpha.unsqueeze(-1)
        rgba = torch.cat([rgb.clamp(0,1), alpha.clamp(0,0.999)], dim=-1).contiguous()
        return rgba  # [D,H,W,4]

class ImageWidget(QtWidgets.QLabel):
    """
    QLabel-based image viewer with orbit controls.
    Left-drag: orbit azimuth/polar
    Right-drag: orbit slowly (fine adjust)
    Wheel: zoom
    """
    requestRender = QtCore.pyqtSignal()

    def __init__(self, viewer, parent=None):
        # Keep an explicit handle to the viewer so orbit controls remain valid
        super().__init__(parent or viewer)
        self._viewer = viewer
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setBackgroundRole(QtGui.QPalette.Base)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setMinimumSize(256, 256)
        self._dragging = False
        self._last_pos = None
        self._slow = False

    def mousePressEvent(self, e):
        if e.button() in (QtCore.Qt.LeftButton, QtCore.Qt.RightButton):
            self._dragging = True
            self._slow = (e.button() == QtCore.Qt.RightButton)
            self._last_pos = e.pos()
            e.accept()
        else:
            super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._dragging and self._last_pos is not None:
            dx = e.x() - self._last_pos.x()
            dy = e.y() - self._last_pos.y()
            factor = 0.3 if not self._slow else 0.08
            self._viewer.nudge_orbit(dx * factor, dy * factor)
            self._last_pos = e.pos()
            self.requestRender.emit()
            e.accept()
        else:
            super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        self._dragging = False
        self._last_pos = None
        e.accept()

    def wheelEvent(self, e):
        delta = e.angleDelta().y() / 120.0  # steps
        self._viewer.nudge_zoom(-delta * 0.1)
        self.requestRender.emit()
        e.accept()

class ControlPanel(QtWidgets.QWidget):
    paramsChanged = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.text_edit = QtWidgets.QLineEdit(self)
        self.text_edit.setPlaceholderText("Type a phrase, e.g., 'thin branches'")

        self.find_btn = QtWidgets.QPushButton("Find region", self)

        self.blob_label = QtWidgets.QLabel("Blob radius (voxels):")
        self.blob_spin = QtWidgets.QDoubleSpinBox(self)
        self.blob_spin.setDecimals(1)
        self.blob_spin.setRange(0.0, 64.0)
        self.blob_spin.setSingleStep(1.0)
        self.blob_spin.setValue(0.0)

        self.alpha_label = QtWidgets.QLabel("Visibility threshold:")
        self.alpha_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.alpha_slider.setMinimum(0)
        self.alpha_slider.setMaximum(100)
        self.alpha_slider.setValue(90)
        self.alpha_value = QtWidgets.QLabel("0.90")

        self.res_label = QtWidgets.QLabel("Resolution (px):")
        self.res_spin = QtWidgets.QSpinBox(self)
        self.res_spin.setRange(128, 640)
        self.res_spin.setValue(512)
        self.res_spin.setSingleStep(32)

        self.samples_label = QtWidgets.QLabel("Samples per ray:")
        self.samples_spin = QtWidgets.QSpinBox(self)
        self.samples_spin.setRange(16, 256)
        self.samples_spin.setValue(48)
        self.samples_spin.setSingleStep(8)

        self.reset_btn = QtWidgets.QPushButton("Reset view", self)

        layout = QtWidgets.QFormLayout()
        layout.addRow("Phrase:", self.text_edit)
        layout.addRow(self.find_btn)
        layout.addRow(self.blob_label, self.blob_spin)
        layout.addRow(self.alpha_label, self.alpha_slider)
        layout.addRow(" ", self.alpha_value)
        layout.addRow(self.res_label, self.res_spin)
        layout.addRow(self.samples_label, self.samples_spin)
        layout.addRow(self.reset_btn)
        self.setLayout(layout)

        self.alpha_slider.valueChanged.connect(self._on_alpha)
        self.res_spin.valueChanged.connect(lambda _: self.paramsChanged.emit())
        self.samples_spin.valueChanged.connect(lambda _: self.paramsChanged.emit())

    def _on_alpha(self, v):
        self.alpha_value.setText(f"{v/100.0:.2f}")
        self.paramsChanged.emit()

    def overlay_alpha(self) -> float:
        return self.alpha_slider.value() / 100.0

    def blob_radius(self) -> float:
        return float(self.blob_spin.value())

    def render_resolution(self) -> int:
        return int(self.res_spin.value())

    def samples_per_ray(self) -> int:
        return int(self.samples_spin.value())

class Stage2Viewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stage-2 Semantic Viewer")

        # Core engine
        self.searcher = VolumeSemanticSearcher(
            opt=opt,
            stage1_path=STAGE1_PATH,
            stage2_head_path=STAGE2_HEAD_PATH,
            default_image_hw=(256, 256),
            default_samples=48
        )

        # UI
        central = QtWidgets.QWidget(self)
        self.image = ImageWidget(self)
        self.image.setStyleSheet("background-color: #222;")
        self.panel = ControlPanel(self)

        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.image, 1)
        hbox.addWidget(self.panel, 0)
        central.setLayout(hbox)
        self.setCentralWidget(central)
        self.resize(1100, 700)

        # Orbit state
        self.state = ViewerState()
        self._cam = Camera(
            azi_deg=self.state.azi_deg,
            polar_deg=self.state.polar_deg,
            dist=self.state.dist,
            center=self.state.center
        )  # same Camera as your render.py  :contentReference[oaicite:4]{index=4}

        # Signals
        self.image.requestRender.connect(self.render_once)
        self.panel.paramsChanged.connect(self.render_once)
        self.panel.find_btn.clicked.connect(self._on_find_region)
        self.panel.reset_btn.clicked.connect(self.reset_view)

        # Initial render (blank highlight)
        self._phrase_ready = False
        self.render_once()

    def nudge_orbit(self, dx_deg: float, dy_deg: float):
        self.state.azi_deg = (self.state.azi_deg + dx_deg) % 360.0
        self.state.polar_deg = _clamp(self.state.polar_deg + dy_deg, 5.0, 175.0)
        self._sync_camera()

    def nudge_zoom(self, delta: float):
        # delta positive -> zoom out
        self.state.dist = _clamp(self.state.dist * math.exp(delta), 50.0, 5000.0)
        self._sync_camera()

    def reset_view(self):
        self.state = ViewerState()
        self._sync_camera()
        self.render_once()

    def _sync_camera(self):
        self._cam = Camera(
            azi_deg=self.state.azi_deg,
            polar_deg=self.state.polar_deg,
            dist=self.state.dist,
            center=self.state.center
        )

    def _on_find_region(self):
        self._phrase_ready = False
        phrase = self.panel.text_edit.text().strip()
        try:
            if phrase:
                t0 = time.time()
                z_text = self.searcher.encode_text(phrase)
                self.searcher.build_similarity_grid(z_text)
                self._phrase_ready = True
                dt = time.time() - t0
                print(f"[viewer] Built similarity grid for '{phrase}' in {dt:.2f}s")
            self.render_once()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))

    def render_once(self):
        try:
            res = self.panel.render_resolution()
            spp = self.panel.samples_per_ray()
            self.searcher.image_hw = [res, res]
            self.searcher.n_samples = spp

            if self._phrase_ready:
                img = self.searcher.render_with_highlight(
                    cam=self._cam,
                    overlay_alpha=self.panel.overlay_alpha(),
                    blob_radius_vox=self.panel.blob_radius(),
                    image_hw=(res, res),
                    n_samples=spp
                )
            else:
                # Render without highlight
                rgba = self.searcher.build_rgba_volume()
                img_hw = render_with_nerfacc(
                    rgba_volume=rgba,
                    camera=self._cam,
                    hw=(res, res),
                    spp=None,
                    batch_size=8192
                )
                img = (img_hw.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)

            qimg = _to_qimage(img)
            self.image.setPixmap(QtGui.QPixmap.fromImage(qimg))
        except Exception as e:
            # Avoid crashing the UI
            print(f"[viewer] Render error: {e}")

def main():
    app = QtWidgets.QApplication(sys.argv)
    viewer = Stage2Viewer()
    viewer.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
