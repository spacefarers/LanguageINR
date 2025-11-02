#!/usr/bin/env python
"""Interactive viewer that overlays LSeg heatmaps on rendered INR views."""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import TRANSFER_FUNCTION_PATH, VOLUME_DIMS, device, dtype, opt
from model import NGP_TCNN
from render import Camera, ParaViewTransferFunction, render_with_nerfacc
from encoders.lseg_encoder.modules.lseg_module import LSegModule
import clip


STAGE1_PATH_DEFAULT = "./models/stage1_ngp_tcnn.pth"
TRANSFER_FUNCTION_DEFAULT = TRANSFER_FUNCTION_PATH


def _ensure(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def _dense_xy(H: int, W: int, dev: torch.device) -> torch.Tensor:
    y = torch.linspace(-1, 1, H, device=dev, dtype=dtype)
    x = torch.linspace(-1, 1, W, device=dev, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


class LSegHeatmapEngine:
    def __init__(
        self,
        stage1_path: str = STAGE1_PATH_DEFAULT,
        transfer_fn_path: str = TRANSFER_FUNCTION_DEFAULT,
        default_res_hw: Tuple[int, int] = (512, 512),
    ) -> None:
        self.image_hw = list(default_res_hw)

        _ensure(os.path.exists(stage1_path), f"Missing Stage-1 model at {stage1_path}")
        ckpt = torch.load(stage1_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model_opt = ckpt.get("opt", opt)
            self.grid_inr = NGP_TCNN(model_opt).to(device)
            self.grid_inr.load_state_dict(ckpt["model_state_dict"])
        else:
            self.grid_inr = NGP_TCNN(opt).to(device)
            self.grid_inr.load_state_dict(ckpt)
        self.grid_inr.eval()

        lseg_ckpt = "encoders/lseg_encoder/demo_e200.ckpt"
        _ensure(os.path.exists(lseg_ckpt), f"Missing LSeg model at {lseg_ckpt}")
        print(f"Loading LSeg teacher model from: {lseg_ckpt}")
        self.lseg_model = LSegModule.load_from_checkpoint(
            checkpoint_path=lseg_ckpt,
            data_path="",
            dataset="ignore",
            batch_size=1,
            base_lr=0,
            max_epochs=0,
            backbone="clip_vitl16_384",
            num_features=256,
            aux=False,
            se_loss=False,
            se_weight=0,
            aux_weight=0,
            ignore_index=-1,
            dropout=0.0,
            scale_inv=True,
            augment=False,
            no_batchnorm=False,
            widehead=True,
            widehead_hr=False,
            map_locatin="cpu",
            arch_option=0,
            strict=False,
            block_depth=0,
            activation="lrelu",
        ).to(device=device, dtype=dtype)
        self.lseg_model.eval()
        self.lseg_processor = self.lseg_model.val_transform

        self.transfer_fn = ParaViewTransferFunction(transfer_fn_path)

        X, Y, Z = map(int, VOLUME_DIMS)
        self._Wv, self._Hv, self._Dv = X, Y, Z
        self._rgba: Optional[torch.Tensor] = None
        self._v_norm: Optional[torch.Tensor] = None

    def _ensure_scalar_field(self) -> torch.Tensor:
        if self._v_norm is not None:
            return self._v_norm

        Dv, Hv, Wv = self._Dv, self._Hv, self._Wv
        xy_flat = _dense_xy(Hv, Wv, device)
        n_xy = xy_flat.shape[0]

        xz_coords = torch.empty((n_xy, 3), device=device, dtype=dtype)
        xz_coords[:, :2] = xy_flat
        slice_vals = torch.empty((n_xy, 1), device=device, dtype=dtype)
        volume = torch.empty((Dv, Hv, Wv, 1), device=device, dtype=dtype)

        z = torch.linspace(-1, 1, Dv, device=device, dtype=dtype)
        chunk = 16384

        with torch.no_grad():
            for zi in range(Dv):
                xz_coords[:, 2] = z[zi]
                for start in range(0, n_xy, chunk):
                    end = min(start + chunk, n_xy)
                    slice_vals[start:end] = self.grid_inr(xz_coords[start:end])
                volume[zi] = slice_vals.view(Hv, Wv, 1)

        v_min = volume.amin()
        v_max = volume.amax()
        v_norm = ((volume - v_min) / (v_max - v_min + 1e-8)).clamp(0, 1).contiguous()
        self._v_norm = v_norm
        return self._v_norm

    @torch.no_grad()
    def _build_rgba(self) -> torch.Tensor:
        if self._rgba is not None:
            return self._rgba
        v_norm = self._ensure_scalar_field()
        rgb, alpha = self.transfer_fn(v_norm)
        if alpha.dim() == 3:
            alpha = alpha.unsqueeze(-1)
        rgba = torch.cat([rgb.clamp(0, 1), alpha.clamp(0, 0.999)], dim=-1).contiguous()
        self._rgba = rgba.to(device=device, dtype=torch.float32)
        return self._rgba

    @torch.no_grad()
    def render_base(self, cam: Camera, image_hw: Tuple[int, int]) -> np.ndarray:
        rgba = self._build_rgba()
        img = render_with_nerfacc(
            rgba_volume=rgba,
            camera=cam,
            hw=image_hw,
            spp=None,
            batch_size=8192,
        )
        return (img.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)

    @torch.no_grad()
    def render_heatmap(self, cam: Camera, image_hw: Tuple[int, int], phrase: str) -> Tuple[np.ndarray, np.ndarray]:
        base = self.render_base(cam, image_hw)
        heat = self._compute_similarity_heatmap(base, image_hw, phrase)
        color = self._heatmap_to_rgb(heat, base.shape)
        overlay = self._blend_heatmap(base, color, heat)
        overlay_only = (np.clip(color, 0.0, 1.0) * 255.0).astype(np.uint8)
        return overlay, overlay_only

    @torch.no_grad()
    def _compute_similarity_heatmap(self, image_rgb: np.ndarray, image_hw: Tuple[int, int], phrase: str) -> np.ndarray:
        pil_image = Image.fromarray(image_rgb)
        pil_resized = pil_image.resize(image_hw[::-1], Image.BILINEAR)
        tensor = self.lseg_processor(pil_resized).unsqueeze(0).to(device=device, dtype=dtype)
        with torch.no_grad():
            feats = self.lseg_model(tensor, return_feature=True).to(torch.float32)
        feats = F.normalize(feats, dim=1)
        tokens = clip.tokenize([phrase or "object"]).to(device)
        with torch.no_grad():
            text_feat = self.lseg_model.net.clip_pretrained.encode_text(tokens)
        text_feat = F.normalize(text_feat.float(), dim=-1)[0]
        sim = torch.einsum("nchw,c->nhw", feats, text_feat)
        sim_map = sim.squeeze(0)
        if sim_map.shape != torch.Size(image_hw):
            sim_map = F.interpolate(sim_map.unsqueeze(0).unsqueeze(0), size=image_hw, mode="bilinear", align_corners=False).squeeze()
        sim_np = sim_map.detach().cpu().numpy()
        sim_np = sim_np - sim_np.min()
        denom = float(sim_np.max())
        if denom > 1e-6:
            sim_np /= denom
        return np.clip(sim_np, 0.0, 1.0)

    @staticmethod
    def _heatmap_to_rgb(heat: np.ndarray, reference_shape: Tuple[int, int, int]) -> np.ndarray:
        heat_norm = np.clip(heat, 0.0, 1.0)
        color = np.zeros(reference_shape, dtype=np.float32)
        color[..., 0] = heat_norm  # red
        color[..., 1] = heat_norm * 0.4
        color[..., 2] = 1.0 - heat_norm
        return color

    @staticmethod
    def _blend_heatmap(base_rgb: np.ndarray, color_rgb: np.ndarray, heat: np.ndarray) -> np.ndarray:
        base = base_rgb.astype(np.float32) / 255.0
        heat_norm = np.clip(heat, 0.0, 1.0)
        alpha = (0.15 + 0.65 * heat_norm)[..., None]
        blended = base * (1.0 - alpha) + color_rgb * alpha
        return (np.clip(blended, 0.0, 1.0) * 255.0).astype(np.uint8)


@dataclass
class Orbit:
    azi_deg: float
    polar_deg: float
    dist: float
    center: Tuple[float, float, float]


def _default_orbit(engine: LSegHeatmapEngine) -> Orbit:
    D, H, W = engine._Dv, engine._Hv, engine._Wv
    dist = math.sqrt(D * D + H * H + W * W) * 0.75
    center = (W / 2.0, H / 2.0, D / 2.0)
    return Orbit(azi_deg=20.0, polar_deg=80.0, dist=dist, center=center)


def _build_camera(orbit: Orbit) -> Camera:
    return Camera(azi_deg=orbit.azi_deg, polar_deg=orbit.polar_deg, dist=orbit.dist, center=orbit.center)


def _try_launch_gui(args, engine: LSegHeatmapEngine) -> int:
    try:
        from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore
    except Exception as exc:
        print("[viewer] PyQt5 not available, falling back to CLI:", exc)
        return 1

    class ImageWidget(QtWidgets.QLabel):
        requestRender = QtCore.pyqtSignal()

        def __init__(self, parent=None, orbit_nudge=None, zoom_nudge=None, drag_state=None):
            super().__init__(parent)
            self.setAlignment(QtCore.Qt.AlignCenter)
            self.setMinimumSize(256, 256)
            self._dragging = False
            self._last = None
            self._slow = False
            self._drag_enabled = True
            self._current_qimage = None
            self._orbit_cb = orbit_nudge
            self._zoom_cb = zoom_nudge
            self._drag_cb = drag_state

        def set_drag_enabled(self, enabled: bool) -> None:
            self._drag_enabled = bool(enabled)

        def mousePressEvent(self, event):
            if not self._drag_enabled:
                event.ignore()
                return
            if event.button() in (QtCore.Qt.LeftButton, QtCore.Qt.RightButton):
                self._dragging = True
                self._slow = event.button() == QtCore.Qt.RightButton
                self._last = event.pos()
                if self._drag_cb is not None:
                    self._drag_cb(True)
                event.accept()
            else:
                super().mousePressEvent(event)

        def mouseMoveEvent(self, event):
            if self._dragging and self._last is not None:
                dx = event.x() - self._last.x()
                dy = event.y() - self._last.y()
                factor = 0.3 if not self._slow else 0.08
                if self._orbit_cb is not None:
                    self._orbit_cb(dx * factor, dy * factor)
                self._last = event.pos()
                self.requestRender.emit()
                event.accept()
            else:
                super().mouseMoveEvent(event)

        def mouseReleaseEvent(self, event):
            if self._dragging:
                self._dragging = False
                self._last = None
                if self._drag_cb is not None:
                    self._drag_cb(False)
                self.requestRender.emit()
                event.accept()
            else:
                super().mouseReleaseEvent(event)

        def wheelEvent(self, event):
            if not self._drag_enabled:
                event.ignore()
                return
            delta = event.angleDelta().y() / 120.0
            if self._zoom_cb is not None:
                self._zoom_cb(-delta * 0.1)
            self.requestRender.emit()
            event.accept()

        def update_image(self, qimg: "QtGui.QImage") -> None:
            self._current_qimage = qimg.copy()
            self._apply_pixmap()

        def resizeEvent(self, event):
            super().resizeEvent(event)
            if self._current_qimage is not None:
                self._apply_pixmap()

        def _apply_pixmap(self) -> None:
            if self._current_qimage is None:
                return
            pixmap = QtGui.QPixmap.fromImage(self._current_qimage)
            size = self.size()
            if size.width() > 0 and size.height() > 0:
                pixmap = pixmap.scaled(size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.setPixmap(pixmap)

    class Panel(QtWidgets.QWidget):
        paramsChanged = QtCore.pyqtSignal()
        heatmapRequested = QtCore.pyqtSignal(str)
        heatmapCleared = QtCore.pyqtSignal()
        overlayModeChanged = QtCore.pyqtSignal()

        def __init__(self, default_res: int):
            super().__init__()
            self.text = QtWidgets.QLineEdit(self)
            self.text.setPlaceholderText("Describe what to highlight")
            self.text.setClearButtonEnabled(True)
            self.compute = QtWidgets.QPushButton("Compute heatmap", self)
            self.done = QtWidgets.QPushButton("Done", self)
            self.done.setEnabled(False)

            self.res_label = QtWidgets.QLabel("Resolution (px)")
            self.res_spin = QtWidgets.QSpinBox(self)
            self.res_spin.setRange(128, 1024)
            self.res_spin.setSingleStep(32)
            self.res_spin.setValue(default_res)

            self.reset = QtWidgets.QPushButton("Reset view", self)
            self.overlay_only_chk = QtWidgets.QCheckBox("Show overlay only", self)

            layout = QtWidgets.QFormLayout()
            layout.addRow("Phrase:", self.text)
            layout.addRow(self.compute)
            layout.addRow(self.done)
            layout.addRow(self.res_label, self.res_spin)
            layout.addRow(self.overlay_only_chk)
            layout.addRow(self.reset)
            self.setLayout(layout)

            self.compute.clicked.connect(lambda: self.heatmapRequested.emit(self.text.text().strip()))
            self.done.clicked.connect(self.heatmapCleared.emit)
            self.res_spin.valueChanged.connect(lambda _: self.paramsChanged.emit())
            self.overlay_only_chk.toggled.connect(lambda _: self.overlayModeChanged.emit())

        def res(self) -> int:
            return int(self.res_spin.value())

        def set_overlay_mode(self, active: bool) -> None:
            self.compute.setEnabled(not active)
            self.done.setEnabled(active)
            self.text.setEnabled(not active)
            self.res_spin.setEnabled(not active)

        def overlay_only(self) -> bool:
            return self.overlay_only_chk.isChecked()

    class Main(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("LSeg Heatmap Viewer")
            self.engine = engine
            self.orbit = _default_orbit(self.engine)
            self.cam = _build_camera(self.orbit)

            self.image = ImageWidget(
                self,
                orbit_nudge=self._nudge_orbit,
                zoom_nudge=self._nudge_zoom,
                drag_state=self._set_drag_state,
            )
            self.panel = Panel(args.res)
            self.image.setStyleSheet("background-color: #222;")

            central = QtWidgets.QWidget(self)
            layout = QtWidgets.QHBoxLayout(central)
            layout.addWidget(self.image, 1)
            layout.addWidget(self.panel, 0)
            self.setCentralWidget(central)
            self.resize(1100, 700)

            self.image.requestRender.connect(self.render_once)
            self.panel.paramsChanged.connect(self.render_once)
            self.panel.reset.clicked.connect(self._on_reset)
            self.panel.heatmapRequested.connect(self._on_heatmap)
            self.panel.heatmapCleared.connect(self._on_clear)
            self.panel.overlayModeChanged.connect(self._on_overlay_mode_changed)

            self._overlay_active = False
            self._drag_active = False
            self._last_overlay: Optional[np.ndarray] = None
            self._last_overlay_only: Optional[np.ndarray] = None
            self.render_once()

        def _on_reset(self):
            if self._overlay_active:
                return
            self.orbit = _default_orbit(self.engine)
            self.cam = _build_camera(self.orbit)
            self.render_once()

        def _nudge_orbit(self, dx_deg: float, dy_deg: float):
            self.orbit.azi_deg = (self.orbit.azi_deg - dx_deg) % 360.0
            self.orbit.polar_deg = max(5.0, min(175.0, self.orbit.polar_deg - dy_deg))
            self.cam = _build_camera(self.orbit)

        def _nudge_zoom(self, delta: float):
            self.orbit.dist = max(50.0, min(5000.0, self.orbit.dist * math.exp(delta)))
            self.cam = _build_camera(self.orbit)

        def _set_drag_state(self, active: bool):
            self._drag_active = active

        def render_once(self):
            if self._overlay_active:
                return
            res = self.panel.res()
            if self._drag_active:
                res = max(128, res // 2)
            self.engine.image_hw = [res, res]
            try:
                img = self.engine.render_base(self.cam, (res, res))
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "Render error", str(exc))
                return
            qimg = QtGui.QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1], QtGui.QImage.Format_RGB888)
            self.image.update_image(qimg)

        def _on_heatmap(self, phrase: str):
            if self._overlay_active:
                return
            if not phrase:
                QtWidgets.QMessageBox.warning(self, "Missing phrase", "Please enter a phrase before computing a heatmap.")
                return
            self._overlay_active = True
            self.panel.set_overlay_mode(True)
            self.image.set_drag_enabled(False)
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.BusyCursor)
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 50)
            try:
                res = self.panel.res()
                overlay, overlay_only = self.engine.render_heatmap(self.cam, (res, res), phrase)
                self._last_overlay = overlay
                self._last_overlay_only = overlay_only
                self._apply_overlay_display()
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "Heatmap error", str(exc))
                self._overlay_active = False
                self.panel.set_overlay_mode(False)
                self.image.set_drag_enabled(True)
                QtWidgets.QApplication.restoreOverrideCursor()
                return
            QtWidgets.QApplication.restoreOverrideCursor()

        def _on_clear(self):
            if not self._overlay_active:
                return
            self._overlay_active = False
            self.panel.set_overlay_mode(False)
            self.image.set_drag_enabled(True)
            self._last_overlay = None
            self._last_overlay_only = None
            self.render_once()

        def _apply_overlay_display(self):
            if not self._overlay_active:
                return
            if self.panel.overlay_only() and self._last_overlay_only is not None:
                img = self._last_overlay_only
            elif self._last_overlay is not None:
                img = self._last_overlay
            else:
                return
            qimg = QtGui.QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1], QtGui.QImage.Format_RGB888)
            self.image.update_image(qimg)

        def _on_overlay_mode_changed(self):
            if self._overlay_active:
                self._apply_overlay_display()

    app = QtWidgets.QApplication(sys.argv)
    win = Main()
    win.show()
    return app.exec_()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="LSeg heatmap overlay viewer")
    parser.add_argument("--stage1", default=STAGE1_PATH_DEFAULT, help="Path to Stage-1 INR checkpoint")
    parser.add_argument("--tf", default=TRANSFER_FUNCTION_DEFAULT, help="ParaView transfer function JSON")
    parser.add_argument("--res", type=int, default=512, help="Render resolution (square)")
    parser.add_argument("--phrase", type=str, default="tree", help="Text phrase for heatmap similarity")
    parser.add_argument("--overlay-only", action="store_true", help="Export overlay-only heatmap in CLI mode")
    parser.add_argument("--cli", action="store_true", help="Force CLI mode (skip GUI)")
    parser.add_argument("--save", type=str, default="heatmap_view.png", help="Path to save overlay in CLI mode")
    args = parser.parse_args(argv)

    engine = LSegHeatmapEngine(
        stage1_path=args.stage1,
        transfer_fn_path=args.tf,
        default_res_hw=(args.res, args.res),
    )

    if not args.cli:
        status = _try_launch_gui(args, engine)
        if status == 0:
            return status
        print("[viewer] Falling back to CLI mode...")

    orbit = _default_orbit(engine)
    cam = _build_camera(orbit)
    overlay, overlay_only = engine.render_heatmap(cam, (args.res, args.res), args.phrase.strip() or "object")
    output = overlay_only if args.overlay_only else overlay
    Image.fromarray(output).save(args.save)
    print(f"[viewer] Saved heatmap overlay to {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
