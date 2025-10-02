#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

import open_clip

from PyQt5 import QtCore, QtGui, QtWidgets


# ----------------------
# CLIP sliding-window IO
# ----------------------
class ClipLocalizer:
    def __init__(self, device=None,
                 model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
        import torch
        import open_clip

        if device is None:
            device = "cuda:1" if torch.cuda.is_available() else "cpu"
        self.device = device

        res = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        # Handle both old and new open_clip return signatures
        if isinstance(res, tuple) and len(res) == 3:
            self.model, _preprocess_train, self.preprocess = res
        elif isinstance(res, tuple) and len(res) == 2:
            self.model, self.preprocess = res
        else:
            raise RuntimeError(
                f"Unexpected create_model_and_transforms return: type={type(res)}, len={len(res) if isinstance(res, tuple) else 'n/a'}"
            )

        self.model.eval().to(self.device)

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        tokens = open_clip.tokenize([text]).to(self.device)
        t = self.model.encode_text(tokens)
        t = t / t.norm(dim=-1, keepdim=True)
        return t  # shape [1, d]

    @torch.no_grad()
    def encode_patch(self, pil_img: Image.Image) -> torch.Tensor:
        """Return normalized image embedding for a single patch PIL image."""
        img = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        f = self.model.encode_image(img)
        f = f / f.norm(dim=-1, keepdim=True)
        return f  # shape [1, d]

    def best_region(
        self,
        pil_img_224: Image.Image,
        text: str,
        patch_size: int = 64,
        stride: int = 16,
    ):
        """
        Slide a window across the 224x224 image and return:
          (best_x, best_y, patch_w, patch_h, best_score, score_map)
        Coordinates are top-left in pixel space of the original 224x224.
        score_map is a 2D numpy array of similarities for visualization if desired.
        """
        if pil_img_224.size != (224, 224):
            raise ValueError(f"Expected image size 224x224, got {pil_img_224.size}")

        text_feat = self.encode_text(text)  # [1, d]

        W, H = pil_img_224.size
        ps = patch_size
        st = stride

        xs = list(range(0, W - ps + 1, st))
        ys = list(range(0, H - ps + 1, st))
        if xs[-1] != W - ps:
            xs.append(W - ps)
        if ys[-1] != H - ps:
            ys.append(H - ps)

        score_map = np.zeros((len(ys), len(xs)), dtype=np.float32)
        best = (-1e9, 0, 0)

        for yi, y in enumerate(ys):
            for xi, x in enumerate(xs):
                crop = pil_img_224.crop((x, y, x + ps, y + ps))
                img_feat = self.encode_patch(crop)  # [1, d]
                sim = float((img_feat @ text_feat.T).item())  # cosine sim
                score_map[yi, xi] = sim
                if sim > best[0]:
                    best = (sim, x, y)

        best_score, bx, by = best
        return bx, by, ps, ps, best_score, score_map


@torch.no_grad()
def clip_patch_heatmap(model, preprocess, pil_img: Image.Image, text: str, device="cuda"):
    """Return (heatmap, grid_size, bbox, score) for CLIP ViT patch tokens, enhanced with GEM."""

    device = device if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    tokens = open_clip.tokenize([text]).to(device)
    text_feat = model.encode_text(tokens)  # [1, D]
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    img = preprocess(pil_img).unsqueeze(0).to(device)

    visual = model.visual
    saved_tokens = {}

    def hook_fn(_module, _inputs, output):
        saved_tokens["tokens"] = output

    handle = visual.transformer.register_forward_hook(hook_fn)
    _ = model.encode_image(img)
    handle.remove()

    if "tokens" not in saved_tokens:
        raise RuntimeError("Failed to capture transformer tokens from CLIP visual encoder.")

    tokens_out = saved_tokens["tokens"]  # [B, 1+N, width]
    patch_tokens = tokens_out[:, 1:, :]  # [B, N, D]
    x = patch_tokens.squeeze(0)  # [N, D]

    # GEM implementation starts here
    last_block = visual.transformer.resblocks[-1]
    attn = last_block.attn
    qkv_weight = attn.in_proj_weight.data  # [3*D, D] (changed from qkv.weight to in_proj_weight)
    dim = visual.embed_dim
    W_q = qkv_weight[0:dim, :]
    W_k = qkv_weight[dim:2 * dim, :]
    W_v = qkv_weight[2 * dim:3 * dim, :]

    n, d = x.shape
    sum_norms = torch.sum(torch.norm(x, dim=1))
    tau = n * math.sqrt(d) / sum_norms.item()

    def compute_p_K(W_proj, K=1):
        p = torch.matmul(x, W_proj.T)
        p = p / torch.norm(p, dim=-1, keepdim=True)
        for _ in range(K):
            sim = torch.matmul(p, p.T) / tau
            attn = F.softmax(sim, dim=-1)
            p = torch.matmul(attn, p)
            p = p / torch.norm(p, dim=-1, keepdim=True)
        return p

    p_q = compute_p_K(W_q)
    p_k = compute_p_K(W_k)
    p_v = compute_p_K(W_v)

    V = torch.matmul(x, W_v.T)

    def compute_O(p, V):
        sim = torch.matmul(p, p.T) / tau
        attn = F.softmax(sim, dim=-1)
        O = torch.matmul(attn, V)
        return O

    O_q = compute_O(p_q, V)
    O_k = compute_O(p_k, V)
    O_v = compute_O(p_v, V)

    O_qkv = (O_q + O_k + O_v) / 3

    # Apply ln_post and proj to refined features
    patch_feats = visual.ln_post(O_qkv.unsqueeze(0))
    proj = getattr(visual, "proj", None)
    if proj is not None:
        patch_feats = patch_feats @ proj
    patch_feats = patch_feats / patch_feats.norm(dim=-1, keepdim=True)

    sim = torch.matmul(patch_feats, text_feat.T).squeeze(-1)
    sim = sim.squeeze(0)  # [N]

    patch_size = getattr(visual, "patch_size", None)
    if patch_size is None:
        patch_size = getattr(getattr(visual, "patch_embed", None), "patch_size", None)
    if isinstance(patch_size, (tuple, list)):
        patch_h, patch_w = patch_size[:2]
        if patch_h != patch_w:
            raise ValueError(f"Non-square patch size unsupported: {patch_size}")
        patch_size = patch_h
    if patch_size is None:
        patch_size = 32

    grid_size = int(round(224 / patch_size))
    heat = sim.reshape(grid_size, grid_size)

    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
    heat_up = F.interpolate(
        heat.unsqueeze(0).unsqueeze(0),
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    ).squeeze().float().cpu().numpy()

    idx = int(torch.argmax(sim).item())
    iy, ix = divmod(idx, grid_size)
    bbox = (ix * patch_size, iy * patch_size, patch_size, patch_size)
    best_score = float(sim.max().item())

    return heat_up, (grid_size, grid_size), bbox, best_score


# ----------------------
# PyQt UI
# ----------------------
class ImageView(QtWidgets.QGraphicsView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setRenderHint(QtGui.QPainter.Antialiasing, True)
        self.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.FullViewportUpdate)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(20, 20, 20)))

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CLIP Region Localizer (open_clip + PyQt5)")
        self.resize(900, 600)

        # State
        self.image_path: Path | None = None
        self.pil_img: Image.Image | None = None
        self.scene = QtWidgets.QGraphicsScene(self)
        self.pixmap_item = None
        self.overlay_rect = None

        # Widgets
        self.view = ImageView()
        self.view.setScene(self.scene)

        self.prompt_edit = QtWidgets.QLineEdit()
        self.prompt_edit.setPlaceholderText("Type a phrase, e.g., 'a cat', 'red ball', 'text on paper' ...")

        self.patch_spin = QtWidgets.QSpinBox()
        self.patch_spin.setRange(16, 224)
        self.patch_spin.setValue(64)
        self.patch_spin.setSingleStep(8)
        self.patch_spin.setPrefix("patch=")
        self.patch_spin.setEnabled(False)
        self.patch_spin.setToolTip("Sliding-window controls disabled; using ViT patch heatmap.")

        self.stride_spin = QtWidgets.QSpinBox()
        self.stride_spin.setRange(4, 128)
        self.stride_spin.setValue(16)
        self.stride_spin.setSingleStep(4)
        self.stride_spin.setPrefix("stride=")
        self.stride_spin.setEnabled(False)
        self.stride_spin.setToolTip("Sliding-window controls disabled; using ViT patch heatmap.")

        self.open_btn = QtWidgets.QPushButton("Open 224×224 Image…")
        self.run_btn = QtWidgets.QPushButton("Find Region")
        self.run_btn.setDefault(True)
        self.status_lbl = QtWidgets.QLabel("Load an image to start.")
        self.status_lbl.setStyleSheet("color: #bbb")

        # Layout
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self.open_btn)
        controls.addWidget(self.prompt_edit, 1)
        controls.addWidget(self.patch_spin)
        controls.addWidget(self.stride_spin)
        controls.addWidget(self.run_btn)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.status_lbl)

        # Logic
        self.localizer = ClipLocalizer()

        self.open_btn.clicked.connect(self.open_image)
        self.run_btn.clicked.connect(self.run_localization)

    def open_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open 224×224 image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if not path:
            return
        try:
            pil = Image.open(path).convert("RGB")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            return

        if pil.size != (224, 224):
            # Offer to auto-resize to 224×224 for convenience
            reply = QtWidgets.QMessageBox.question(
                self,
                "Resize?",
                f"Image is {pil.size}, expected 224×224.\nResize a copy to 224×224?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if reply == QtWidgets.QMessageBox.Yes:
                pil = pil.resize((224, 224), Image.BICUBIC)
            else:
                return

        self.image_path = Path(path)
        self.pil_img = pil
        self.set_pixmap(pil)
        self.status_lbl.setText(f"Loaded {self.image_path.name} ({pil.size[0]}×{pil.size[1]}).")

    def set_pixmap(self, pil_img: Image.Image):
        # Clear scene
        self.scene.clear()
        self.overlay_rect = None

        # Convert PIL to QPixmap
        qimg = self.pil2qimage(pil_img)
        pm = QtGui.QPixmap.fromImage(qimg)
        self.pixmap_item = self.scene.addPixmap(pm)
        # QGraphicsScene expects QRectF, while QPixmap.rect() returns QRect
        self.scene.setSceneRect(QtCore.QRectF(pm.rect()))

        # Center view
        self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.scene.items():
            self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def pil2qimage(self, pil_img: Image.Image) -> QtGui.QImage:
        rgb = pil_img.convert("RGB")
        w, h = rgb.size
        data = rgb.tobytes("raw", "RGB")
        qimg = QtGui.QImage(data, w, h, QtGui.QImage.Format.Format_RGB888)
        return qimg

    def run_localization(self):
        if self.pil_img is None:
            QtWidgets.QMessageBox.information(self, "No image", "Please open a 224×224 image first.")
            return

        phrase = self.prompt_edit.text().strip()
        if not phrase:
            QtWidgets.QMessageBox.information(self, "No phrase", "Please enter a phrase to search for.")
            return

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            base_img = self.pil_img
            if base_img.size != (224, 224):
                base_img = base_img.resize((224, 224), Image.BICUBIC)

            heat, (gy, gx), (bx, by, bw, bh), score = clip_patch_heatmap(
                self.localizer.model,
                self.localizer.preprocess,
                base_img,
                phrase,
                device=self.localizer.device,
            )
        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        heat = np.clip(heat, 0.0, 1.0)
        heat_img = (255 * heat).astype(np.uint8)
        alpha = np.clip((heat * 180.0), 0.0, 255.0).astype(np.uint8)

        rgba_arr = np.zeros((224, 224, 4), dtype=np.uint8)
        rgba_arr[..., 0] = heat_img
        rgba_arr[..., 3] = alpha
        heat_overlay = Image.fromarray(rgba_arr, mode="RGBA")

        base_rgba = base_img.convert("RGBA")
        composite = Image.alpha_composite(base_rgba, heat_overlay)

        self.set_pixmap(composite.convert("RGB"))
        self.draw_overlay_rect(bx, by, bw, bh)

        self.status_lbl.setText(
            f"Best patch {gx}x{gy} grid at (x={bx}, y={by}, w={bw}, h={bh}) — cosine {score:.4f}"
        )

    def draw_overlay_rect(self, x, y, w, h):
        # Remove previous overlay if any
        for item in self.scene.items():
            # Keep the pixmap; remove any rectangles we added
            if isinstance(item, QtWidgets.QGraphicsRectItem):
                self.scene.removeItem(item)

        rect_item = QtWidgets.QGraphicsRectItem(QtCore.QRectF(x, y, w, h))
        pen = QtGui.QPen(QtGui.QColor(255, 50, 50))
        pen.setWidth(2)
        rect_item.setPen(pen)
        brush = QtGui.QBrush(QtGui.QColor(255, 50, 50, 60))  # translucent fill
        rect_item.setBrush(brush)
        rect_item.setZValue(10)
        self.scene.addItem(rect_item)
        self.overlay_rect = rect_item

        # Keep view fitted
        self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()