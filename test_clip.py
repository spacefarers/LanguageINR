#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from PyQt5 import QtCore, QtGui, QtWidgets

from models.clipseg import CLIPDensePredT


class ClipSegLocalizer:
    def __init__(self, weights_path="./weights/rd64-uni.pth", device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        weights_path = Path(weights_path).expanduser().resolve()
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"CLIPSeg weights not found at {weights_path}. "
                f"Download rd64-uni.pth from CLIPSeg repo."
            )

        self.model = CLIPDensePredT(version="ViT-B/16", reduce_dim=64)
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.model = self.model.to(self.device)

        for p in self.model.parameters():
            p.requires_grad = False

    @staticmethod
    def normalize_image(img_tensor):
        """ImageNet normalization for CLIPSeg."""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (img_tensor - mean.to(img_tensor.device)) / std.to(img_tensor.device)

    @torch.no_grad()
    def segment(self, pil_img: Image.Image, text: str, out_size: int = 352) -> np.ndarray:
        """
        Run CLIPSeg inference on an image with a text prompt.

        Args:
            pil_img: PIL Image (any size, will be resized to out_size)
            text: text description for segmentation
            out_size: CLIPSeg input resolution

        Returns:
            segmentation_map: numpy array [H, W] with scores in [0, 1]
        """
        orig_size = pil_img.size
        if pil_img.size != (out_size, out_size):
            pil_img = pil_img.resize((out_size, out_size), Image.BICUBIC)

        img_array = np.array(pil_img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        img_tensor = self.normalize_image(img_tensor)
        img_tensor = img_tensor.to(self.device)

        prediction = self.model(img_tensor, text)[0]
        prediction = torch.sigmoid(prediction)

        seg_map = F.interpolate(
            prediction, size=orig_size[::-1], mode='bilinear', align_corners=False
        )
        seg_map = seg_map[0, 0].cpu().numpy()

        return seg_map

    def best_region(self, pil_img: Image.Image, text: str, threshold: float = 0.5):
        """
        Find the best matching region in the image.

        Returns:
            (bbox_x, bbox_y, bbox_w, bbox_h, best_score, score_map)
        """
        seg_map = self.segment(pil_img, text)

        best_score = float(seg_map.max())

        binary_mask = (seg_map > threshold).astype(np.uint8)

        if binary_mask.sum() == 0:
            h, w = seg_map.shape
            return 0, 0, w, h, best_score, seg_map

        coords = np.column_stack(np.where(binary_mask > 0))
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
        else:
            h, w = seg_map.shape
            bbox = (0, 0, w, h)

        return bbox[0], bbox[1], bbox[2], bbox[3], best_score, seg_map


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
        self.setWindowTitle("CLIPSeg Region Localizer")
        self.resize(900, 600)

        self.image_path: Path | None = None
        self.pil_img: Image.Image | None = None
        self.scene = QtWidgets.QGraphicsScene(self)
        self.pixmap_item = None
        self.overlay_rect = None

        self.view = ImageView()
        self.view.setScene(self.scene)

        self.prompt_edit = QtWidgets.QLineEdit()
        self.prompt_edit.setPlaceholderText("Type a phrase, e.g., 'a cat', 'red ball', 'leaves' ...")

        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setValue(0.5)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setPrefix("threshold=")

        self.open_btn = QtWidgets.QPushButton("Open Image…")
        self.run_btn = QtWidgets.QPushButton("Find Region")
        self.run_btn.setDefault(True)
        self.status_lbl = QtWidgets.QLabel("Load an image to start.")
        self.status_lbl.setStyleSheet("color: #bbb")

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self.open_btn)
        controls.addWidget(self.prompt_edit, 1)
        controls.addWidget(self.threshold_spin)
        controls.addWidget(self.run_btn)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.status_lbl)

        self.localizer = ClipSegLocalizer()

        self.open_btn.clicked.connect(self.open_image)
        self.run_btn.clicked.connect(self.run_localization)

    def open_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if not path:
            return
        try:
            pil = Image.open(path).convert("RGB")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            return

        self.image_path = Path(path)
        self.pil_img = pil
        self.set_pixmap(pil)
        self.status_lbl.setText(f"Loaded {self.image_path.name} ({pil.size[0]}×{pil.size[1]}).")

    def set_pixmap(self, pil_img: Image.Image):
        self.scene.clear()
        self.overlay_rect = None

        qimg = self.pil2qimage(pil_img)
        pm = QtGui.QPixmap.fromImage(qimg)
        self.pixmap_item = self.scene.addPixmap(pm)
        self.scene.setSceneRect(QtCore.QRectF(pm.rect()))

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
            QtWidgets.QMessageBox.information(self, "No image", "Please open an image first.")
            return

        phrase = self.prompt_edit.text().strip()
        if not phrase:
            QtWidgets.QMessageBox.information(self, "No phrase", "Please enter a phrase to search for.")
            return

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            threshold = self.threshold_spin.value()
            bx, by, bw, bh, score, heat = self.localizer.best_region(
                self.pil_img, phrase, threshold=threshold
            )
        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        heat = np.clip(heat, 0.0, 1.0)
        h, w = heat.shape
        heat_img = (255 * heat).astype(np.uint8)
        alpha = np.clip((heat * 200.0), 0.0, 255.0).astype(np.uint8)

        rgba_arr = np.zeros((h, w, 4), dtype=np.uint8)
        rgba_arr[..., 0] = heat_img
        rgba_arr[..., 3] = alpha
        heat_overlay = Image.fromarray(rgba_arr, mode="RGBA")

        base_rgba = self.pil_img.convert("RGBA")
        if base_rgba.size != (w, h):
            base_rgba = base_rgba.resize((w, h), Image.BICUBIC)
        composite = Image.alpha_composite(base_rgba, heat_overlay)

        self.set_pixmap(composite.convert("RGB"))
        self.draw_overlay_rect(bx, by, bw, bh)

        self.status_lbl.setText(
            f"Best region at (x={bx}, y={by}, w={bw}, h={bh}) — score {score:.4f}"
        )

    def draw_overlay_rect(self, x, y, w, h):
        for item in self.scene.items():
            if isinstance(item, QtWidgets.QGraphicsRectItem):
                self.scene.removeItem(item)

        rect_item = QtWidgets.QGraphicsRectItem(QtCore.QRectF(x, y, w, h))
        pen = QtGui.QPen(QtGui.QColor(255, 50, 50))
        pen.setWidth(2)
        rect_item.setPen(pen)
        brush = QtGui.QBrush(QtGui.QColor(255, 50, 50, 60))
        rect_item.setBrush(brush)
        rect_item.setZValue(10)
        self.scene.addItem(rect_item)
        self.overlay_rect = rect_item

        self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
