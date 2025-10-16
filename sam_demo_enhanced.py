"""
SAM Hierarchy Demo - Enhanced version with tree structure and individual mask visualization

Shows:
1. Original image
2. Individual masks colored by hierarchy level
3. Tree structure showing parent-child relationships
4. Click to highlight specific masks and their children
"""

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                 QHBoxLayout, QPushButton, QLabel, QFileDialog,
                                 QScrollArea, QGridLayout, QProgressBar, QGroupBox,
                                 QSplitter, QTextEdit, QComboBox, QTreeWidget,
                                 QTreeWidgetItem, QTabWidget)
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QPixmap, QImage, QFont, QColor
except ImportError:
    print("PyQt5 not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyQt5"])
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                 QHBoxLayout, QPushButton, QLabel, QFileDialog,
                                 QScrollArea, QGridLayout, QProgressBar, QGroupBox,
                                 QSplitter, QTextEdit, QComboBox, QTreeWidget,
                                 QTreeWidgetItem, QTabWidget)
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QPixmap, QImage, QFont, QColor

# Import from stage2
from config import device
from stage2 import (
    build_sam_generator,
    load_clipseg_model,
    clipseg_image_encoder,
    _sam_partition_masks
)


def compute_mask_containment(mask_child: np.ndarray, mask_parent: np.ndarray, iou_threshold: float = 0.7) -> bool:
    """
    Check if mask_child is spatially contained within mask_parent.

    Returns True if mask_child is mostly inside mask_parent (high IoU with parent).
    """
    intersection = np.logical_and(mask_child, mask_parent).sum()
    child_area = mask_child.sum()

    if child_area == 0:
        return False

    # Child is contained if most of its area overlaps with parent
    containment_ratio = intersection / child_area
    return containment_ratio >= iou_threshold


def build_hierarchy_tree(groups: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
    """
    Build a tree structure showing parent-child relationships between masks.

    Hierarchy: Whole contains Parts, Parts contain Smalls
    """
    tree = []

    # Start with Whole masks as root nodes
    for w_idx, whole_mask in enumerate(groups['w']):
        whole_node = {
            'level': 'whole',
            'idx': w_idx,
            'mask_data': whole_mask,
            'children': []
        }

        whole_seg = whole_mask['segmentation']

        # Find Part masks contained in this Whole
        for p_idx, part_mask in enumerate(groups['p']):
            part_seg = part_mask['segmentation']

            if compute_mask_containment(part_seg, whole_seg, iou_threshold=0.5):
                part_node = {
                    'level': 'part',
                    'idx': p_idx,
                    'mask_data': part_mask,
                    'children': []
                }

                # Find Small masks contained in this Part
                for s_idx, small_mask in enumerate(groups['s']):
                    small_seg = small_mask['segmentation']

                    if compute_mask_containment(small_seg, part_seg, iou_threshold=0.5):
                        small_node = {
                            'level': 'small',
                            'idx': s_idx,
                            'mask_data': small_mask,
                            'children': []
                        }
                        part_node['children'].append(small_node)

                whole_node['children'].append(part_node)

        # Also check for Small masks directly in Whole (skip Part level if no intermediate Parts)
        for s_idx, small_mask in enumerate(groups['s']):
            small_seg = small_mask['segmentation']

            # Only add if not already added as child of a Part
            already_added = False
            for part_node in whole_node['children']:
                for small_node in part_node['children']:
                    if small_node['idx'] == s_idx:
                        already_added = True
                        break

            if not already_added and compute_mask_containment(small_seg, whole_seg, iou_threshold=0.5):
                small_node = {
                    'level': 'small',
                    'idx': s_idx,
                    'mask_data': small_mask,
                    'children': []
                }
                whole_node['children'].append(small_node)

        tree.append(whole_node)

    # Add orphaned Part masks (not contained in any Whole)
    for p_idx, part_mask in enumerate(groups['p']):
        already_added = False
        for whole_node in tree:
            for child in whole_node['children']:
                if child['level'] == 'part' and child['idx'] == p_idx:
                    already_added = True
                    break

        if not already_added:
            part_seg = part_mask['segmentation']
            part_node = {
                'level': 'part',
                'idx': p_idx,
                'mask_data': part_mask,
                'children': []
            }

            # Find Small masks contained in this Part
            for s_idx, small_mask in enumerate(groups['s']):
                small_seg = small_mask['segmentation']

                if compute_mask_containment(small_seg, part_seg, iou_threshold=0.5):
                    small_node = {
                        'level': 'small',
                        'idx': s_idx,
                        'mask_data': small_mask,
                        'children': []
                    }
                    part_node['children'].append(small_node)

            tree.append(part_node)

    # Add orphaned Small masks
    for s_idx, small_mask in enumerate(groups['s']):
        already_added = False
        for node in tree:
            if _is_mask_in_tree(s_idx, 'small', node):
                already_added = True
                break

        if not already_added:
            small_node = {
                'level': 'small',
                'idx': s_idx,
                'mask_data': small_mask,
                'children': []
            }
            tree.append(small_node)

    return tree


def _is_mask_in_tree(idx: int, level: str, node: Dict) -> bool:
    """Check if a mask with given idx and level is in the tree."""
    if node['level'] == level and node['idx'] == idx:
        return True
    for child in node['children']:
        if _is_mask_in_tree(idx, level, child):
            return True
    return False


class SAMProcessor(QThread):
    """Background thread for SAM processing to keep UI responsive."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, image_path: str, sam_gen, clipseg_model, max_regions: int = 128):
        super().__init__()
        self.image_path = image_path
        self.sam_gen = sam_gen
        self.clipseg_model = clipseg_model
        self.max_regions = max_regions

    def run(self):
        try:
            self.progress.emit("Loading image...")

            # Load image
            from PIL import Image
            img_pil = Image.open(self.image_path).convert('RGB')
            img_np = np.array(img_pil)

            # Convert to torch tensor [H, W, 3] in range [0, 1]
            img_tensor = torch.from_numpy(img_np).float() / 255.0

            self.progress.emit("Running SAM segmentation...")

            # Run SAM
            masks = self.sam_gen.generate(img_np)

            self.progress.emit(f"Generated {len(masks)} masks. Partitioning into hierarchies...")

            # Partition into hierarchies
            groups = _sam_partition_masks(masks)

            self.progress.emit("Building hierarchy tree...")

            # Build tree structure
            tree = build_hierarchy_tree(groups)

            self.progress.emit("Generating CLIPSeg embeddings...")

            # Generate hierarchy maps
            H, W = img_tensor.shape[:2]
            targets: Dict[str, torch.Tensor] = {}

            for key in ("s", "p", "w"):
                hierarchy_name = {"s": "Small", "p": "Part", "w": "Whole"}[key]
                self.progress.emit(f"Processing {hierarchy_name} regions ({len(groups[key])} masks)...")

                L = torch.zeros((H, W, 512), device=device, dtype=torch.float32)
                count = 0

                for m in groups[key]:
                    if count >= self.max_regions:
                        break

                    seg = torch.from_numpy(m["segmentation"]).to(device=device, dtype=torch.float32)
                    if seg.sum() < 8:
                        continue

                    masked = img_tensor.to(device) * seg.unsqueeze(-1)
                    z = clipseg_image_encoder(self.clipseg_model, masked)
                    z = F.normalize(z, dim=-1)
                    L[seg > 0.5] = z
                    count += 1

                targets[key] = L.cpu()

            self.progress.emit("Processing complete!")

            # Return results
            result = {
                'image': img_np,
                'image_tensor': img_tensor.cpu(),
                'masks': masks,
                'groups': groups,
                'targets': targets,
                'tree': tree,
                'stats': self._compute_stats(groups, H, W, tree)
            }

            self.finished.emit(result)

        except Exception as e:
            import traceback
            self.error.emit(f"Error: {str(e)}\n{traceback.format_exc()}")

    def _compute_stats(self, groups: Dict[str, List[Dict]], H: int, W: int, tree: List[Dict]) -> Dict[str, Any]:
        """Compute statistics for each hierarchy level."""
        stats = {}
        total_pixels = H * W

        for key, name in [("s", "Small"), ("p", "Part"), ("w", "Whole")]:
            masks = groups[key]
            if not masks:
                stats[key] = {
                    'name': name,
                    'num_masks': 0,
                    'total_pixels': 0,
                    'coverage_pct': 0.0,
                    'avg_area': 0.0,
                    'min_area': 0,
                    'max_area': 0
                }
            else:
                areas = [m['area'] for m in masks]
                total_active = sum(areas)

                stats[key] = {
                    'name': name,
                    'num_masks': len(masks),
                    'total_pixels': total_active,
                    'coverage_pct': (total_active / total_pixels) * 100,
                    'avg_area': np.mean(areas),
                    'min_area': np.min(areas),
                    'max_area': np.max(areas)
                }

        stats['tree'] = {
            'num_roots': len(tree),
            'total_nodes': self._count_tree_nodes(tree)
        }

        return stats

    def _count_tree_nodes(self, tree: List[Dict]) -> int:
        """Count total nodes in tree."""
        count = 0
        for node in tree:
            count += 1
            count += self._count_tree_nodes(node['children'])
        return count


class ImageLabel(QLabel):
    """Custom QLabel for displaying images with proper scaling."""

    clicked = pyqtSignal()

    def __init__(self, title: str = ""):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(200, 200)
        self.setStyleSheet("QLabel { background-color: #2b2b2b; border: 2px solid #555; }")
        self.title = title
        self._pixmap = None

    def set_image(self, img_np: np.ndarray, title: str = None):
        """Set image from numpy array."""
        if title:
            self.title = title

        # Convert numpy to QPixmap
        if img_np.dtype != np.uint8:
            img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)

        h, w = img_np.shape[:2]

        if len(img_np.shape) == 2:  # Grayscale
            qimg = QImage(img_np.data, w, h, w, QImage.Format_Grayscale8)
        else:  # RGB
            bytes_per_line = 3 * w
            qimg = QImage(img_np.data, w, h, bytes_per_line, QImage.Format_RGB888)

        self._pixmap = QPixmap.fromImage(qimg)
        self.update_display()

    def update_display(self):
        """Update the display with current pixmap."""
        if self._pixmap:
            scaled = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled)

    def resizeEvent(self, event):
        """Handle resize events."""
        super().resizeEvent(event)
        self.update_display()

    def mousePressEvent(self, event):
        """Handle mouse click."""
        self.clicked.emit()


class SAMDemoWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()

        self.sam_gen = None
        self.clipseg_model = None
        self.current_result = None
        self.selected_node = None

        self.setWindowTitle("SAM Hierarchy Demo - Enhanced")
        self.setGeometry(100, 100, 1800, 1000)

        self.init_ui()
        self.init_models()

    def init_ui(self):
        """Initialize the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # Top control panel
        control_panel = QHBoxLayout()

        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)
        self.load_btn.setMinimumHeight(40)

        self.process_btn = QPushButton("Process with SAM")
        self.process_btn.clicked.connect(self.process_image)
        self.process_btn.setEnabled(False)
        self.process_btn.setMinimumHeight(40)

        self.export_btn = QPushButton("Export Results")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        self.export_btn.setMinimumHeight(40)

        self.max_regions_combo = QComboBox()
        self.max_regions_combo.addItems(["32", "64", "128", "256"])
        self.max_regions_combo.setCurrentText("128")

        control_panel.addWidget(QLabel("Max Regions:"))
        control_panel.addWidget(self.max_regions_combo)
        control_panel.addWidget(self.load_btn)
        control_panel.addWidget(self.process_btn)
        control_panel.addWidget(self.export_btn)
        control_panel.addStretch()

        main_layout.addLayout(control_panel)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Ready. Load an image to begin.")
        self.status_label.setStyleSheet("QLabel { padding: 5px; background-color: #333; }")
        main_layout.addWidget(self.status_label)

        # Main content area with splitter (3 panels)
        splitter = QSplitter(Qt.Horizontal)

        # Left panel: Images
        images_scroll = QScrollArea()
        images_scroll.setWidgetResizable(True)
        images_widget = QWidget()
        images_layout = QVBoxLayout(images_widget)

        # Original image
        self.original_label = ImageLabel("Original")
        images_layout.addWidget(QLabel("<b>Original Image</b>", alignment=Qt.AlignCenter))
        images_layout.addWidget(self.original_label)

        # Individual masks visualization
        self.individual_masks_label = ImageLabel("Individual Masks")
        self.individual_masks_label.clicked.connect(self.reset_selection)
        images_layout.addWidget(QLabel("<b>Individual Masks (colored by hierarchy)</b>", alignment=Qt.AlignCenter))
        images_layout.addWidget(self.individual_masks_label)

        # Selected mask visualization
        self.selected_mask_label = ImageLabel("Selected Mask")
        images_layout.addWidget(QLabel("<b>Selected Mask & Children</b>", alignment=Qt.AlignCenter))
        images_layout.addWidget(self.selected_mask_label)

        images_scroll.setWidget(images_widget)

        # Middle panel: Tree structure
        tree_group = QGroupBox("Hierarchy Tree")
        tree_layout = QVBoxLayout()

        help_label = QLabel(
            "Click on a node to highlight it and its children.\n"
            "Colors: Whole=Blue, Part=Green, Small=Red"
        )
        help_label.setWordWrap(True)
        help_label.setStyleSheet("QLabel { padding: 5px; background-color: #333; }")
        tree_layout.addWidget(help_label)

        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(["Mask", "Area", "Level"])
        self.tree_widget.itemClicked.connect(self.on_tree_item_clicked)
        tree_layout.addWidget(self.tree_widget)

        tree_group.setLayout(tree_layout)

        # Right panel: Statistics and visualizations
        right_panel = QTabWidget()

        # Tab 1: Statistics
        stats_widget = QWidget()
        stats_layout = QVBoxLayout(stats_widget)
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setFont(QFont("Courier", 9))
        stats_layout.addWidget(self.stats_text)
        right_panel.addTab(stats_widget, "Statistics")

        # Tab 2: Hierarchy visualizations
        viz_widget = QWidget()
        viz_layout = QGridLayout(viz_widget)

        self.small_mask_label = ImageLabel("Small - Mask")
        self.small_pca_label = ImageLabel("Small - PCA")
        self.part_mask_label = ImageLabel("Part - Mask")
        self.part_pca_label = ImageLabel("Part - PCA")
        self.whole_mask_label = ImageLabel("Whole - Mask")
        self.whole_pca_label = ImageLabel("Whole - PCA")

        viz_layout.addWidget(QLabel("<b>Small</b>", alignment=Qt.AlignCenter), 0, 0, 1, 2)
        viz_layout.addWidget(QLabel("Mask"), 1, 0)
        viz_layout.addWidget(self.small_mask_label, 2, 0)
        viz_layout.addWidget(QLabel("PCA"), 1, 1)
        viz_layout.addWidget(self.small_pca_label, 2, 1)

        viz_layout.addWidget(QLabel("<b>Part</b>", alignment=Qt.AlignCenter), 3, 0, 1, 2)
        viz_layout.addWidget(QLabel("Mask"), 4, 0)
        viz_layout.addWidget(self.part_mask_label, 5, 0)
        viz_layout.addWidget(QLabel("PCA"), 4, 1)
        viz_layout.addWidget(self.part_pca_label, 5, 1)

        viz_layout.addWidget(QLabel("<b>Whole</b>", alignment=Qt.AlignCenter), 6, 0, 1, 2)
        viz_layout.addWidget(QLabel("Mask"), 7, 0)
        viz_layout.addWidget(self.whole_mask_label, 8, 0)
        viz_layout.addWidget(QLabel("PCA"), 7, 1)
        viz_layout.addWidget(self.whole_pca_label, 8, 1)

        right_panel.addTab(viz_widget, "Hierarchy Views")

        # Add to splitter
        splitter.addWidget(images_scroll)
        splitter.addWidget(tree_group)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 1)

        main_layout.addWidget(splitter)

        # Apply dark theme
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1e1e1e;
                color: #d4d4d4;
            }
            QPushButton {
                background-color: #0e639c;
                color: white;
                border: none;
                padding: 8px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #888;
            }
            QLabel {
                color: #d4d4d4;
            }
            QTextEdit {
                background-color: #252526;
                color: #d4d4d4;
                border: 1px solid #555;
            }
            QProgressBar {
                border: 1px solid #555;
                border-radius: 3px;
                text-align: center;
                background-color: #252526;
            }
            QProgressBar::chunk {
                background-color: #0e639c;
            }
            QGroupBox {
                border: 2px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
            QComboBox {
                background-color: #252526;
                border: 1px solid #555;
                padding: 5px;
            }
            QTreeWidget {
                background-color: #252526;
                border: 1px solid #555;
                color: #d4d4d4;
            }
            QTreeWidget::item:selected {
                background-color: #0e639c;
            }
            QTabWidget::pane {
                border: 1px solid #555;
            }
            QTabBar::tab {
                background-color: #252526;
                color: #d4d4d4;
                padding: 8px;
                border: 1px solid #555;
            }
            QTabBar::tab:selected {
                background-color: #0e639c;
            }
        """)

    def init_models(self):
        """Initialize SAM and CLIPSeg models."""
        self.status_label.setText("Loading models...")
        self.progress_bar.setRange(0, 0)

        QApplication.processEvents()

        try:
            self.status_label.setText("Loading SAM 2 model (this may take a minute)...")
            QApplication.processEvents()

            # Use SAM 2 with more points for better quality in demo
            self.sam_gen = build_sam_generator(
                model_size="large",  # Use large for best quality
                sam_device=device,
                points_per_side=32,  # Good coverage
                points_per_batch=64,  # Fast processing
            )

            self.status_label.setText("Loading CLIPSeg model...")
            QApplication.processEvents()
            clipseg_weights = "weights/rd64-uni.pth"
            self.clipseg_model = load_clipseg_model(clipseg_weights)

            self.status_label.setText("Models loaded successfully! Using SAM 2 for better segmentation.")
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(100)

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            self.status_label.setText(f"Error loading models: {str(e)}")
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            print(f"[ERROR] Failed to load models:\n{error_detail}")

    def load_image(self):
        """Load an image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )

        if file_path:
            self.image_path = file_path

            from PIL import Image
            img = Image.open(file_path).convert('RGB')
            img_np = np.array(img)

            self.original_label.set_image(img_np)

            self.status_label.setText(f"Loaded: {Path(file_path).name} ({img_np.shape[1]}x{img_np.shape[0]})")
            self.process_btn.setEnabled(True)

            self.clear_results()

    def clear_results(self):
        """Clear previous results."""
        placeholder = np.zeros((200, 200, 3), dtype=np.uint8)

        self.individual_masks_label.set_image(placeholder)
        self.selected_mask_label.set_image(placeholder)
        self.small_mask_label.set_image(placeholder)
        self.small_pca_label.set_image(placeholder)
        self.part_mask_label.set_image(placeholder)
        self.part_pca_label.set_image(placeholder)
        self.whole_mask_label.set_image(placeholder)
        self.whole_pca_label.set_image(placeholder)

        self.tree_widget.clear()
        self.stats_text.clear()
        self.current_result = None
        self.selected_node = None
        self.export_btn.setEnabled(False)

    def process_image(self):
        """Process the loaded image with SAM."""
        if not hasattr(self, 'image_path'):
            return

        self.process_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.progress_bar.setRange(0, 0)

        max_regions = int(self.max_regions_combo.currentText())
        self.worker = SAMProcessor(self.image_path, self.sam_gen, self.clipseg_model, max_regions)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.process_complete)
        self.worker.error.connect(self.process_error)
        self.worker.start()

    def update_progress(self, message: str):
        """Update progress display."""
        self.status_label.setText(message)

    def process_complete(self, result: Dict[str, Any]):
        """Handle processing completion."""
        self.current_result = result

        # Display results
        self.visualize_individual_masks(result)
        self.visualize_hierarchy(result)
        self.populate_tree(result['tree'])
        self.display_statistics(result['stats'])

        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.status_label.setText("Processing complete!")

        self.process_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.export_btn.setEnabled(True)

    def process_error(self, error_msg: str):
        """Handle processing error."""
        self.status_label.setText(error_msg)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self.process_btn.setEnabled(True)
        self.load_btn.setEnabled(True)

    def visualize_individual_masks(self, result: Dict[str, Any]):
        """Visualize individual masks colored by hierarchy level."""
        img = result['image']
        groups = result['groups']
        H, W = img.shape[:2]

        # Create colored mask overlay
        mask_img = img.copy().astype(np.float32) * 0.5  # Dim the original image

        # Color scheme: Whole=Blue, Part=Green, Small=Red
        colors = {
            'w': np.array([0, 0, 255], dtype=np.float32),  # Blue
            'p': np.array([0, 255, 0], dtype=np.float32),   # Green
            's': np.array([255, 0, 0], dtype=np.float32)    # Red
        }

        # Draw masks with alpha blending
        alpha = 0.4
        for key in ['w', 'p', 's']:
            for mask_data in groups[key]:
                seg = mask_data['segmentation']
                color = colors[key]
                mask_img[seg] = mask_img[seg] * (1 - alpha) + color * alpha

        self.individual_masks_label.set_image(mask_img.astype(np.uint8))

    def visualize_hierarchy(self, result: Dict[str, Any]):
        """Visualize hierarchy results (mask and PCA)."""
        targets = result['targets']

        for key, mask_label, pca_label in [
            ('s', self.small_mask_label, self.small_pca_label),
            ('p', self.part_mask_label, self.part_pca_label),
            ('w', self.whole_mask_label, self.whole_pca_label)
        ]:
            target = targets[key].numpy()
            H, W, D = target.shape

            # Create mask
            mask = np.linalg.norm(target, axis=-1) > 0.01
            mask_img = (mask * 255).astype(np.uint8)
            mask_img = np.stack([mask_img] * 3, axis=-1)
            mask_label.set_image(mask_img)

            # Create PCA visualization
            embeddings_flat = target.reshape(-1, D)
            valid_mask = np.linalg.norm(embeddings_flat, axis=1) > 0.01

            if valid_mask.sum() > 3:
                valid_embeddings = embeddings_flat[valid_mask]

                pca = PCA(n_components=3)
                pca_result = pca.fit_transform(valid_embeddings)

                pca_normalized = (pca_result - pca_result.min(axis=0)) / (
                    pca_result.max(axis=0) - pca_result.min(axis=0) + 1e-8
                )

                pca_img = np.zeros((H * W, 3))
                pca_img[valid_mask] = pca_normalized
                pca_img = pca_img.reshape(H, W, 3)
                pca_img = (pca_img * 255).astype(np.uint8)

                pca_label.set_image(pca_img)
            else:
                pca_label.set_image(np.zeros((H, W, 3), dtype=np.uint8))

    def populate_tree(self, tree: List[Dict]):
        """Populate the tree widget with hierarchy structure."""
        self.tree_widget.clear()

        def add_node_to_tree(node: Dict, parent_item: Optional[QTreeWidgetItem] = None):
            level = node['level']
            idx = node['idx']
            area = node['mask_data']['area']

            item = QTreeWidgetItem()
            item.setText(0, f"{level.capitalize()} #{idx}")
            item.setText(1, f"{area:,}")
            item.setText(2, level)

            # Color by level
            if level == 'whole':
                item.setForeground(0, QColor(100, 150, 255))
            elif level == 'part':
                item.setForeground(0, QColor(100, 255, 150))
            else:
                item.setForeground(0, QColor(255, 100, 100))

            # Store node data
            item.setData(0, Qt.UserRole, node)

            if parent_item:
                parent_item.addChild(item)
            else:
                self.tree_widget.addTopLevelItem(item)

            # Recursively add children
            for child in node['children']:
                add_node_to_tree(child, item)

            item.setExpanded(True)

        for root in tree:
            add_node_to_tree(root)

    def on_tree_item_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle tree item click - highlight selected mask and children."""
        node = item.data(0, Qt.UserRole)
        if node:
            self.selected_node = node
            self.visualize_selected_mask(node)

    def visualize_selected_mask(self, node: Dict):
        """Visualize the selected mask and its children."""
        img = self.current_result['image']
        H, W = img.shape[:2]

        # Create overlay
        mask_img = img.copy().astype(np.float32) * 0.5

        # Highlight selected mask
        seg = node['mask_data']['segmentation']
        color = {
            'whole': np.array([0, 0, 255], dtype=np.float32),
            'part': np.array([0, 255, 0], dtype=np.float32),
            'small': np.array([255, 0, 0], dtype=np.float32)
        }[node['level']]

        mask_img[seg] = mask_img[seg] * 0.3 + color * 0.7

        # Highlight children with different colors
        def highlight_children(node: Dict, depth: int = 1):
            for child in node['children']:
                child_seg = child['mask_data']['segmentation']
                child_color = {
                    'whole': np.array([0, 0, 255], dtype=np.float32),
                    'part': np.array([0, 255, 0], dtype=np.float32),
                    'small': np.array([255, 0, 0], dtype=np.float32)
                }[child['level']]

                # Slightly dimmer for children
                alpha = 0.5 * (0.8 ** depth)
                mask_img[child_seg] = mask_img[child_seg] * (1 - alpha) + child_color * alpha

                highlight_children(child, depth + 1)

        highlight_children(node)

        self.selected_mask_label.set_image(mask_img.astype(np.uint8))

    def reset_selection(self):
        """Reset mask selection."""
        self.selected_node = None
        if self.current_result:
            placeholder = np.zeros_like(self.current_result['image'])
            self.selected_mask_label.set_image(placeholder)

    def display_statistics(self, stats: Dict[str, Any]):
        """Display statistics in text widget."""
        text = "=" * 70 + "\n"
        text += "SAM HIERARCHY STATISTICS\n"
        text += "=" * 70 + "\n\n"

        text += "VISUALIZATION GUIDE:\n"
        text += "-" * 70 + "\n"
        text += "Individual Masks (Left):\n"
        text += "  - Shows ALL masks colored by hierarchy level\n"
        text += "  - Blue = Whole regions (largest)\n"
        text += "  - Green = Part regions (medium)\n"
        text += "  - Red = Small regions (smallest)\n\n"
        text += "Hierarchy Views (Right Tab):\n"
        text += "  - Mask: Binary mask showing active pixels\n"
        text += "  - PCA: 512D embeddings reduced to RGB\n"
        text += "    (Different colors = semantically different regions)\n\n"
        text += "=" * 70 + "\n\n"

        for key in ['s', 'p', 'w']:
            s = stats[key]
            text += f"{s['name']} Regions:\n"
            text += f"  Number of masks: {s['num_masks']}\n"
            text += f"  Total pixels: {s['total_pixels']:,}\n"
            text += f"  Coverage: {s['coverage_pct']:.2f}%\n"

            if s['num_masks'] > 0:
                text += f"  Avg area: {s['avg_area']:.1f}\n"
                text += f"  Area range: {s['min_area']} - {s['max_area']}\n"

            text += "\n"

        text += "=" * 70 + "\n"
        text += "Tree Structure:\n"
        text += "=" * 70 + "\n\n"
        text += f"Number of root nodes: {stats['tree']['num_roots']}\n"
        text += f"Total nodes in tree: {stats['tree']['total_nodes']}\n\n"
        text += "The tree shows parent-child containment relationships:\n"
        text += "  Whole -> Part -> Small\n"
        text += "Click on any node to highlight it and its children!\n"

        self.stats_text.setText(text)

    def export_results(self):
        """Export visualization results."""
        if not self.current_result:
            return

        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")

        if output_dir:
            try:
                output_dir = Path(output_dir)

                # Save images
                from PIL import Image

                # Original
                img = Image.fromarray(self.current_result['image'])
                img.save(output_dir / "original.png")

                # Individual masks
                individual_img = self.individual_masks_label.pixmap()
                if individual_img:
                    individual_img.save(str(output_dir / "individual_masks.png"))

                # Hierarchy visualizations
                for key, name in [('s', 'small'), ('p', 'part'), ('w', 'whole')]:
                    target = self.current_result['targets'][key].numpy()
                    H, W, D = target.shape

                    mask = np.linalg.norm(target, axis=-1) > 0.01
                    mask_img = (mask * 255).astype(np.uint8)
                    Image.fromarray(mask_img, mode='L').save(output_dir / f"{name}_mask.png")

                    embeddings_flat = target.reshape(-1, D)
                    valid_mask = np.linalg.norm(embeddings_flat, axis=1) > 0.01

                    if valid_mask.sum() > 3:
                        valid_embeddings = embeddings_flat[valid_mask]
                        pca = PCA(n_components=3)
                        pca_result = pca.fit_transform(valid_embeddings)
                        pca_normalized = (pca_result - pca_result.min(axis=0)) / (
                            pca_result.max(axis=0) - pca_result.min(axis=0) + 1e-8
                        )

                        pca_img = np.zeros((H * W, 3))
                        pca_img[valid_mask] = pca_normalized
                        pca_img = pca_img.reshape(H, W, 3)
                        pca_img = (pca_img * 255).astype(np.uint8)

                        Image.fromarray(pca_img).save(output_dir / f"{name}_pca.png")

                # Save statistics
                with open(output_dir / "statistics.txt", 'w') as f:
                    f.write(self.stats_text.toPlainText())

                # Save tree structure
                with open(output_dir / "tree_structure.txt", 'w') as f:
                    self._write_tree_structure(f, self.current_result['tree'])

                self.status_label.setText(f"Results exported to {output_dir}")

            except Exception as e:
                self.status_label.setText(f"Export failed: {str(e)}")

    def _write_tree_structure(self, f, tree: List[Dict], indent: int = 0):
        """Write tree structure to file."""
        for node in tree:
            level = node['level']
            idx = node['idx']
            area = node['mask_data']['area']
            f.write("  " * indent + f"{level.capitalize()} #{idx} (area={area})\n")
            self._write_tree_structure(f, node['children'], indent + 1)


def main():
    app = QApplication(sys.argv)
    window = SAMDemoWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
