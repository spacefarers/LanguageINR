#!/usr/bin/env python3
"""
LSeg Text-Image Similarity Demo
================================

Pick an image, type a prompt, and visualize a dense per-pixel correlation map
computed with LSeg between the prompt embedding and the image. The overlay
highlights regions that respond strongly to the entered phrase.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from PIL import Image, ImageTk
except ImportError as exc:
    print("Error: GUI dependencies not found:", exc)
    print("Install with: pip install pillow")
    sys.exit(1)

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
except ImportError as exc:
    print("Error: required package not found:", exc)
    print("Install dependencies from requirements.txt and try again.")
    raise

# Import LSeg module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "encoders", "lseg_encoder"))
from modules.lseg_module import LSegModule


class LSegDemoGUI:
    """Interactive LSeg visualization GUI."""

    def __init__(self, root: tk.Tk, checkpoint_path: str, device: str):
        self.root = root
        self.root.title("LSeg Text-Image Demo")
        self.root.geometry("960x780")
        self.root.resizable(True, True)

        self.checkpoint_path = checkpoint_path
        self.device = device

        self.lseg_model: Optional[LSegModule] = None

        self.selected_image_path: Optional[Path] = None
        self.original_image: Optional[Image.Image] = None
        self.display_image: Optional[Image.Image] = None
        self.photo_image: Optional[ImageTk.PhotoImage] = None
        self.last_heatmap: Optional[np.ndarray] = None

        self.overlay_only_var = tk.BooleanVar(value=False)
        self._setup_ui()
        self._load_lseg_model()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Control panel -------------------------------------------------
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # Image selection row
        image_row = ttk.Frame(control_frame)
        image_row.pack(fill=tk.X, pady=5)
        ttk.Button(image_row, text="Select Image", command=self._select_image).pack(side=tk.LEFT)
        self.image_label = ttk.Label(image_row, text="No image selected", width=60)
        self.image_label.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        # Prompt row
        prompt_row = ttk.Frame(control_frame)
        prompt_row.pack(fill=tk.X, pady=5)
        ttk.Label(prompt_row, text="Prompt:").pack(side=tk.LEFT)
        self.prompt_var = tk.StringVar(value="tree")
        prompt_entry = ttk.Entry(prompt_row, textvariable=self.prompt_var)
        prompt_entry.pack(side=tk.LEFT, padx=(5, 10), fill=tk.X, expand=True)
        prompt_entry.bind("<Return>", lambda _event: self._visualize())
        ttk.Button(prompt_row, text="Visualize", command=self._visualize).pack(side=tk.LEFT)

        # Overlay strength row
        overlay_row = ttk.Frame(control_frame)
        overlay_row.pack(fill=tk.X, pady=5)
        ttk.Label(overlay_row, text="Overlay Strength:").pack(side=tk.LEFT)
        self.overlay_var = tk.DoubleVar(value=0.6)
        overlay_scale = ttk.Scale(
            overlay_row,
            from_=0.1,
            to=1.0,
            variable=self.overlay_var,
            orient=tk.HORIZONTAL,
        )
        overlay_scale.pack(side=tk.LEFT, padx=(5, 0), fill=tk.X, expand=True)
        overlay_scale.configure(command=lambda _value: self._on_overlay_strength_change())

        self.overlay_only_checkbox = ttk.Checkbutton(
            overlay_row,
            text="Hide original image",
            variable=self.overlay_only_var,
            command=self._update_display,
        )
        self.overlay_only_checkbox.pack(side=tk.LEFT, padx=(10, 0))

        # Status / similarity info
        info_frame = ttk.Frame(control_frame)
        info_frame.pack(fill=tk.X, pady=(10, 0))

        self.status_label = ttk.Label(info_frame, text="Loading LSeg...", foreground="blue")
        self.status_label.pack(fill=tk.X)

        self.mean_label = ttk.Label(info_frame, text="Mean correlation: —")
        self.mean_label.pack(fill=tk.X)

        self.max_label = ttk.Label(info_frame, text="Max correlation: —")
        self.max_label.pack(fill=tk.X)

        # Canvas for image display -------------------------------------
        image_frame = ttk.LabelFrame(main_frame, text="Visualization", padding=5)
        image_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(image_frame, bg="gray15")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.config(height=540)

    # ------------------------------------------------------------------
    # Model handling
    # ------------------------------------------------------------------
    def _load_lseg_model(self) -> None:
        """Load LSeg weights and processor."""
        try:
            self.status_label.config(
                text=f"Loading LSeg model from: {self.checkpoint_path}",
                foreground="blue",
            )
            self.root.update_idletasks()

            # Load LSeg module from checkpoint
            self.lseg_model = LSegModule.load_from_checkpoint(
                checkpoint_path=self.checkpoint_path,
                data_path=None,
                dataset='ignore',
                backbone='clip_vitl16_384',
                aux=False,
                num_features=256,
                aux_weight=0,
                se_loss=False,
                se_weight=0,
                base_lr=0,
                batch_size=1,
                max_epochs=0,
                ignore_index=-1,
                dropout=0.0,
                scale_inv=True,
                augment=False,
                no_batchnorm=False,
                widehead=False,
                widehead_hr=False,
                map_locatin="cpu",
                arch_option=0,
                strict=False,
                block_depth=0,
                activation='lrelu',
            )
            
            self.lseg_model.to(self.device)
            self.lseg_model.eval()

            self.status_label.config(
                text=f"Model ready on {self.device}. Select an image to start.",
                foreground="green",
            )
        except Exception as exc:
            self.status_label.config(text=f"Failed to load LSeg: {exc}", foreground="red")
            messagebox.showerror("Model Error", f"Could not load LSeg model:\n{exc}")
            raise

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------
    def _select_image(self) -> None:
        """Prompt for an image file and display it."""
        filetypes = [("Image files", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All files", "*.*")]
        filename = filedialog.askopenfilename(title="Select Image", filetypes=filetypes, initialdir=os.getcwd())
        if not filename:
            return

        try:
            image = Image.open(filename).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Image Error", f"Failed to load image:\n{exc}")
            return

        self.selected_image_path = Path(filename)
        self.original_image = image
        self.display_image = image.copy()
        self.last_heatmap = None

        self.image_label.config(text=str(self.selected_image_path))
        self.status_label.config(text="Image loaded. Enter a prompt and click Visualize.", foreground="blue")
        self.mean_label.config(text="Mean correlation: —")
        self.max_label.config(text="Max correlation: —")

        self._update_display()

    def _display_image(self, image: Image.Image) -> None:
        """Render an image on the canvas, preserving aspect ratio."""
        canvas_width = max(self.canvas.winfo_width(), 10)
        canvas_height = max(self.canvas.winfo_height(), 10)

        img_w, img_h = image.size
        img_aspect = img_w / img_h
        canvas_aspect = canvas_width / canvas_height

        if img_aspect > canvas_aspect:
            new_w = canvas_width
            new_h = int(canvas_width / img_aspect)
        else:
            new_h = canvas_height
            new_w = int(canvas_height * img_aspect)

        resized = image.resize((max(new_w, 1), max(new_h, 1)), Image.Resampling.LANCZOS)
        self.photo_image = ImageTk.PhotoImage(resized)

        self.canvas.delete("all")
        self.canvas.create_image(canvas_width // 2, canvas_height // 2, image=self.photo_image)

    # ------------------------------------------------------------------
    # Visualization logic
    # ------------------------------------------------------------------
    def _visualize(self) -> None:
        """Run LSeg on the current image and overlay similarity heatmap."""
        if self.lseg_model is None:
            messagebox.showerror("Model Error", "LSeg model is not ready.")
            return

        if self.original_image is None:
            messagebox.showwarning("No Image", "Select an image first.")
            return

        prompt = self.prompt_var.get().strip()
        if not prompt:
            messagebox.showwarning("Missing Prompt", "Enter a prompt to compare.")
            return

        self.status_label.config(text=f"Analyzing similarity for '{prompt}'...", foreground="blue")
        self.root.update_idletasks()

        try:
            heatmap, mean_score, max_score = self._compute_lseg_heatmap(self.original_image, prompt)
        except Exception as exc:
            self.status_label.config(text=f"Error during computation: {exc}", foreground="red")
            messagebox.showerror("Computation Error", f"Failed to compute similarity:\n{exc}")
            return

        self.last_heatmap = heatmap
        self.mean_label.config(text=f"Mean correlation: {mean_score:.4f}")
        self.max_label.config(text=f"Max correlation: {max_score:.4f}")

        self._update_display()

        self.status_label.config(text="Done. Adjust prompt or overlay strength and try again.", foreground="green")

    def _compute_lseg_heatmap(self, image: Image.Image, prompt: str) -> Tuple[np.ndarray, float, float]:
        """Compute a dense LSeg similarity map for the given prompt."""
        if self.lseg_model is None:
            raise RuntimeError("LSeg model is not loaded.")

        # Prepare image using LSeg's transform
        input_transform = self.lseg_model.val_transform
        img_tensor = input_transform(image).unsqueeze(0).to(self.device)

        # Get text labels - LSeg expects labels as a list
        labels = [prompt]
        
        with torch.no_grad():
            # Forward through model to get per-pixel features
            features = self.lseg_model.net.forward(img_tensor, labelset=labels, return_feature=True)
            
            # Get text embeddings
            import clip
            text_tokens = clip.tokenize(labels).to(self.device)
            text_features = self.lseg_model.net.clip_pretrained.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Ensure consistent dtype (convert both to float32)
            features = features.float()
            text_features = text_features.float()
            
            # Compute similarity between image features and text
            # features shape: [B, C, H, W], text_features: [num_labels, C]
            features_flat = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
            features_flat = features_flat / features_flat.norm(dim=-1, keepdim=True)
            
            similarity = features_flat @ text_features.t()
            similarity = similarity.reshape(features.shape[0], features.shape[2], features.shape[3], -1)
            similarity = similarity[0, :, :, 0]  # Get first image, first label

        # Resize to original image size
        similarity_resized = F.interpolate(
            similarity.unsqueeze(0).unsqueeze(0),
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=False,
        )
        
        prob_map = similarity_resized[0, 0].cpu().numpy()
        
        mean_score = float(prob_map.mean())
        max_score = float(prob_map.max())

        # Normalize for visualization
        if prob_map.max() > prob_map.min():
            heatmap = (prob_map - prob_map.min()) / (prob_map.max() - prob_map.min())
        else:
            heatmap = np.zeros_like(prob_map)

        return heatmap, mean_score, max_score

    def _apply_overlay(self, image: Image.Image, heatmap: np.ndarray) -> Image.Image:
        """Blend a colored heatmap onto the image."""
        strength = float(self.overlay_var.get())
        strength = max(0.0, min(1.0, strength))

        base = np.array(image).astype(np.float32) / 255.0
        heat_rgb = self._heatmap_to_rgb(heatmap, base.shape)
        heat = np.clip(heatmap, 0.0, 1.0)

        alpha = (heat[..., None] * strength).astype(np.float32)
        blended = base * (1.0 - alpha) + heat_rgb * alpha
        blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(blended)

    def _render_overlay_only(self, heatmap: np.ndarray) -> Image.Image:
        """Render the overlay without the original image."""
        overlay = self._heatmap_to_rgb(heatmap)
        overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(overlay)

    def _heatmap_to_rgb(self, heatmap: np.ndarray, reference_shape: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        """Convert a normalized heatmap to an RGB array."""
        heat = np.clip(heatmap, 0.0, 1.0)
        if reference_shape is None:
            shape = heat.shape + (3,)
        else:
            shape = reference_shape
        heat_rgb = np.zeros(shape, dtype=np.float32)
        heat_rgb[..., 0] = heat  # red channel
        heat_rgb[..., 1] = heat * 0.6  # green channel
        heat_rgb[..., 2] = (1.0 - heat) * 0.4  # blue channel fades with heat
        return heat_rgb

    def _update_display(self) -> None:
        """Update canvas based on current overlay settings."""
        if self.original_image is None:
            self.canvas.delete("all")
            self.display_image = None
            self.photo_image = None
            return

        if self.overlay_only_var.get() and self.last_heatmap is not None:
            image_to_show = self._render_overlay_only(self.last_heatmap)
        elif self.last_heatmap is not None:
            image_to_show = self._apply_overlay(self.original_image, self.last_heatmap)
        else:
            image_to_show = self.original_image

        self.display_image = image_to_show
        self._display_image(image_to_show)

    def _on_overlay_strength_change(self) -> None:
        """Refresh display when overlay strength slider moves."""
        if self.last_heatmap is not None and not self.overlay_only_var.get():
            self._update_display()


def main() -> None:
    parser = argparse.ArgumentParser(description="LSeg Text-Image Similarity Demo GUI")
    parser.add_argument(
        "--checkpoint",
        default="encoders/lseg_encoder/demo_e200.ckpt",
        help="LSeg checkpoint path (default: encoders/lseg_encoder/demo_e200.ckpt)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: auto-detect)",
    )
    args = parser.parse_args()

    root = tk.Tk()
    gui = LSegDemoGUI(root, checkpoint_path=args.checkpoint, device=args.device)

    button_frame = ttk.Frame(root, padding=(10, 0, 10, 10))
    button_frame.pack(fill=tk.X)
    ttk.Button(button_frame, text="Clear Image", command=lambda: _clear_image(gui)).pack(side=tk.LEFT)
    ttk.Button(button_frame, text="Exit", command=root.quit).pack(side=tk.RIGHT)

    root.mainloop()


def _clear_image(gui: LSegDemoGUI) -> None:
    """Clear current image and reset display."""
    gui.selected_image_path = None
    gui.original_image = None
    gui.display_image = None
    gui.photo_image = None
    gui.last_heatmap = None
    gui.image_label.config(text="No image selected")
    gui.mean_label.config(text="Mean correlation: —")
    gui.max_label.config(text="Max correlation: —")
    gui.status_label.config(text="Select an image to begin.", foreground="blue")
    gui.canvas.delete("all")


if __name__ == "__main__":
    main()
