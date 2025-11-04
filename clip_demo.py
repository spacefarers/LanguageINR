#!/usr/bin/env python3
"""
CLIPSeg Text-Image Similarity Demo
==================================

Pick an image, type a prompt, and visualize a dense per-pixel correlation map
computed with CLIPSeg between the prompt embedding and the image. The overlay
highlights regions that respond strongly to the entered phrase.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from PIL import Image, ImageTk
except ImportError as exc:  # pragma: no cover - GUI dependency guard
    print("Error: GUI dependencies not found:", exc)
    print("Install with: pip install pillow")
    sys.exit(1)

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
except ImportError as exc:  # pragma: no cover - runtime guard
    print("Error: required package not found:", exc)
    print("Install dependencies from requirements.txt and try again.")
    raise


class CLIPSegDemoGUI:
    """Interactive CLIPSeg visualization GUI."""

    def __init__(self, root: tk.Tk, model_name: str, device: str):
        self.root = root
        self.root.title("CLIPSeg Text-Image Demo")
        self.root.geometry("960x780")
        self.root.resizable(True, True)

        self.model_name = model_name
        self.device = device

        self.clipseg_model: Optional[CLIPSegForImageSegmentation] = None
        self.clipseg_processor: Optional[CLIPSegProcessor] = None

        self.selected_image_path: Optional[Path] = None
        self.original_image: Optional[Image.Image] = None
        self.display_image: Optional[Image.Image] = None
        self.photo_image: Optional[ImageTk.PhotoImage] = None
        self.last_heatmap: Optional[np.ndarray] = None
        self.image_inputs: Optional[Dict[str, torch.Tensor]] = None

        self._setup_ui()
        self._load_clipseg_model()

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
        self.prompt_var = tk.StringVar(value="highlight the tree trunk")
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

        # Status / similarity info
        info_frame = ttk.Frame(control_frame)
        info_frame.pack(fill=tk.X, pady=(10, 0))

        self.status_label = ttk.Label(info_frame, text="Loading CLIPSeg...", foreground="blue")
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
    def _load_clipseg_model(self) -> None:
        """Load CLIPSeg weights and processor."""
        try:
            self.status_label.config(
                text=f"Loading CLIPSeg model: {self.model_name}",
                foreground="blue",
            )
            self.root.update_idletasks()

            self.clipseg_processor = CLIPSegProcessor.from_pretrained(self.model_name)
            self.clipseg_model = CLIPSegForImageSegmentation.from_pretrained(self.model_name)
            self.clipseg_model.to(self.device)
            self.clipseg_model.eval()

            self.status_label.config(
                text=f"Model ready on {self.device}. Select an image to start.",
                foreground="green",
            )
        except Exception as exc:  # pragma: no cover - runtime guard
            self.status_label.config(text=f"Failed to load CLIPSeg: {exc}", foreground="red")
            messagebox.showerror("Model Error", f"Could not load CLIPSeg model:\n{exc}")
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
        self.image_inputs = None

        if self.clipseg_processor:
            try:
                self.status_label.config(text="Processing image features...", foreground="blue")
                self.root.update_idletasks()

                img_inputs = self.clipseg_processor(images=self.original_image, return_tensors="pt")
                self.image_inputs = {k: v.to(self.device) for k, v in img_inputs.items()}

                self.status_label.config(
                    text="Image loaded. Enter a prompt and click Visualize.",
                    foreground="blue",
                )
            except Exception as exc:
                self.status_label.config(text=f"Failed to process image: {exc}", foreground="red")
                messagebox.showerror("Image Error", f"Failed to process image:\n{exc}")
                self.image_inputs = None
        else:  # pragma: no cover - defensive
            self.image_inputs = None

        self.image_label.config(text=str(self.selected_image_path))
        self.mean_label.config(text="Mean correlation: —")
        self.max_label.config(text="Max correlation: —")

        self._display_image(self.display_image)

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
        """Run CLIPSeg on the current image and overlay similarity heatmap."""
        if self.clipseg_model is None or self.clipseg_processor is None:
            messagebox.showerror("Model Error", "CLIPSeg model is not ready.")
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
            heatmap, mean_score, max_score = self._compute_clipseg_heatmap(prompt)
        except Exception as exc:
            self.status_label.config(text=f"Error during computation: {exc}", foreground="red")
            messagebox.showerror("Computation Error", f"Failed to compute similarity:\n{exc}")
            return

        self.last_heatmap = heatmap
        self.mean_label.config(text=f"Mean correlation: {mean_score:.4f}")
        self.max_label.config(text=f"Max correlation: {max_score:.4f}")

        overlay = self._apply_overlay(self.original_image, heatmap)
        self.display_image = overlay
        self._display_image(self.display_image)

        self.status_label.config(text="Done. Adjust prompt or overlay strength and try again.", foreground="green")

    def _compute_clipseg_heatmap(self, prompt: str) -> Tuple[np.ndarray, float, float]:
        """Compute a dense CLIPSeg similarity map for the given prompt."""
        if self.clipseg_processor is None or self.clipseg_model is None:  # pragma: no cover - defensive
            raise RuntimeError("CLIPSeg model is not loaded.")

        if self.image_inputs is None or self.original_image is None:
            raise RuntimeError("Image is not loaded or processed.")

        text_inputs = self.clipseg_processor(text=[prompt], return_tensors="pt")
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        inputs = {**self.image_inputs, **text_inputs}

        with torch.no_grad():
            outputs = self.clipseg_model(**inputs)

        logits = outputs.logits
        probs = torch.sigmoid(logits)

        if probs.ndim == 3:
            probs = probs.unsqueeze(1)
        elif probs.ndim != 4:  # pragma: no cover - defensive
            raise ValueError(f"Unexpected logits shape from CLIPSeg: {tuple(probs.shape)}")

        # Resize to the original image resolution
        probs = F.interpolate(
            probs,
            size=(self.original_image.height, self.original_image.width),
            mode="bilinear",
            align_corners=False,
        )

        prob_map = probs[0, 0].cpu().numpy()
        prob_map = np.clip(prob_map, 0.0, 1.0)

        mean_score = float(prob_map.mean())
        max_score = float(prob_map.max())

        # Normalize for visualization while keeping correlations meaningful
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
        heat = np.clip(heatmap, 0.0, 1.0)

        # Simple blue→red gradient for visualization
        heat_rgb = np.zeros_like(base)
        heat_rgb[..., 0] = heat  # red channel
        heat_rgb[..., 1] = heat * 0.6  # green channel
        heat_rgb[..., 2] = (1.0 - heat) * 0.4  # blue channel fades with heat

        alpha = (heat[..., None] * strength).astype(np.float32)
        blended = base * (1.0 - alpha) + heat_rgb * alpha
        blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(blended)


def main() -> None:
    parser = argparse.ArgumentParser(description="CLIPSeg Text-Image Similarity Demo GUI")
    parser.add_argument(
        "--model",
        default="CIDAS/clipseg-rd64-refined",
        help="CLIPSeg model name (default: CIDAS/clipseg-rd64-refined)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: auto-detect)",
    )
    args = parser.parse_args()

    root = tk.Tk()
    gui = CLIPSegDemoGUI(root, model_name=args.model, device=args.device)

    button_frame = ttk.Frame(root, padding=(10, 0, 10, 10))
    button_frame.pack(fill=tk.X)
    ttk.Button(button_frame, text="Clear Image", command=lambda: _clear_image(gui)).pack(side=tk.LEFT)
    ttk.Button(button_frame, text="Exit", command=root.quit).pack(side=tk.RIGHT)

    root.mainloop()


def _clear_image(gui: CLIPSegDemoGUI) -> None:
    """Clear current image and reset display."""
    gui.selected_image_path = None
    gui.original_image = None
    gui.display_image = None
    gui.photo_image = None
    gui.last_heatmap = None
    gui.image_inputs = None
    gui.image_label.config(text="No image selected")
    gui.mean_label.config(text="Mean correlation: —")
    gui.max_label.config(text="Max correlation: —")
    gui.status_label.config(text="Select an image to begin.", foreground="blue")
    gui.canvas.delete("all")


if __name__ == "__main__":
    main()
