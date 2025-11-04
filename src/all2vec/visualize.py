"""
ALL2Vec: Real-time Visualization

2D PCA projection and attractor dynamics visualization.

Reference:
Ken I. (2025). ALL2Vec: Continuous Predictive Representations 
for Dynamic Visual Streams. Preprint.
https://doi.org/10.5281/zenodo.17513405

"""

from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from typing import Tuple, Optional

from .model import ModelConfig


class Visualizer:
    """Matplotlib-based dashboard for USF evolution and webcam preview."""

    def __init__(self, config: ModelConfig, pca) -> None:
        self.config = config
        self.pca = pca

        plt.ion()
        self.fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(4, 2, width_ratios=[2.0, 1.2], height_ratios=[2.0, 1.0, 1.0, 1.0])

        self.ax_sc = self.fig.add_subplot(gs[:, 0])
        self.ax_cam = self.fig.add_subplot(gs[0, 1])
        self.ax_dyn = self.fig.add_subplot(gs[1, 1])
        self.ax_rec = self.fig.add_subplot(gs[2, 1])
        self.ax_stb = self.fig.add_subplot(gs[3, 1])

        self.sc = self.ax_sc.scatter([], [], s=12, alpha=0.9)
        self.ax_sc.set_title(f"USF ({config.B} particles)")
        self.ax_sc.set_xlabel("PC1")
        self.ax_sc.set_ylabel("PC2")
        self.ax_sc.grid(True, alpha=0.25)

        self.ax_cam.set_title("Webcam")
        self.ax_cam.axis("off")
        self._cam_artist = self.ax_cam.imshow(np.zeros((480, 640, 3), dtype=np.uint8))

        self.ts_dyn, self.ts_rec, self.ts_stb = [], [], []
        (self.line_dyn,) = self.ax_dyn.plot([], [], label="Pred")
        (self.line_rec,) = self.ax_rec.plot([], [], label="Recon", color="tab:orange")
        (self.line_stb,) = self.ax_stb.plot([], [], label="Stability", color="tab:green")

        for ax, label in zip(
            (self.ax_dyn, self.ax_rec, self.ax_stb),
            ("Future pred loss", "Reconstruction loss", "Stability loss"),
        ):
            ax.grid(True, alpha=0.3)
            ax.set_ylabel(label)
            ax.legend(loc="upper right")

        self.ax_stb.set_xlabel("Step")

        self.fps_text = self.ax_sc.text(
            0.02,
            0.95,
            "FPS: --",
            transform=self.ax_sc.transAxes,
            fontsize=10,
            color="black",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
        )

        self.running = True
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

    def _on_key(self, event) -> None:  # pragma: no cover - UI callback
        if event.key == "q":
            self._shutdown()

    def _on_close(self, _event) -> None:  # pragma: no cover - UI callback
        self._shutdown()

    def _shutdown(self) -> None:
        if self.running:
            self.running = False
            try:
                plt.close(self.fig)
            except Exception:  # pragma: no cover - defensive
                pass

    def update(
        self,
        S_cpu: np.ndarray,
        frame_bgr: np.ndarray,
        metrics: Tuple[float, float, float],
        step: int,
        fps: Optional[float] = None,
    ) -> None:
        if not self.running:
            return

        XY = self.pca.transform(S_cpu)
        self.sc.set_offsets(XY)
        self.ax_sc.set_xlim(-0.3, 0.3)
        self.ax_sc.set_ylim(-0.3, 0.3)
        self.ax_sc.set_aspect("equal")
        self.ax_sc.set_title(f"USF ({self.config.B} particles) — step {step}")

        if fps is not None:
            self.fps_text.set_text(f"FPS: {fps:.1f}")

        dyn, rec, stb = metrics
        self.ts_dyn.append(dyn)
        self.ts_rec.append(rec)
        self.ts_stb.append(stb)

        xs = np.arange(len(self.ts_dyn))
        self.line_dyn.set_data(xs, self.ts_dyn)
        self.line_rec.set_data(xs, self.ts_rec)
        self.line_stb.set_data(xs, self.ts_stb)

        for ax, series in zip(
            (self.ax_dyn, self.ax_rec, self.ax_stb),
            (self.ts_dyn, self.ts_rec, self.ts_stb),
        ):
            if len(series) < 3:
                ax.set_xlim(0, 30)
                ax.set_ylim(0, 1.0)
            else:
                ax.set_xlim(0, max(30, len(series)))
                ymax = np.percentile(series, 95) if np.isfinite(series).all() else 1.0
                ax.set_ylim(0, float(max(1e-6, ymax)) * 1.2)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._cam_artist.set_data(frame_rgb)
        self.ax_cam.set_xlim(0, frame_rgb.shape[1])
        self.ax_cam.set_ylim(frame_rgb.shape[0], 0)

        plt.pause(0.001)


__all__ = ["Visualizer"]

