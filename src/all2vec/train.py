"""
ALL2Vec: Training Loop

Online learning implementation for continuous visual stream processing.

Reference:
Ken I. (2025). ALL2Vec: Continuous Predictive Representations 
for Dynamic Visual Streams. Preprint.
https://doi.org/10.5281/zenodo.17513405

"""


from __future__ import annotations

import os
import time
import pickle
from collections import defaultdict
from typing import Optional, Tuple
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Allow running as a script (`python src/all2vec/train.py`) as well as a module (
# `python -m all2vec.train`). When executed as a script, __package__ is None and
# relative imports would fail; we add the repository root to sys.path in that case.
if __package__ is None:  # pragma: no cover - import shim
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from all2vec.model import ModelConfig, FeatureExtractor, ALL2VecModel, init_state
    from all2vec.visualize import Visualizer
else:
    from .model import ModelConfig, FeatureExtractor, ALL2VecModel, init_state
    from .visualize import Visualizer


def open_camera(config: ModelConfig) -> cv2.VideoCapture:
    tried = set()
    candidates = list(config.preferred_cameras) + list(config.fallback_cameras)
    for idx in candidates:
        if idx in tried:
            continue
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"[INFO] Camera {idx} selected")
            return cap
        tried.add(idx)
        cap.release()
    raise RuntimeError("No webcam available")


def run_live(config: Optional[ModelConfig] = None) -> None:
    cfg = config or ModelConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    print(f"[INFO] device={device}, dtype={dtype}, B={cfg.B}")

    os.makedirs(cfg.log_dir, exist_ok=True)

    cap = open_camera(cfg)
    extractor = FeatureExtractor(cfg, device, dtype)
    model = ALL2VecModel(cfg.DU).to(device=device, dtype=dtype)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-5)

    S = init_state(cfg, device, dtype)

    pca = PCA(n_components=2, random_state=cfg.pca_seed)
    with torch.no_grad():
        ref = torch.randn(cfg.pca_ref, cfg.DU)
        ref = ref / (ref.norm(dim=-1, keepdim=True) + 1e-6)
        pca.fit(ref.numpy())

    visualizer = Visualizer(cfg, pca)

    LOG = defaultdict(list)
    step = 0
    buffer_prev = None
    prev_time = time.perf_counter()
    fps_smooth: Optional[float] = None

    try:
        while visualizer.running:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] No frame from webcam, stopping")
                break

            U_batch = extractor(frame).to(device=device, dtype=dtype)

            S_prev = S
            S_hat, recon_mean, U_mean = model.forward_step(S_prev, U_batch)
            U_repeat = U_mean.repeat(cfg.B, 1)

            with torch.no_grad():
                S_target = (1 - cfg.eta_s) * S_prev + cfg.eta_s * S_hat.detach() + cfg.u_inj * U_repeat
                S_target = S_target / (S_target.norm(dim=-1, keepdim=True) + cfg.eps)

            S_prev_c = S_prev.clamp(-cfg.clamp_val, cfg.clamp_val)
            S_hat_c = S_hat.clamp(-cfg.clamp_val, cfg.clamp_val)
            stability_loss = F.mse_loss(S_hat_c, S_prev_c)
            smooth_loss = F.mse_loss(S_hat, S_prev)

            recon_c = recon_mean.clamp(-cfg.clamp_val, cfg.clamp_val)
            target_c = U_batch.clamp(-cfg.clamp_val, cfg.clamp_val)
            recon_loss = F.mse_loss(recon_c, target_c)

            future_pred_loss = torch.tensor(0.0, device=device, dtype=dtype)
            if buffer_prev is not None:
                S_tm1, Umean_tm1 = buffer_prev
                U_tm1 = Umean_tm1.repeat(cfg.B, 1)
                S_pred = model.predict(S_tm1, U_tm1)
                S_target_detach = S_target.detach()
                future_pred_loss = F.mse_loss(
                    S_pred.clamp(-cfg.clamp_val, cfg.clamp_val),
                    S_target_detach.clamp(-cfg.clamp_val, cfg.clamp_val),
                )

            loss = (
                cfg.dyn_lambda * future_pred_loss
                + cfg.recon_lambda * recon_loss
                + cfg.stability_lambda * stability_loss
                + cfg.smooth_lambda * smooth_loss
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            with torch.no_grad():
                noise = cfg.noise_std * torch.randn_like(S_target)
                S = S_target + noise
                S = S / (S.norm(dim=-1, keepdim=True) + cfg.eps)

            buffer_prev = (S_prev.detach(), U_mean.detach())

            if step % cfg.save_interval == 0:
                with torch.no_grad():
                    S_np = S.detach().cpu().numpy()
                    LOG['states'].append(S_np.copy())
                    LOG['states_pca'].append(pca.transform(S_np))
                    LOG['step'].append(step)
                    LOG['future_pred_loss'].append(float(future_pred_loss.cpu()))
                    LOG['recon_loss'].append(float(recon_loss.cpu()))
                    LOG['stability_loss'].append(float(stability_loss.cpu()))
                    S_mean = S.mean(dim=0, keepdim=True)
                    dispersion = ((S - S_mean) ** 2).sum(dim=1).mean()
                    LOG['dispersion'].append(float(dispersion.cpu()))
                    if len(LOG['states']) > 1:
                        S_prev_saved = torch.from_numpy(LOG['states'][-2]).to(device)
                        temporal = ((S - S_prev_saved) ** 2).sum(dim=1).mean()
                        LOG['temporal_stability'].append(float(temporal.cpu()))
                    else:
                        LOG['temporal_stability'].append(0.0)

            if step % (cfg.save_interval * 10) == 0 and step > 0:
                path = os.path.join(cfg.log_dir, f"log_step_{step:06d}.pkl")
                with open(path, "wb") as f:
                    pickle.dump(dict(LOG), f)
                print(f"[LOG] Saved {path}")

            if step % cfg.frame_skip == 0:
                now = time.perf_counter()
                dt = now - prev_time
                prev_time = now
                if dt > 0:
                    inst_fps = 1.0 / dt
                    fps_smooth = (
                        inst_fps if fps_smooth is None else 0.9 * fps_smooth + 0.1 * inst_fps
                    )

                S_cpu = S.detach().float().cpu().numpy()
                visualizer.update(
                    S_cpu,
                    frame,
                    (
                        float(future_pred_loss.cpu()),
                        float(recon_loss.cpu()),
                        float(stability_loss.cpu()),
                    ),
                    step,
                    fps=fps_smooth,
                )

            step += 1

    finally:
        final_log_path = os.path.join(cfg.log_dir, "log_final.pkl")
        with open(final_log_path, "wb") as f:
            pickle.dump(dict(LOG), f)
        print(f"[LOG] Final: {final_log_path}")

        cap.release()
        try:
            plt.close("all")
        except Exception:
            pass


if __name__ == "__main__":
    run_live()

