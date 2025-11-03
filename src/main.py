# all2vec_live_14x14.py

import os, cv2, time, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models, transforms
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle
from collections import defaultdict

# =========================
# Config
# =========================
DU = 256
B = 49   # 196個のパッチから128個を選択
PATCH_SIZE = 7  # 14×14 = 196 patches
LR = 1e-4
ETA_S = 0.08
U_INJ = 0.05
NOISE_STD = 0.00
FRAME_SKIP = 2
CLAMP_VAL = 5.0
EPS = 1e-6

STABILITY_LAMBDA = 0.0
SMOOTH_LAMBDA   = 0.0
#RECON_LAMBDA    = 1.0
#DYN_LAMBDA      = 1.0
RECON_LAMBDA    = 1.0
DYN_LAMBDA      = 1.0

PCA_REF = 2000
PCA_SEED = 42
SAVE_INTERVAL = 100
LOG_DIR = "all2vec_logs_7x7"
os.makedirs(LOG_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32
print(f"[INFO] device={device}, B={B}, patch_grid={PATCH_SIZE}×{PATCH_SIZE}")

# =========================
# Webcam
# =========================
def open_camera(preferred=(1,0), fallback=range(2,6)):
    tried = set()
    for idx in list(preferred)+list(fallback):
        if idx in tried: continue
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"[INFO] Camera {idx}")
            return cap
        tried.add(idx); cap.release()
    raise RuntimeError("No webcam")
cap = open_camera()

# =========================
# Encoder
# =========================
with torch.no_grad():
    mobilenet = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.DEFAULT
    ).to(device).eval()

preproc = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224), 
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

@torch.no_grad()
def image_to_feat(frame_bgr: np.ndarray) -> torch.Tensor:
    """
    粒子数に応じた最適レイヤー選択（効率化版）
    初回実行時に最適レイヤーを決定し、以降はそのレイヤーのみ使用
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    x = preproc(rgb).unsqueeze(0).to(device=device, dtype=dtype)
    
    # 初回のみ：最適レイヤーを決定
    if not hasattr(image_to_feat, 'selected_layer_idx'):
        print(f"[INFO] Determining optimal layer for B={B} particles...")
        
        # 全層を実行して解像度を調査
        layer_candidates = []
        temp_x = x.clone()
        
        with torch.no_grad():
            for i, layer in enumerate(mobilenet.features):
                temp_x = layer(temp_x)
                _, C, H, W = temp_x.shape
                n_patches = H * W
                layer_candidates.append({
                    'idx': i,
                    'n_patches': n_patches,
                    'channels': C,
                    'height': H,
                    'width': W
                })
        
        # B 以上のパッチを持つ層の中で最も深いものを選択
        valid_layers = [l for l in layer_candidates if l['n_patches'] >= B]

        if valid_layers:
            if B < 50:
                selected = valid_layers[-1]
            else:
                selected = valid_layers[-1]

            image_to_feat.selected_layer_idx = selected['idx']
            image_to_feat.n_patches = selected['n_patches']
            image_to_feat.channels = selected['channels']
            image_to_feat.height = selected['height']
            image_to_feat.width = selected['width']
            
            print(f"[INFO] Selected layer: {selected['idx']}")
            print(f"[INFO] Resolution: [{selected['channels']}, {selected['height']}, {selected['width']}]")
            print(f"[INFO] Patches: {selected['n_patches']} (required: {B})")
            print(f"[DEBUG] Sufficient patches. No interpolation needed.")
        else:
            # 全層がB未満の場合（ありえないはず）
            selected = layer_candidates[-1]
            image_to_feat.selected_layer_idx = selected['idx']
            image_to_feat.n_patches = selected['n_patches']
            image_to_feat.channels = selected['channels']
            image_to_feat.height = selected['height']
            image_to_feat.width = selected['width']
            
            print(f"[WARNING] No layer has {B}+ patches. Using deepest layer.")
            print(f"[WARNING] Layer {selected['idx']}: {selected['n_patches']} patches < {B} particles")
            print(f"[DEBUG] INTERPOLATION WOULD BE NEEDED BUT IS DISABLED")
    
    # 選択されたレイヤーまで順伝播
    with torch.no_grad():
        for i, layer in enumerate(mobilenet.features):
            x = layer(x)
            if i == image_to_feat.selected_layer_idx:
                fmap = x
                break
    
    # パッチ化
    C = image_to_feat.channels
    H = image_to_feat.height
    W = image_to_feat.width
    n_patches = image_to_feat.n_patches
    
    patches = fmap.view(C, H * W).T  # [n_patches, C]
    
    # B 個にサンプリング（補間なし）
    if B <= n_patches:
        indices = torch.linspace(0, n_patches - 1, B, device=device).long()
        particle_feats = patches[indices]
    else:
        # n_patches < B の場合（本来ありえない）
        # 補間禁止なので、利用可能なパッチを繰り返し使用
        indices = torch.linspace(0, n_patches - 1, B, device=device).long() % n_patches
        particle_feats = patches[indices]
        
        # 区別のための微小ノイズ
        noise = torch.randn_like(particle_feats) * 0.05
        particle_feats = particle_feats + noise
        
        if not hasattr(image_to_feat, 'warned'):
            print(f"[DEBUG] Using {n_patches} patches for {B} particles with noise differentiation")
            image_to_feat.warned = True
    
    # 線形射影
    if not hasattr(image_to_feat, "proj_W"):
        torch.manual_seed(42)
        image_to_feat.proj_W = torch.randn(DU, C, device=device, dtype=dtype) / np.sqrt(C)
    
    U = F.linear(particle_feats, image_to_feat.proj_W)
    U = U / (U.norm(dim=-1, keepdim=True) + EPS)
    
    # デバッグ出力
    if not hasattr(image_to_feat, 'call_count'):
        image_to_feat.call_count = 0
    image_to_feat.call_count += 1
    
    if image_to_feat.call_count == 1 or image_to_feat.call_count % 500 == 0:
        unique_count = len(torch.unique(U.cpu(), dim=0))
        print(f"[DEBUG] Step {image_to_feat.call_count}: {unique_count}/{B} unique particles")
    
    return U

# =========================
# Model（変更なし）
# =========================
class Projector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Identity()
    def forward(self, z):
        return self.net(z)

class Predictor(nn.Module):
    def __init__(self, du=DU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(du*2, du),
            nn.GELU(),
            nn.Linear(du, du)
        )
    def forward(self, S, U):
        return self.net(torch.cat([S, U], dim=-1))

class Reconstructor(nn.Module):
    def __init__(self, du=DU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(du, du),
            nn.GELU(),
            nn.Linear(du, du)
        )
    def forward(self, S):
        return self.net(S)

class ALL2VecLive(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_v = Projector()
        self.pred = Predictor()
        self.recon_v = Reconstructor()
    def forward_step(self, S, U_batch):
        U_batch = self.proj_v(U_batch)
        S_hat = self.pred(S, U_batch)
        recon = self.recon_v(S_hat)
        U_mean = U_batch.mean(dim=0, keepdim=True)
        return S_hat, recon, U_mean

model = ALL2VecLive().to(device).to(dtype)
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

# =========================
# State & PCA
# =========================
@torch.no_grad()
def init_state(b=B, du=DU):
    S = torch.randn(b, du, device=device, dtype=dtype)
    return S / (S.norm(dim=-1, keepdim=True) + EPS)

S = init_state()

pca = PCA(n_components=2, random_state=PCA_SEED)
with torch.no_grad():
    ref = torch.randn(PCA_REF, DU)
    ref = ref / (ref.norm(dim=-1, keepdim=True) + EPS)
    pca.fit(ref.cpu().numpy())
print("[INFO] PCA fitted")

# =========================
# Logging
# =========================
LOG = defaultdict(list)

# =========================
# Visualization（変更なし）
# =========================
plt.ion()
fig = plt.figure(figsize=(12,6))
gs = fig.add_gridspec(4,2, width_ratios=[2.0,1.2], height_ratios=[2.0,1.0,1.0,1.0])
ax_sc = fig.add_subplot(gs[:,0])
ax_cam = fig.add_subplot(gs[0,1])
ax_m1 = fig.add_subplot(gs[1,1])
ax_m2 = fig.add_subplot(gs[2,1])
ax_m3 = fig.add_subplot(gs[3,1])

sc = ax_sc.scatter([], [], s=12, alpha=0.9)
ax_sc.set_title(f"USF ({B} particles, {PATCH_SIZE}×{PATCH_SIZE} patches)")
ax_sc.set_xlabel("PC1"); ax_sc.set_ylabel("PC2")
ax_sc.grid(True, alpha=0.25)

ts_dyn, ts_recon, ts_stab = [], [], []
line1, = ax_m1.plot([], [], label='Pred')
line2, = ax_m2.plot([], [], label='Recon', color='tab:orange')
line3, = ax_m3.plot([], [], label='Stab', color='tab:green')
for ax in (ax_m1, ax_m2, ax_m3):
    ax.grid(True, alpha=0.3); ax.legend(loc='upper right')

ax_cam.set_title("Webcam")
ax_cam.axis('off')
_cam_artist = ax_cam.imshow(np.zeros((480,640,3), dtype=np.uint8))

fps_text = ax_sc.text(
    0.02, 0.95, 'FPS: --', transform=ax_sc.transAxes,
    fontsize=10, color='black', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
)

running = True
def _on_key(event):
    global running
    if event.key == 'q':
        running = False
        plt.close(fig)
fig.canvas.mpl_connect('key_press_event', _on_key)

def update_plot(S_cpu, frame_bgr, m_dyn, m_rec, m_stb, step, fps=None):
    XY = pca.transform(S_cpu)
    sc.set_offsets(XY)
    ax_sc.set_xlim(-0.3, 0.3); ax_sc.set_ylim(-0.3, 0.3)
    ax_sc.set_title(f"USF ({B} particles, {PATCH_SIZE}×{PATCH_SIZE}) — step {step}")
    
    if fps is not None:
        fps_text.set_text(f"FPS: {fps:.1f}")
    else:
        fps_text.set_text("FPS: --")

    ts_dyn.append(m_dyn); ts_recon.append(m_rec); ts_stab.append(m_stb)
    xs = np.arange(len(ts_dyn))
    line1.set_data(xs, ts_dyn)
    line2.set_data(xs, ts_recon)
    line3.set_data(xs, ts_stab)
    for ax,y in zip((ax_m1,ax_m2,ax_m3),(ts_dyn,ts_recon,ts_stab)):
        if len(y) < 3:
            ax.set_xlim(0, 30); ax.set_ylim(0, 1.0)
        else:
            ax.set_xlim(0, max(30, len(xs)))
            ymax = np.percentile(y, 95) if np.isfinite(y).all() else 1.0
            ax.set_ylim(0, float(max(1e-6, ymax))*1.2)
    
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    _cam_artist.set_data(frame_rgb)
    plt.pause(0.001)

# =========================
# Main loop（変更なし）
# =========================
print("[INFO] Press 'q' to quit")
step = 0
buffer_prev = None
prev_time = time.perf_counter()
fps_smooth = None

while running:
    ret, frame = cap.read()
    if not ret: break
    
    z_v = image_to_feat(frame).to(dtype)
    
    S_prev = S
    S_hat, recon_mean, U_mean = model.forward_step(S_prev, z_v)
    U_repeat = U_mean.repeat(B,1)
    
    with torch.no_grad():
        S_target = (1-ETA_S)*S_prev + ETA_S*S_hat.detach() + U_INJ*U_repeat
        S_target = S_target / (S_target.norm(dim=-1, keepdim=True) + EPS)
    
    S_prev_c = S_prev.clamp(-CLAMP_VAL, CLAMP_VAL)
    S_hat_c = S_hat.clamp(-CLAMP_VAL, CLAMP_VAL)
    stability_loss = F.mse_loss(S_hat_c, S_prev_c)
    smooth_loss = F.mse_loss(S_hat, S_prev)
    
    recon_c = recon_mean.clamp(-CLAMP_VAL, CLAMP_VAL)
    z_v_c = z_v.clamp(-CLAMP_VAL, CLAMP_VAL)
    recon_loss = F.mse_loss(recon_c, z_v_c)
    
    future_pred_loss = torch.tensor(0.0, device=device, dtype=dtype)
    if buffer_prev is not None:
        S_tm1, Umean_tm1 = buffer_prev
        U_tm1 = Umean_tm1.repeat(B,1)
        S_pred_from_tm1 = model.pred(S_tm1, U_tm1)
        S_target_detach = S_target.detach()
        future_pred_loss = F.mse_loss(
            S_pred_from_tm1.clamp(-CLAMP_VAL,CLAMP_VAL),
            S_target_detach.clamp(-CLAMP_VAL,CLAMP_VAL)
        )
    
    loss = (DYN_LAMBDA*future_pred_loss + RECON_LAMBDA*recon_loss + 
            STABILITY_LAMBDA*stability_loss + SMOOTH_LAMBDA*smooth_loss)
    
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    
    with torch.no_grad():
        noise = NOISE_STD * torch.randn_like(S_target)
        S = S_target + noise
        S = S / (S.norm(dim=-1, keepdim=True) + EPS)
    
    buffer_prev = (S_prev.detach(), U_mean.detach())
    
    # Logging
    if step % SAVE_INTERVAL == 0:
        with torch.no_grad():
            S_np = S.detach().cpu().numpy()
            LOG['states'].append(S_np.copy())
            S_pca = pca.transform(S_np)
            LOG['states_pca'].append(S_pca.copy())
            LOG['step'].append(step)
            LOG['future_pred_loss'].append(float(future_pred_loss.cpu()))
            LOG['recon_loss'].append(float(recon_loss.cpu()))
            LOG['stability_loss'].append(float(stability_loss.cpu()))
            S_mean = S.mean(dim=0, keepdim=True)
            dispersion = ((S - S_mean)**2).sum(dim=1).mean()
            LOG['dispersion'].append(float(dispersion.cpu()))
            if len(LOG['states']) > 1:
                S_prev_saved = torch.from_numpy(LOG['states'][-2]).to(device)
                temporal_stab = ((S - S_prev_saved)**2).sum(dim=1).mean()
                LOG['temporal_stability'].append(float(temporal_stab.cpu()))
            else:
                LOG['temporal_stability'].append(0.0)
    
    if step % (SAVE_INTERVAL * 10) == 0 and step > 0:
        log_path = f"{LOG_DIR}/log_step_{step:06d}.pkl"
        with open(log_path, 'wb') as f:
            pickle.dump(dict(LOG), f)
        print(f"[LOG] Saved {log_path}")
    
    if step % FRAME_SKIP == 0:
        S_cpu = S.detach().float().cpu().numpy()
        now = time.perf_counter()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            inst_fps = 1.0 / dt
            fps_smooth = inst_fps if fps_smooth is None else (0.9 * fps_smooth + 0.1 * inst_fps)
        update_plot(S_cpu, frame,
                    float(future_pred_loss.cpu()),
                    float(recon_loss.cpu()),
                    float(stability_loss.cpu()), step,
                    fps=fps_smooth)
    
    step += 1

final_log_path = f"{LOG_DIR}/log_final.pkl"
with open(final_log_path, 'wb') as f:
    pickle.dump(dict(LOG), f)
print(f"[LOG] Final: {final_log_path}")

cap.release()
plt.ioff(); plt.show()