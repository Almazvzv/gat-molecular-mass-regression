# 5 000 молекул + tPSA 

import sys, subprocess, warnings, signal, time, random
warnings.filterwarnings('ignore')

def install(pkg, imp=None):
    if imp is None: imp = pkg.replace('-', '_')
    try: __import__(imp)
    except: subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

print("Установка библиотек...")
install("numpy"), install("pandas"), install("matplotlib"), install("scipy")
install("scikit-learn", "sklearn")
try: import torch
except: subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch"]); import torch
install("rdkit", "rdkit")
install("pubchempy", "pubchempy")
try: import torch_geometric
except:
    tv = torch.__version__.split("+")[0]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
        "torch-scatter", "torch-sparse", "torch-cluster", "torch-spline-conv", 
        "torch-geometric", "-f", f"https://data.pyg.org/whl/torch-{tv}.html"])

import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy import stats
from rdkit import Chem
from rdkit.Chem import Descriptors
import pubchempy as pcp

print("Библиотеки загружены\n")

# 1. СКАЧИВАНИЕ 5 000

TOTAL, TIMEOUT = 5000, 8
random.seed(42)
cid_ranges = [
    random.sample(range(100_000, 1_000_000), 2000),
    random.sample(range(1_000_000, 5_000_000), 2000),
    random.sample(range(5_000_000, 20_000_000), 2000),
]
CID_LIST = []
for r in cid_ranges: CID_LIST.extend(r)
random.shuffle(CID_LIST)
CID_LIST = CID_LIST[:TOTAL]

print(f"Скачивание {TOTAL} молекул из PubChem...\n")
compounds, start = [], time.time()
for i, cid in enumerate(CID_LIST):
    try:
        signal.alarm(TIMEOUT)
        c = pcp.Compound.from_cid(cid)
        signal.alarm(0)
        if c and c.canonical_smiles and c.molecular_weight:
            compounds.append(c)
    except: signal.alarm(0)
    if (i+1) % 500 == 0:
        print(f"  {i+1}/{TOTAL} | успешно: {len(compounds)} | {time.time()-start:.0f} сек")
print(f"\nСкачано {len(compounds)} молекул за {time.time()-start:.0f} сек\n")


# 2. ПРИЗНАКИ + tPSA

ATOM_MASSES = {1:1.008,5:10.81,6:12.01,7:14.01,8:16.0,9:19.0,11:22.99,12:24.31,13:26.98,14:28.09,15:30.97,16:32.07,17:35.45,19:39.1,20:40.08,25:54.94,26:55.85,27:58.93,28:58.69,29:63.55,30:65.38,35:79.9,47:107.87,53:126.9,79:196.97}

def atom_features(a):
    an = a.GetAtomicNum()
    return [an, a.GetDegree(), a.GetFormalCharge(), a.GetTotalNumHs(),
            int(a.GetIsAromatic()), a.GetExplicitValence(), int(a.IsInRing()),
            ATOM_MASSES.get(an, 20.0)]

def mol_to_data(smiles, mass):
    if not smiles: return None
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    na = mol.GetNumAtoms()
    if na < 6 or na > 200: return None
    if na == 1 and mol.GetNumBonds() == 0: return None
    try: Chem.SanitizeMol(mol)
    except: return None
    if mass is None or mass < 30: return None
    if not any(a.GetAtomicNum() == 6 for a in mol.GetAtoms()): return None
    feats = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    s = feats.std(dim=0); s[s == 0] = 1e-6
    feats = (feats - feats.mean(dim=0)) / s
    edges = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edges.extend([[i, j], [j, i]])
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2, 0), dtype=torch.long)
    tpsa = Descriptors.TPSA(mol)
    return Data(x=feats, edge_index=ei, 
                y=torch.tensor([mass], dtype=torch.float),
                tpsa=torch.tensor([tpsa], dtype=torch.float),
                mass_raw=mass)

print("Создание графов...")
graphs = []
for c in compounds:
    g = mol_to_data(c.canonical_smiles, c.molecular_weight)
    if g: graphs.append(g)
print(f"Графов: {len(graphs)}")
sizes = [g.x.shape[0] for g in graphs]
tpsa_vals = [g.tpsa.item() for g in graphs]
print(f"Размеры: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.0f}")
print(f"tPSA: min={min(tpsa_vals):.0f}, max={max(tpsa_vals):.0f}, mean={np.mean(tpsa_vals):.0f}")


# 3. НОРМАЛИЗАЦИЯ

m_raw = np.array([g.mass_raw for g in graphs])
MASS_MEAN = m_raw.mean()
MASS_STD = m_raw.std()
TPSA_MEAN = np.mean(tpsa_vals); TPSA_STD = np.std(tpsa_vals)
print(f"Масса: средняя {MASS_MEAN:.0f} Да, std {MASS_STD:.0f} Да")

for g in graphs:
    g.y = (g.y - MASS_MEAN) / MASS_STD
    g.tpsa = (g.tpsa - TPSA_MEAN) / TPSA_STD


# 4. СТРАТИФИКАЦИЯ

mass_bins = pd.cut(m_raw, bins=5, labels=False)
bin_counts = np.bincount(mass_bins)
if np.min(bin_counts) < 2:
    mass_bins = pd.qcut(m_raw, q=5, labels=False, duplicates='drop')

idx_all = np.arange(len(graphs))
train_val_idx, test_idx = train_test_split(idx_all, test_size=0.15, random_state=42, stratify=mass_bins)
mass_bins_tv = mass_bins[train_val_idx]
train_idx, val_idx = train_test_split(train_val_idx, test_size=0.15/0.85, random_state=42, stratify=mass_bins_tv)
tr_g = [graphs[i] for i in train_idx]
va_g = [graphs[i] for i in val_idx]
te_g = [graphs[i] for i in test_idx]
print(f"Train: {len(tr_g)}, Val: {len(va_g)}, Test: {len(te_g)}")

B = 32
tr_ld = DataLoader(tr_g, B, shuffle=True, drop_last=True)
va_ld = DataLoader(va_g, B)
te_ld = DataLoader(te_g, B)


# 5. МОДЕЛЬ

class GATRegressor(nn.Module):
    def __init__(self, in_feat=8, hid=128, heads=4, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(in_feat, hid, heads=heads, dropout=dropout)
        self.ln1 = nn.LayerNorm(hid * heads)
        self.conv2 = GATConv(hid*heads, hid, heads=heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(hid * heads)
        self.conv3 = GATConv(hid*heads, hid, heads=1, dropout=dropout)
        self.ln3 = nn.LayerNorm(hid)
        self.dropout = nn.Dropout(dropout)
        self.reg = nn.Linear(hid + 1, 1, bias=False)
        self.global_bias = nn.Parameter(torch.tensor(0.0))
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.reg.weight)
        nn.init.zeros_(self.global_bias)

    def forward(self, x, edge_index, tpsa, batch):
        h = self.dropout(F.elu(self.ln1(self.conv1(x, edge_index))))
        h = self.dropout(F.elu(self.ln2(self.conv2(h, edge_index))))
        h = F.elu(self.ln3(self.conv3(h, edge_index)))
        h_G = global_mean_pool(h, batch)
        h_G = torch.cat([h_G, tpsa.unsqueeze(-1)], dim=-1)
        return self.reg(h_G).squeeze(-1) + self.global_bias

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GATRegressor().to(dev)
print(f"Параметры: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

crit = nn.MSELoss()
opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.5)


# 6. ОБУЧЕНИЕ

def denorm(p, t):
    return p * MASS_STD + MASS_MEAN, t * MASS_STD + MASS_MEAN

def run_epoch(ld, train=True):
    if train: model.train()
    else: model.eval()
    loss_sum, preds, targs = 0, [], []
    for d in ld:
        d = d.to(dev)
        if train: opt.zero_grad()
        with torch.set_grad_enabled(train):
            p = model(d.x, d.edge_index, d.tpsa, d.batch)
            l = crit(p, d.y)
        if train:
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        loss_sum += l.item() * d.num_graphs
        preds.extend(p.detach().cpu().numpy())
        targs.extend(d.y.cpu().numpy())
    p_real = np.array(preds) * MASS_STD + MASS_MEAN
    t_real = np.array(targs) * MASS_STD + MASS_MEAN
    n = len(ld.dataset)
    return loss_sum/n, np.mean(np.abs(p_real-t_real)), np.sqrt(np.mean((p_real-t_real)**2)), r2_score(t_real, p_real), p_real, t_real

EPOCHS = 100
best_mae = float('inf')
history = {'val_loss': [], 'val_mae': [], 'val_r2': []}

print("\n" + "=" * 50)
print("ОБУЧЕНИЕ (v14: tPSA, без рёбер, без LogP)")
print("=" * 50)

for ep in range(1, EPOCHS+1):
    run_epoch(tr_ld, True)
    vl, vmae, vrmse, vr2, _, _ = run_epoch(va_ld, False)
    sch.step()
    history['val_loss'].append(vl); history['val_mae'].append(vmae); history['val_r2'].append(vr2)
    if vmae < best_mae:
        best_mae = vmae
        torch.save(model.state_dict(), 'best_model_v14.pth')
    if ep % 10 == 0 or ep == 1:
        print(f"Ep {ep:3d} | Loss:{vl:.3f} | MAE:{vmae:.0f} Да | R²:{vr2:.4f} | Bias:{model.global_bias.item():.4f}")

print(f"\nЛучшая MAE на валидации: {best_mae:.0f} Да")
print(f"Глобальный bias: {model.global_bias.item() * MASS_STD:.1f} Да")


# 7. ТЕСТ

model.load_state_dict(torch.load('best_model_v14.pth', map_location=dev))
_, tmae, trmse, tr2, tp, tt = run_epoch(te_ld, False)
print(f"\nТЕСТ: MAE = {tmae:.0f} Да | RMSE = {trmse:.0f} Да | R² = {tr2:.4f}")

errors = tp - tt
mean_err = np.mean(errors)
std_err = np.std(errors, ddof=1)
ci = stats.t.interval(0.95, df=len(errors)-1, loc=mean_err, scale=std_err/np.sqrt(len(errors)))
print(f"Средняя ошибка: {mean_err:.1f} Да")
print(f"95% ДИ: [{ci[0]:.1f}, {ci[1]:.1f}] Да")
print(f"Ошибка < 50 Да: {np.mean(np.abs(errors) < 50)*100:.1f}%")
print(f"Ошибка < 100 Да: {np.mean(np.abs(errors) < 100)*100:.1f}%")


# 8. ГРАФИКИ

def smooth(y, w=7):
    if len(y) < w: return np.array(y)
    return np.convolve(y, np.ones(w)/w, mode='valid')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (key, label, color) in zip(axes, 
    [('val_loss', 'Loss', 'red'), ('val_mae', 'MAE (Да)', 'green'), ('val_r2', 'R²', 'purple')]):
    ax.plot(history[key], alpha=0.3, color=color)
    if len(history[key]) >= 7:
        ax.plot(range(6, len(history[key])), smooth(history[key]), lw=2, color=color)
    ax.set_xlabel('Эпоха'); ax.set_ylabel(label); ax.set_title(label); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig('v14_curves.png', dpi=300); plt.show()

plt.figure(figsize=(8, 8))
plt.scatter(tt, tp, alpha=0.4, edgecolors='k', linewidth=0.3, s=20)
mx = max(tt.max(), tp.max()) * 1.05
plt.plot([0, mx], [0, mx], 'r--', lw=2)
plt.fill_between([0, mx], [0, mx], [tmae, mx+tmae], alpha=0.08, color='orange', label=f'±MAE = {tmae:.0f} Да')
plt.xlabel('Истинная масса (Да)'); plt.ylabel('Предсказанная масса (Да)')
plt.title(f'v14: R² = {tr2:.4f}, MAE = {tmae:.0f} Да')
plt.legend(); plt.grid(alpha=.3); plt.axis('square')
plt.xlim(0, mx); plt.ylim(0, mx)
plt.tight_layout(); plt.savefig('v14_scatter.png', dpi=300); plt.show()

plt.figure(figsize=(10, 6))
plt.hist(errors, bins=50, edgecolor='k', alpha=0.7, color='steelblue', density=True)
x_vals = np.linspace(errors.min(), errors.max(), 300)
plt.plot(x_vals, stats.norm.pdf(x_vals, mean_err, std_err), 'r-', lw=2)
plt.axvline(0, color='k', ls='--', lw=2)
plt.axvline(mean_err, color='orange', lw=2, label=f'Среднее = {mean_err:.1f} Да')
plt.axvline(ci[0], color='green', ls=':', lw=1.5)
plt.axvline(ci[1], color='green', ls=':', lw=1.5, label=f'95% ДИ = [{ci[0]:.0f}, {ci[1]:.0f}] Да')
plt.xlabel('Ошибка (Да)'); plt.ylabel('Плотность')
plt.title(f'Распределение ошибок (n = {len(errors)})')
plt.legend(); plt.grid(alpha=.3, axis='y')
plt.tight_layout(); plt.savefig('v14_hist.png', dpi=300); plt.show()

mass_bins_plot, labels = [0, 200, 400, 600, 800, 2000], ['<200','200–400','400–600','600–800','>800']
binned = [errors[(tt >= mass_bins_plot[i]) & (tt < mass_bins_plot[i+1])] for i in range(len(mass_bins_plot)-1)]
fig, ax = plt.subplots(figsize=(10, 6))
bp = ax.boxplot(binned, labels=labels, patch_artist=True)
for p in bp['boxes']: p.set_facecolor('steelblue'); p.set_alpha(0.6)
ax.axhline(0, color='r', ls='--', lw=2)
ax.set_xlabel('Диапазон истинной массы (Да)'); ax.set_ylabel('Ошибка (Да)')
ax.set_title('Ошибки по диапазонам массы (v14)'); ax.grid(alpha=.3, axis='y')
plt.tight_layout(); plt.savefig('v14_boxplot.png', dpi=300); plt.show()

print("\nГОТОВО! (v14)")