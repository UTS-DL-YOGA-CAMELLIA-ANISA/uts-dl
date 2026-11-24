import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# selalu jalan relatif ke root repositori
os.chdir(Path(__file__).resolve().parent.parent)
os.makedirs('figures', exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

print("Generating EDA Figures for Report...")

# ==========================================
# 1. TASK 1 FIGURES
# ==========================================
print("Generating Task 1 figures...")
meta_train = pd.read_csv('klasifikasi_train_meta.csv')
meta_test = pd.read_csv('klasifikasi_test_meta.csv')

with np.load('klasifikasi_train.npz') as d:
    train_imgs = d['gambar']
    train_sens = d['sensor'].astype(np.float32)

# Figure 1.1: Class distribution & Lini domain shift
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
class_counts = meta_train['kelas'].value_counts()
sns.barplot(x=class_counts.values, y=class_counts.index, hue=class_counts.index, palette='viridis', legend=False, ax=axes[0])
axes[0].set_title('(a) Distribusi Kelas Data Latih (Imbalance Ratio ~10:1)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Jumlah Sampel')
for i, v in enumerate(class_counts.values):
    axes[0].text(v + 200, i, f"{v} ({v/len(meta_train)*100:.1f}%)", va='center', fontsize=9)

# Domain shift Lini A vs B
domain_df = pd.DataFrame({
    'Train (48k)': meta_train['lini'].value_counts(normalize=True) * 100,
    'Test (16k)': meta_test['lini'].value_counts(normalize=True) * 100
}).T
domain_df.plot(kind='bar', stacked=True, color=['#3498db', '#e74c3c'], ax=axes[1])
axes[1].set_title('(b) Pergeseran Domain Lini Produksi (Train vs Test)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Persentase (%)')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
for p in axes[1].patches:
    width, height = p.get_width(), p.get_height()
    if height > 5:
        x, y = p.get_xy()
        axes[1].text(x + width/2, y + height/2, f"{height:.1f}%", ha='center', va='center', color='white', fontweight='bold')

plt.tight_layout()
plt.savefig('figures/task1_class_dist.png', dpi=200)
plt.close()

# Figure 1.2: Modality comparison across classes (7 classes)
classes = ['normal', 'gores', 'noda', 'retak', 'gelembung', 'las_dingin', 'kontaminasi']
fig, axes = plt.subplots(7, 4, figsize=(14, 14), gridspec_kw={'width_ratios': [1, 2, 2, 2]})

for row, cls in enumerate(classes):
    # Find a valid sample with both image and sensor
    sample_indices = meta_train[meta_train['kelas'] == cls].index.values
    for idx in sample_indices:
        if not np.isnan(train_sens[idx]).any():
            chosen_idx = idx
            break
            
    img = train_imgs[chosen_idx]
    sens = train_sens[chosen_idx]
    
    # Image
    axes[row, 0].imshow(img, cmap='gray')
    axes[row, 0].set_title(f"Kelas: {cls}\n(Unit: {meta_train.loc[chosen_idx, 'id_unit']})", fontsize=9, fontweight='bold')
    axes[row, 0].axis('off')
    
    # Suhu (C0)
    axes[row, 1].plot(sens[0], color='#d35400', lw=1.2)
    axes[row, 1].set_title('Suhu (°C)', fontsize=9)
    if row == 6: axes[row, 1].set_xlabel('Time Step (128)')
    
    # Getaran (C1)
    axes[row, 2].plot(sens[1], color='#27ae60', lw=1.0)
    axes[row, 2].set_title('Getaran (Relatif)', fontsize=9)
    if row == 6: axes[row, 2].set_xlabel('Time Step (128)')
    
    # Arus (C2)
    axes[row, 3].plot(sens[2], color='#2980b9', lw=1.2)
    axes[row, 3].set_title('Arus (A)', fontsize=9)
    if row == 6: axes[row, 3].set_xlabel('Time Step (128)')

plt.suptitle('Karakteristik Multimodal per Kelas Cacat (Citra Permukaan & Sinyal Sensor Pres)', fontsize=13, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('figures/task1_sample_modalities.png', dpi=200)
plt.close()

# ==========================================
# 2. TASK 2 FIGURES
# ==========================================
print("Generating Task 2 figures...")
rul_train = pd.read_csv('rul_train.csv.gz')

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Lifespan distribution
max_cycles = rul_train.groupby('id_unit')['siklus'].max()
sns.histplot(max_cycles, bins=30, kde=True, color='#2c3e50', ax=axes[0])
axes[0].axvline(max_cycles.median(), color='red', linestyle='--', label=f'Median: {max_cycles.median():.0f} siklus')
axes[0].set_title('(a) Distribusi Umur Pakai Mesin (900 Unit Train)', fontweight='bold')
axes[0].set_xlabel('Total Siklus Hingga Rusak')
axes[0].set_ylabel('Jumlah Mesin')
axes[0].legend()

# Operating conditions (setel_1 vs setel_2)
axes[1].scatter(rul_train['setel_1'].iloc[::50], rul_train['setel_2'].iloc[::50], c=rul_train['setel_3'].iloc[::50], cmap='coolwarm', alpha=0.4, s=15)
axes[1].set_title('(b) 6 Kondisi Operasi Mesin (Regimes)', fontweight='bold')
axes[1].set_xlabel('Pengaturan Operasi 1 (setel_1)')
axes[1].set_ylabel('Pengaturan Operasi 2 (setel_2)')

# Sensor degradation example (unit 3)
unit3 = rul_train[rul_train['id_unit'] == 3].copy()
axes[2].plot(unit3['siklus'], unit3['s02'], label='Sensor s02', color='#e67e22', lw=1.2)
axes[2].plot(unit3['siklus'], unit3['s11'], label='Sensor s11', color='#8e44ad', lw=1.2)
axes[2].set_title('(c) Tren Degradasi Sensor Unit 3 Hingga Rusak', fontweight='bold')
axes[2].set_xlabel('Siklus Operasi')
axes[2].set_ylabel('Nilai Sensor Mentah')
axes[2].legend()

plt.tight_layout()
plt.savefig('figures/task2_sensors_and_regimes.png', dpi=200)
plt.close()

# Figur eksplorasi Task 3 dibuat langsung oleh kode/task3_clustering.py
print("All EDA figures successfully created in figures/ directory!")
