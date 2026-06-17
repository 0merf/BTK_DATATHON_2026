import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Seaborn ayarları
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

def run_eda():
    print("Veriler Yükleniyor...")
    train = pd.read_csv('../../train.csv')
    test = pd.read_csv('../../test_x.csv')
    
    img_dir = 'images'
    os.makedirs(img_dir, exist_ok=True)
    
    # 1. Target Distribution (Hedef Değişken Dağılımı)
    print("1. Target Distribution çiziliyor...")
    plt.figure(figsize=(10, 6))
    sns.histplot(train['career_success_score'], bins=50, kde=True, color='blue', alpha=0.6)
    plt.title('Kariyer Başarı Skoru Dağılımı', fontsize=15, fontweight='bold')
    plt.xlabel('Başarı Skoru (0-100)', fontsize=12)
    plt.ylabel('Öğrenci Sayısı', fontsize=12)
    plt.axvline(train['career_success_score'].mean(), color='red', linestyle='--', label=f"Ortalama: {train['career_success_score'].mean():.2f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{img_dir}/target_distribution.png', dpi=300)
    plt.close()
    
    # 2. Temporal Drift (Zaman Kayması - Başvuru Yılı Dağılımı)
    print("2. Temporal Drift çiziliyor...")
    plt.figure(figsize=(12, 6))
    train_counts = train['application_year'].value_counts(normalize=True).sort_index() * 100
    test_counts = test['application_year'].value_counts(normalize=True).sort_index() * 100
    
    years = sorted(list(set(train_counts.index).union(set(test_counts.index))))
    x = np.arange(len(years))
    width = 0.35
    
    train_vals = [train_counts.get(y, 0) for y in years]
    test_vals = [test_counts.get(y, 0) for y in years]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, train_vals, width, label='Eğitim Verisi', color='royalblue')
    ax.bar(x + width/2, test_vals, width, label='Test Verisi', color='darkorange')
    
    ax.set_title('Zaman Kayması (Temporal Drift) Analizi: Başvuru Yılı Dağılımı', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(y)) for y in years])
    ax.set_xlabel('Başvuru Yılı', fontsize=12)
    ax.set_ylabel('Oran (%)', fontsize=12)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{img_dir}/temporal_drift.png', dpi=300)
    plt.close()
    
    # 3. Korelasyon Matrisi (En önemli 10 sayısal özellik)
    print("3. Correlation Matrix çiziliyor...")
    plt.figure(figsize=(12, 10))
    numeric_cols = train.select_dtypes(include=['int64', 'float64']).columns
    correlations = train[numeric_cols].corr()['career_success_score'].abs().sort_values(ascending=False)
    top_cols = correlations.index[1:11] # Hedef hariç en yüksek 10
    
    heatmap_cols = list(top_cols) + ['career_success_score']
    cm = train[heatmap_cols].corr()
    
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, square=True)
    plt.title('En Önemli 10 Özelliğin Korelasyon Matrisi', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{img_dir}/correlation_matrix.png', dpi=300)
    plt.close()
    
    # 4. Üniversite Seviyesi vs Başarı Skoru (Kategorik Etki)
    print("4. University Tier vs Score çiziliyor...")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='university_tier', y='career_success_score', data=train, order=['Tier 1', 'Tier 2', 'Tier 3'], palette='Set2')
    plt.title('Üniversite Seviyesine (Tier) Göre Kariyer Başarısı', fontsize=15, fontweight='bold')
    plt.xlabel('Üniversite Seviyesi', fontsize=12)
    plt.ylabel('Başarı Skoru', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{img_dir}/university_tier_boxplot.png', dpi=300)
    plt.close()
    
    print("Tüm grafikler 'images/' klasörüne başarıyla kaydedildi!")

if __name__ == '__main__':
    run_eda()
