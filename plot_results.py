import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_experiment_results(csv_path="results/hasil_eksperimen_variasi.csv"):
    if not os.path.exists(csv_path):
        print(f"[ERROR] File {csv_path} tidak ditemukan. Run eksperimen variasi dulu!")
        return

    # Membaca CSV
    df = pd.read_csv(csv_path, comment='#')

    # Membersihkan nama kolom dari spasi berlebih
    df.columns = df.columns.str.strip()

    # Ambil data X (Ukuran Payload)
    x_sizes = df['Ukuran (char)']

    # =========================================================
    # GRAFIK 1: PSNR (Kualitas Visual)
    # =========================================================
    plt.figure(figsize=(8, 5))
    plt.plot(x_sizes, df['PSNR Plain'], marker='o', linestyle='-', linewidth=2, label='LSB Biasa')
    plt.plot(x_sizes, df['PSNR AES'], marker='s', linestyle='--', linewidth=2, label='LSB + AES-128')
    
    plt.title('Pengaruh Ukuran Payload terhadap PSNR', fontsize=14, fontweight='bold')
    plt.xlabel('Ukuran Payload (Karakter/Bytes)', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=11)
    
    # Menyimpan grafik
    plt.tight_layout()
    plt.savefig('results/grafik_psnr.png', dpi=300)
    plt.close()
    print("[INFO] Grafik PSNR berhasil disimpan: results/grafik_psnr.png")

    # =========================================================
    # GRAFIK 2: Chi-Square Statistic (Keamanan)
    # =========================================================
    plt.figure(figsize=(8, 5))
    plt.plot(x_sizes, df['Chi Plain Stat'], marker='o', color='red', linestyle='-', linewidth=2, label='LSB Biasa')
    plt.plot(x_sizes, df['Chi AES Stat'], marker='s', color='purple', linestyle='--', linewidth=2, label='LSB + AES-128')
    
    plt.title('Ketahanan terhadap Chi-Square Attack', fontsize=14, fontweight='bold')
    plt.xlabel('Ukuran Payload (Karakter/Bytes)', fontsize=12)
    plt.ylabel('Nilai Statistik Chi-Square', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=11)
    
    # Menyimpan grafik
    plt.tight_layout()
    plt.savefig('results/grafik_chi_square.png', dpi=300)
    plt.close()
    print("[INFO] Grafik Chi-Square berhasil disimpan: results/grafik_chi_square.png")

    # =========================================================
    # GRAFIK 3: RS Analysis diff_R (Keamanan)
    # =========================================================
    plt.figure(figsize=(8, 5))
    plt.plot(x_sizes, df['RS Plain diff_R (Red)'], marker='o', color='green', linestyle='-', linewidth=2, label='LSB Biasa')
    plt.plot(x_sizes, df['RS AES diff_R (Red)'], marker='s', color='orange', linestyle='--', linewidth=2, label='LSB + AES-128')
    
    plt.title('Ketahanan terhadap RS Analysis (Red Channel)', fontsize=14, fontweight='bold')
    plt.xlabel('Ukuran Payload (Karakter/Bytes)', fontsize=12)
    plt.ylabel('Selisih R dan R-min (diff_R)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=11)
    
    # Menyimpan grafik
    plt.tight_layout()
    plt.savefig('results/grafik_rs_analysis.png', dpi=300)
    plt.close()
    print("[INFO] Grafik RS Analysis berhasil disimpan: results/grafik_rs_analysis.png")

if __name__ == "__main__":
    plot_experiment_results()