import sys
import os
import csv
import numpy as np
from PIL import Image

from src import aes, lsb, steganalysis


# ============================================================
# PIPELINE UTAMA: EMBED
# ============================================================

def embed_pipeline(cover_path: str, message: str, password: str, output_path: str):
    """
    Pipeline embedding: enkripsi AES-128 → sisipkan ke LSB.
    """
    print(f"\n{'='*55}")
    print("  EMBED: LSB + AES-128")
    print(f"{'='*55}")

    cover_img = lsb.load_image(cover_path)
    capacity = lsb.get_capacity(cover_img)
    print(f"  Citra cover   : {cover_path} ({cover_img.shape[1]}x{cover_img.shape[0]})")
    print(f"  Kapasitas     : {capacity} bytes")

    key = aes.generate_key(password)
    iv, ciphertext = aes.encrypt(message.encode('utf-8'), key)
    payload = iv + ciphertext
    print(f"  Ukuran pesan  : {len(message)} karakter")
    print(f"  Payload (enc) : {len(payload)} bytes")

    if len(payload) > capacity:
        print(f"\n  [ERROR] Payload melebihi kapasitas! ({len(payload)} > {capacity})")
        return None

    stego_img = lsb.embed(cover_img, payload)
    lsb.save_image(stego_img, output_path)

    extracted_payload = lsb.extract(stego_img)
    ber = lsb.calculate_ber(payload, extracted_payload)

    metrics = lsb.calculate_metrics(cover_img, stego_img)
    print(f"\n  Output stego  : {output_path}")
    print(f"  MSE           : {metrics['mse']}")
    print(f"  PSNR          : {metrics['psnr']} dB")
    print(f"  BER           : {ber} ({'Lossless ✓' if ber == 0.0 else 'Ada error!'})")
    print(f"{'='*55}\n")

    return {**metrics, 'ber': ber}


# ============================================================
# PIPELINE UTAMA: EXTRACT
# ============================================================

def extract_pipeline(stego_path: str, password: str) -> str:
    """
    Pipeline ekstraksi: ekstrak bit LSB → dekripsi AES-128.
    """
    print(f"\n{'='*55}")
    print("  EXTRACT: LSB + AES-128")
    print(f"{'='*55}")

    stego_img = lsb.load_image(stego_path)
    payload = lsb.extract(stego_img)

    iv = payload[:16]
    ciphertext = payload[16:]
    key = aes.generate_key(password)
    plaintext = aes.decrypt(ciphertext, key, iv)
    message = plaintext.decode('utf-8')

    print(f"  Stego-image   : {stego_path}")
    print(f"  Payload size  : {len(payload)} bytes")
    print(f"  Pesan asli    : {message}")
    print(f"{'='*55}\n")

    return message


# ============================================================
# EKSPERIMEN 1: PERBANDINGAN LSB BIASA vs LSB + AES-128
# ============================================================

def run_experiment(cover_path: str, message: str, password: str):
    """
    Eksperimen komparasi LSB Biasa dan LSB + AES-128.
    """
    print(f"\n{'='*60}")
    print("  EKSPERIMEN: LSB Biasa vs LSB + AES-128")
    print(f"{'='*60}")

    cover_img = lsb.load_image(cover_path)
    print(f"  Citra cover   : {cover_path} ({cover_img.shape[1]}x{cover_img.shape[0]})")
    print(f"  Kapasitas     : {lsb.get_capacity(cover_img)} bytes")
    print(f"  Pesan uji     : '{message[:50]}...' ({len(message)} char)\n")

    # print("=== CHI-SQUARE COVER IMAGE ===")
    # cover_chi = steganalysis.chi_square_attack(cover_img)
    # print(cover_chi)
    # print()

    payload_plain = message.encode('utf-8')

    # --- EXP 1: LSB Biasa ---
    print("  [EXP 1] Menjalankan LSB Biasa...")
    stego_plain = lsb.embed(cover_img, payload_plain)
    extracted_plain = lsb.extract(stego_plain)
    m1 = lsb.calculate_metrics(cover_img, stego_plain)
    b1 = lsb.calculate_ber(payload_plain, extracted_plain)
    c1 = steganalysis.chi_square_attack(stego_plain)
    r1 = steganalysis.rs_analysis(stego_plain)
    print(f"    MSE         : {m1['mse']}")
    print(f"    PSNR        : {m1['psnr']} dB")
    print(f"    BER         : {b1}")
    print(f"    Chi-square  : {c1['conclusion']}")
    print(f"    RS Analysis : {r1['conclusion']}\n")

    # --- EXP 2: LSB + AES-128 ---
    print("  [EXP 2] Menjalankan LSB + AES-128...")
    key = aes.generate_key(password)
    iv, ciphertext = aes.encrypt(payload_plain, key)
    payload_enc = iv + ciphertext
    stego_aes = lsb.embed(cover_img, payload_enc)
    extracted_enc = lsb.extract(stego_aes)
    m2 = lsb.calculate_metrics(cover_img, stego_aes)
    b2 = lsb.calculate_ber(payload_enc, extracted_enc)
    c2 = steganalysis.chi_square_attack(stego_aes)
    r2 = steganalysis.rs_analysis(stego_aes)
    print(f"    MSE         : {m2['mse']}")
    print(f"    PSNR        : {m2['psnr']} dB")
    print(f"    BER         : {b2}")
    print(f"    Chi-square  : {c2['conclusion']}")
    print(f"    RS Analysis : {r2['conclusion']}\n")

    # --- OUTPUT TABEL DATA ---
    print(f"{'='*65}")
    print(f"  {'Metrik Pengujian':<25} | {'LSB Biasa':<16} | {'LSB + AES-128':<16}")
    print(f"  {'-'*61}")
    print(f"  {'MSE':<25} | {str(m1['mse']):<16} | {str(m2['mse']):<16}")
    print(f"  {'PSNR (dB)':<25} | {str(m1['psnr']):<16} | {str(m2['psnr']):<16}")
    print(f"  {'BER':<25} | {str(b1):<16} | {str(b2):<16}")
    print(f"  {'Chi-sq p-value':<25} | {str(c1['p_value']):<16} | {str(c2['p_value']):<16}")
    print(f"  {'Chi-sq Terdeteksi':<25} | {str(c1['detected']):<16} | {str(c2['detected']):<16}")
    print(f"  {'RS diff_R':<25} | {str(r1['diff_R']):<16} | {str(r2['diff_R']):<16}")
    print(f"  {'RS diff_S':<25} | {str(r1['diff_S']):<16} | {str(r2['diff_S']):<16}")
    print(f"  {'RS Terdeteksi':<25} | {str(r1['detected']):<16} | {str(r2['detected']):<16}")
    print(f"{'='*65}\n")

    results = {
        'lsb_plain': {'metrics': m1, 'ber': b1, 'chi': c1, 'rs': r1},
        'lsb_aes':   {'metrics': m2, 'ber': b2, 'chi': c2, 'rs': r2}
    }

    save = input("  Export hasil ke CSV? [y/n]: ").strip().lower()
    if save == 'y':
        _export_experiment_csv(results, cover_path, len(message))

    return results


# ============================================================
# EKSPERIMEN 2: VARIASI UKURAN PESAN
# ============================================================

def run_varexperiment(cover_path: str, password: str):
    """
    Eksperimen variasi ukuran payload terhadap tingkat kecurigaan deteksi.
    """
    cover_img = lsb.load_image(cover_path)
    capacity = lsb.get_capacity(cover_img)

    print(f"\n{'='*65}")
    print("  EKSPERIMEN VARIASI UKURAN PESAN")
    print(f"{'='*65}")
    print(f"  Citra cover : {cover_path} ({cover_img.shape[1]}x{cover_img.shape[0]})")
    print(f"  Kapasitas   : {capacity} bytes")
    print(f"  Password    : {password}\n")

    try:
        n = int(input("  Berapa variasi ukuran pesan yang ingin diuji? (contoh: 3): "))
    except ValueError:
        print("  [ERROR] Input harus berupa angka bulat.")
        return None

    sizes = []
    for i in range(n):
        while True:
            try:
                s = int(input(f"  Ukuran pesan ke-{i+1} (dalam karakter, maks ~{capacity-50}): "))
                if s <= 0:
                    print("  [WARNING] Ukuran pesan harus lebih besar dari 0.")
                elif s > capacity - 50:
                    print(f"  [WARNING] Payload terlalu besar! Batas aman sekitar {capacity-50} karakter.")
                else:
                    sizes.append(s)
                    break
            except ValueError:
                print("  [ERROR] Sila masukkan angka yang valid.")

    print(f"\n  Memproses pengujian untuk {n} variasi ukuran...\n")

    results = []
    BASE_CHAR = "A"

    for size in sizes:
        message = BASE_CHAR * size
        payload_plain = message.encode('utf-8')

        stego_plain = lsb.embed(cover_img, payload_plain)
        extracted_plain = lsb.extract(stego_plain)
        m1 = lsb.calculate_metrics(cover_img, stego_plain)
        b1 = lsb.calculate_ber(payload_plain, extracted_plain)
        c1 = steganalysis.chi_square_attack(stego_plain)
        r1 = steganalysis.rs_analysis(stego_plain)

        key = aes.generate_key(password)
        iv, ciphertext = aes.encrypt(payload_plain, key)
        payload_enc = iv + ciphertext
        stego_aes = lsb.embed(cover_img, payload_enc)
        extracted_enc = lsb.extract(stego_aes)
        m2 = lsb.calculate_metrics(cover_img, stego_aes)
        b2 = lsb.calculate_ber(payload_enc, extracted_enc)
        c2 = steganalysis.chi_square_attack(stego_aes)
        r2 = steganalysis.rs_analysis(stego_aes)

        results.append({
            'size': size,
            'lsb_plain': {'metrics': m1, 'ber': b1, 'chi': c1, 'rs': r1},
            'lsb_aes':   {'metrics': m2, 'ber': b2, 'chi': c2, 'rs': r2}
        })

    # --- OUTPUT TABEL DATA ---
    print("=" * 100)
    print("  [MENU 4] HASIL EKSPERIMEN SKALABILITAS (VARIASI UKURAN PAYLOAD)")
    print("=" * 100)
    print(f"  Citra Cover : {cover_path} ({cover_img.shape[1]}x{cover_img.shape[0]})")
    print(f"  Kapasitas   : {capacity} bytes")
    print("-" * 100)

    print("\n[BAGIAN 1: METRIK KUALITAS VISUAL & INTEGRITAS DATA]\n")
    print(f"  {'Ukuran':>6} | {'MSE Plain':>12} | {'MSE AES':>12} | {'PSNR Plain':>13} | {'PSNR AES':>13} | {'BER Plain':>9} | {'BER AES':>7}")
    print("  " + "-" * 91)
    for r in results:
        size = r['size']
        p = r['lsb_plain']['metrics']
        a = r['lsb_aes']['metrics']
        b1 = r['lsb_plain']['ber']
        b2 = r['lsb_aes']['ber']
        print(f"  {size:>6} | {p['mse']:>12.6f} | {a['mse']:>12.6f} | {p['psnr']:>10.4f} dB | {a['psnr']:>10.4f} dB | {b1:>9.1f} | {b2:>7.1f}")

    print("\n" + "-" * 100)

    print("\n[BAGIAN 2: METRIK KEAMANAN STEGANALISIS]\n")
    print(f"  {'Ukuran':>6} | {'Chi Stat (Plain)':<22} | {'Chi Stat (AES)':<22} | {'RS Diff (Plain)':<19} | {'RS Diff (AES)':<19}")
    print("  " + "-" * 91)
    for r in results:
        size = r['size']
        cp = r['lsb_plain']['chi']
        ca = r['lsb_aes']['chi']
        rp = r['lsb_plain']['rs']
        ra = r['lsb_aes']['rs']

        chi1_str = f"{cp['chi_square']:.2f} [{'TERDETEKSI' if cp['detected'] else 'AMAN'}]"
        chi2_str = f"{ca['chi_square']:.2f} [{'TERDETEKSI' if ca['detected'] else 'AMAN'}]"
        rs1_str  = f"{rp['diff_R']:.4f} [{'TERDETEKSI' if rp['detected'] else 'AMAN'}]"
        rs2_str  = f"{ra['diff_R']:.4f} [{'TERDETEKSI' if ra['detected'] else 'AMAN'}]"

        print(f"  {size:>6} | {chi1_str:<22} | {chi2_str:<22} | {rs1_str:<19} | {rs2_str:<19}")

    print("\n" + "=" * 100 + "\n")

    save = input("  Export hasil ke CSV? [y/n]: ").strip().lower()
    if save == 'y':
        _export_varexperiment_csv(results, cover_path)

    return results


# ============================================================
# EXPORT CSV
# ============================================================

def _export_experiment_csv(results: dict, cover_path: str, msg_len: int):
    filename = "hasil_eksperimen_komparasi.csv"
    plain = results['lsb_plain']
    aes_r = results['lsb_aes']

    rows = [
        ["Metrik", "LSB Biasa", "LSB + AES-128"],
        ["MSE",           plain['metrics']['mse'],   aes_r['metrics']['mse']],
        ["PSNR (dB)",     plain['metrics']['psnr'],  aes_r['metrics']['psnr']],
        ["BER",           plain['ber'],               aes_r['ber']],
        ["Chi-sq Stat",   plain['chi']['chi_square'], aes_r['chi']['chi_square']],
        ["Chi-sq p-value",plain['chi']['p_value'],    aes_r['chi']['p_value']],
        ["Chi-sq Deteksi",plain['chi']['detected'],   aes_r['chi']['detected']],
        ["RS diff_R",     plain['rs']['diff_R'],      aes_r['rs']['diff_R']],
        ["RS diff_S",     plain['rs']['diff_S'],      aes_r['rs']['diff_S']],
        ["RS Deteksi",    plain['rs']['detected'],    aes_r['rs']['detected']],
    ]

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f"# Cover: {cover_path} | Panjang pesan: {msg_len} karakter"])
        writer.writerows(rows)

    print(f"\n  [INFO] Hasil disimpan ke: {filename}")


def _export_varexperiment_csv(results: list, cover_path: str):
    filename = "hasil_eksperimen_variasi.csv"

    header = [
        "Ukuran (char)",
        "MSE Plain", "PSNR Plain", "BER Plain",
        "Chi Plain Stat", "Chi Plain p-val", "Chi Plain Deteksi",
        "RS Plain diff_R", "RS Plain diff_S", "RS Plain Deteksi",
        "MSE AES", "PSNR AES", "BER AES",
        "Chi AES Stat", "Chi AES p-val", "Chi AES Deteksi",
        "RS AES diff_R", "RS AES diff_S", "RS AES Deteksi",
    ]

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f"# Cover: {cover_path}"])
        writer.writerow(header)
        for r in results:
            p = r['lsb_plain']
            a = r['lsb_aes']
            writer.writerow([
                r['size'],
                p['metrics']['mse'], p['metrics']['psnr'], p['ber'],
                p['chi']['chi_square'], p['chi']['p_value'], p['chi']['detected'],
                p['rs']['diff_R'], p['rs']['diff_S'], p['rs']['detected'],
                a['metrics']['mse'], a['metrics']['psnr'], a['ber'],
                a['chi']['chi_square'], a['chi']['p_value'], a['chi']['detected'],
                a['rs']['diff_R'], a['rs']['diff_S'], a['rs']['detected'],
            ])

    print(f"\n  [INFO] Hasil disimpan ke: {filename}")


# ============================================================
# MODE INTERAKTIF
# ============================================================

def interactive_mode():
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')

        print("╔══════════════════════════════════════════════════════════════╗")
        print("║                                                              ║")
        print("║      ██╗     ██████╗ ██████╗        █████╗ ███████╗██████╗   ║")
        print("║      ██║    ██╔════╝ ██╔══██╗      ██╔══██╗██╔════╝██╔════╝  ║")
        print("║      ██║    ╚█████╗  ██████╔╝      ███████║█████╗  ███████╗  ║")
        print("║      ██║     ╚═══██╗ ██╔══██╗      ██╔══██║██╔══╝  ╚════██║  ║")
        print("║      ███████╗██████╔╝██████╔╝      ██║  ██║███████╗███████║  ║")
        print("║      ╚══════╝╚═════╝ ╚═════╝       ╚═╝  ╚═╝╚══════╝╚══════╝  ║")
        print("║                                                              ║")
        print("║                  S T E G A N O G R A P H Y                   ║")
        print("║                                                              ║")
        print("╠══════════════════════════════════════════════════════════════╣")
        print("║  WELCOME TO LSB-AES VAULT                                    ║")
        print("║  Secure Image Steganography Pipeline CLI                     ║")
        print("║  AES-128-CBC | Least Significant Bit Implementation          ║")
        print("║  Metrics: PSNR, MSE, BER, Chi-Square, RS Analysis            ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        print("-" * 64)
        print("  [1] Embed Pesan Rahasia ke Citra (Enkripsi AES)")
        print("  [2] Ekstrak & Dekripsi Pesan dari Citra Stego")
        print("  [3] Eksperimen Komparasi Matriks (1 Sampel Pesan)")
        print("  [4] Eksperimen Variasi Skala Ukuran Payload")
        print("  [5] Keluar dari Sistem")
        print("-" * 64)

        choice = input("\n  Pilih menu [1/2/3/4/5]: ").strip()

        if choice == "1":
            print("\n[MENU 1] PROSES EMBEDDING")
            cover  = input("  Path citra cover (contoh: cover.png): ").strip()
            output = input("  Path output stego (contoh: stego.png): ").strip()
            msg    = input("  Pesan rahasia: ").strip()
            pw     = input("  Password kunci: ").strip()
            embed_pipeline(cover, msg, pw, output)
            input("\n  [INFO] Proses selesai. Tekan Enter untuk kembali ke menu utama...")

        elif choice == "2":
            print("\n[MENU 2] PROSES EKSTRAKSI")
            stego = input("  Path citra stego (contoh: stego.png): ").strip()
            pw    = input("  Password kunci: ").strip()
            extract_pipeline(stego, pw)
            input("\n  [INFO] Proses selesai. Tekan Enter untuk kembali ke menu utama...")

        elif choice == "3":
            print("\n[MENU 3] EKSPERIMEN PERBANDINGAN")
            cover = input("  Path citra cover (contoh: cover.png): ").strip()
            msg   = input("  Pesan rahasia: ").strip()
            pw    = input("  Password kunci: ").strip()
            run_experiment(cover, msg, pw)
            input("\n  [INFO] Eksperimen selesai. Tekan Enter untuk kembali ke menu utama...")

        elif choice == "4":
            print("\n[MENU 4] EKSPERIMEN SKALABILITAS")
            cover = input("  Path citra cover (contoh: cover.png): ").strip()
            pw    = input("  Password kunci: ").strip()
            run_varexperiment(cover, pw)
            input("\n  [INFO] Eksperimen selesai. Tekan Enter untuk kembali ke menu utama...")

        elif choice == "5":
            print("\n  Terima kasih! Keluar dari sistem LSB+AES Vault.")
            break

        else:
            print("  [ERROR] Pilihan menu tidak tersedia.")
            input("  Tekan Enter untuk mencoba lagi...")


# ============================================================
# UTILITY HELPER
# ============================================================

def _make_dummy_cover(path: str, size: int = 512):
    dummy = np.random.randint(50, 200, (size, size, 3), dtype=np.uint8)
    Image.fromarray(dummy).save(path)
    print(f"  [INFO] Citra dummy {size}x{size} sukses dibuat: {path}")


# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:
        interactive_mode()

    elif sys.argv[1] == "embed" and len(sys.argv) == 6:
        _, _, cover, output, msg, pw = sys.argv
        embed_pipeline(cover, msg, pw, output)

    elif sys.argv[1] == "extract" and len(sys.argv) == 4:
        _, _, stego, pw = sys.argv
        extract_pipeline(stego, pw)

    elif sys.argv[1] == "experiment" and len(sys.argv) == 5:
        _, _, cover, msg, pw = sys.argv
        run_experiment(cover, msg, pw)

    elif sys.argv[1] == "varexperiment" and len(sys.argv) == 4:
        _, _, cover, pw = sys.argv
        run_varexperiment(cover, pw)

    else:
        print("\n[ERROR] Format parameter CLI salah.")
        print("Panduan Penggunaan:")
        print("  python main.py                               → Mode interaktif")
        print("  python main.py embed <cover> <out> <msg> <pw>")
        print("  python main.py extract <stego> <pw>")
        print("  python main.py experiment <cover> <msg> <pw>")
        print("  python main.py varexperiment <cover> <pw>")