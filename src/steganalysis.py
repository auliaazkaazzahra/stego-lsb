import numpy as np
import math
from scipy.stats import chi2

def calculate_metrics(cover: np.ndarray, stego: np.ndarray) -> dict:
    cover_f = cover.astype(np.float64)
    stego_f = stego.astype(np.float64)
    mse = np.mean((cover_f - stego_f) ** 2)

    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10 * math.log10((255.0 ** 2) / mse)
    return {'mse': round(mse, 6), 'psnr': round(psnr, 4)}

def chi_square_attack(image: np.ndarray) -> dict:
    flat = image.flatten().astype(int)
    freq = np.bincount(flat, minlength=256)
    chi_sq = 0.0
    dof = 0
    
    for k in range(0, 256, 2):
        expected = (freq[k] + freq[k+1]) / 2
        if expected > 0:
            chi_sq += ((freq[k] - expected) ** 2) / expected
            chi_sq += ((freq[k+1] - expected) ** 2) / expected
            dof += 1

    p_value = 1 - chi2.cdf(chi_sq, df=dof)
    detected = p_value < 0.05
    return {
        'chi_square': round(chi_sq, 4),
        'p_value': round(p_value, 6),
        'detected': detected,
        'conclusion': 'TERDETEKSI mengandung pesan' if detected else 'TIDAK terdeteksi'
    }

def rs_analysis(image: np.ndarray) -> dict:
    gray = np.mean(image, axis=2).astype(int)
    h, w = gray.shape
    group_size = 4

    def discriminant(group):
        return sum(abs(int(group[i+1]) - int(group[i])) for i in range(len(group)-1))

    def flip_lsb(group, mask):
        flipped = group.copy()
        for i, m in enumerate(mask):
            if m == 1:
                flipped[i] = flipped[i] ^ 1
            elif m == -1:
                if flipped[i] % 2 == 0:
                    flipped[i] += 1
                else:
                    flipped[i] -= 1
        return flipped

    mask = [1, 0, 1, 0]
    R, S, N = 0, 0, 0
    Rm, Sm, Nm = 0, 0, 0
    total_groups = 0

    for i in range(0, h - 1, 1):
        for j in range(0, w - group_size, group_size):
            group = gray[i, j:j+group_size].tolist()
            if len(group) < group_size: continue

            total_groups += 1
            f_original = discriminant(group)

            # Positive mask
            f_flipped = discriminant(flip_lsb(group, mask))
            if f_flipped > f_original: R += 1
            elif f_flipped < f_original: S += 1

            # Negative mask
            f_flipped_neg = discriminant(flip_lsb(group, [-m for m in mask]))
            if f_flipped_neg > f_original: Rm += 1
            elif f_flipped_neg < f_original: Sm += 1

    R_norm, S_norm = R / total_groups, S / total_groups
    Rm_norm, Sm_norm = Rm / total_groups, Sm / total_groups
    
    diff_R = abs(R_norm - Rm_norm)
    diff_S = abs(S_norm - Sm_norm)
    detected = diff_R < 0.02 and diff_S < 0.02

    return {
        'diff_R': round(diff_R, 4), 'diff_S': round(diff_S, 4),
        'detected': detected,
        'conclusion': 'TERDETEKSI mengandung pesan' if detected else 'TIDAK terdeteksi'
    }