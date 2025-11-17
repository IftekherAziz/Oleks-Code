import numpy as np

def convert_proper_motions(df):
    k = 4.74047
    parallax = df['parallax']
    safe = (parallax != 0) & parallax.notna()

    df['v_ra_kms'] = np.where(safe, k * df['pmra'] / parallax, np.nan)
    df['v_dec_kms'] = np.where(safe, k * df['pmdec'] / parallax, np.nan)
    return df