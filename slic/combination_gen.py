from skimage import io
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# TODO: get the combination of bands and output folder through command line

COMBINATION = np.array([6, 5, 1]) - 1


def band_norm(band):
    band_min, band_max = band.min(), band.max()
    return ((band - band_min) / (band_max - band_min))


loop = tqdm(range(1, 11))

for i in loop:
    region_scene = np.load(f'scenes_allbands/allbands_x{i :02d}.npy')
    region_scene = region_scene[COMBINATION]

    np.save(f'scenes_651/651_x{i :02d}.npy', region_scene)

print('All Done!')