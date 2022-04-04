# %%
import pandas as pd
import skimage as sk
import numpy as np
import os
from pathlib import Path
from matplotlib import pyplot as plt

# %%
WORKING_DIR = Path(os.getcwd())
# get file names stems of filtered images
fm_dir = list(WORKING_DIR.glob("fm.csv"))
assert fm_dir, "no file names fm.csv"

fm_files = pd.read_csv(fm_dir[0], header = None).iloc[:,0].to_numpy()
filteredImStem = [i[4:-6] for i in fm_files]
# %%
image_dir = Path("/Users/yiqingmelodywang/MFGTMP_220317120003")
filtered_ch2_dir = []
for stem in filteredImStem:
    filtered_ch2_dir.append(str(list(image_dir.glob(stem + "d2.TIF"))[0]))
# %%
ch2_img = []
for dir in filtered_ch2_dir:
    ch2_img.append(sk.io.imread(dir))

ch2_img_np = np.array(ch2_img)
max = np.percentile(ch2_img_np,99.9999)
min = float(np.min(ch2_img_np))
# %%
ch2_img_cs = (ch2_img_np - min) / (max - min) * 255.0
ch2_img_cs[ch2_img_cs > 255] = 255.0
# %%
ch2_img_cs = ch2_img_cs.round().astype(np.uint8)

# %%
# FIXME: a. VERY inefficient
# contrast stretch complete. now detecting edges
from LineDetector import *
# line_detection_non_vectorized(ch2_img_cs[0])
# %%
numLines = []
for i in range(len(ch2_img_cs)):
    numLines.append(line_detection_non_vectorized(ch2_img_cs[i]))

# %%
# numLinesNp= np.array(numLines)
# zeros = np.where(numLinesNp == 0)
# filtered_ch2_dirNp = np.array(filtered_ch2_dir)
# filtered_ch2_dir_noZero = np.delete(filtered_ch2_dirNp,np.where(numLinesNp == 0))
# %%
blast_untreated = []
continuous_untreated = []
wt_untreated = []
blast_colchicine = []
cont_colchicine = []
wt_colchicine = []
for i in range(len(filtered_ch2_dir)):
    if ("A01" in filtered_ch2_dir[i] or "H01" in filtered_ch2_dir[i]) and numLines[i] != 0:
        blast_untreated.append(numLines[i])
    if ("A02" in filtered_ch2_dir[i] or "H02" in filtered_ch2_dir[i]) and numLines[i] != 0:
        continuous_untreated.append(numLines[i])
    if ("A03" in filtered_ch2_dir[i] or "H03" in filtered_ch2_dir[i]) and numLines[i] != 0:
        wt_untreated.append(numLines[i])
    if ("D01" in filtered_ch2_dir[i] or "G01" in filtered_ch2_dir[i]) and numLines[i] != 0:
        blast_colchicine.append(numLines[i])
    if ("D02" in filtered_ch2_dir[i] or "G02" in filtered_ch2_dir[i]) and numLines[i] != 0:
        cont_colchicine.append(numLines[i])
    if ("D03" in filtered_ch2_dir[i] or "G03" in filtered_ch2_dir[i]) and numLines[i] != 0:
        wt_colchicine.append(numLines[i])
    
# %%
from scipy.stats import ttest_ind as ttest

blast_t, blast_p = ttest(blast_untreated, blast_colchicine)
cont_t, cont_p = ttest(continuous_untreated, cont_colchicine)
wt_t,wt_p = ttest(wt_untreated, wt_colchicine)

# %%
plt.boxplot(np.array([blast_untreated,blast_colchicine]).T, labels = ["untreated", "colchicine"])
plt.title("blast")
plt.show()
# %%
plt.boxplot(np.array([continuous_untreated,cont_colchicine]).T, labels = ["untreated", "colchicine"])
plt.title("continuous")
plt.show()
# %%
plt.boxplot(np.array([wt_untreated,wt_colchicine]).T, labels = ["untreated", "colchicine"])
plt.title("Wt")
plt.show()
# %%
