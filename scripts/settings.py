import matplotlib.pyplot as plt

EJ2_SERIES = ["3", "4", "1", "7", "8", "6"]
EJ2_SERIES_NAMES = ['EJ2({})'.format(i) for i in EJ2_SERIES]
EJ2_SERIES_LABELS = [r'$EJ2^{pro3}$', r'$EJ2^{pro4}$', r'$EJ2^{pro1}$',
                     r'$EJ2^{pro7}$', r'$EJ2^{pro8}$' ,r'$EJ2^{pro6}$']
G_SITES_SIMPLE = ["PLT3", "PLT7", "J2"]
G_SITES = G_SITES_SIMPLE + EJ2_SERIES_NAMES
GE_SITES = G_SITES + ["Season"]

SEASONS = ["Summer 22", "Fall 22", "Spring 23", "Summer 23"]  #
ALLELES = ["W", "H", "M"]
NUCLEOTIDES = ["A", "C", "G", "T"]
SEASONS_DICT = dict(zip(SEASONS, NUCLEOTIDES))
ALLELES_DICT = dict(zip(ALLELES, NUCLEOTIDES))

FIG_WIDTH = 7
LIMS = (0.0025, 150)

# Fonts
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.titlesize"] = 8
plt.rcParams["axes.labelsize"] = 6
plt.rcParams["xtick.labelsize"] = 5
plt.rcParams["ytick.labelsize"] = 5
plt.rcParams["legend.fontsize"] = 5
plt.rcParams["legend.labelspacing"] = 0.1

plt.rcParams["axes.titlepad"] = 3
plt.rcParams['axes.labelpad'] = 2
plt.rcParams['xtick.major.pad'] = 1.5
plt.rcParams['ytick.major.pad'] = 1.5

# Linewidths
plt.rcParams["axes.linewidth"] = 0.5
plt.rcParams["xtick.major.width"] = 0.5
plt.rcParams["ytick.major.width"] = 0.5
plt.rcParams["xtick.minor.width"] = 0.25
plt.rcParams["ytick.minor.width"] = 0.25
