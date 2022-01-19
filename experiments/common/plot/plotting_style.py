import matplotlib as mpl
import seaborn as sns


def set_rc_params():
    # Figure
    mpl.rcParams['figure.figsize'] = (6, 3)

    # Fontsizes
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['ytick.labelsize'] = 10
    mpl.rcParams['axes.labelsize'] = 12
    mpl.rcParams['axes.titlesize'] = 12

    # Colors
    # - Seaborn Color Palette: colorblind
    # - default context always plotted in black

    sns.set_style("whitegrid")
    sns.set_palette("colorblind")
