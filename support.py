import seaborn as sns
import matplotlib.pyplot as plt
def custom_theme(ax, title = "", subtitle = ""):
    # Paleta tipo Dark2
    palette = sns.color_palette("Dark2")
    ax.set_prop_cycle(color = palette)

    # Título y subtítulo
    ax.set_title(title, fontsize = 15, fontweight = 'bold', color ="#003366", loc = 'center', pad = 20)
    if subtitle:
        ax.text(
            0.5, 1.01, subtitle,
            transform = ax.transAxes,
            ha = 'center', va = 'bottom',
            fontsize = 12, fontstyle = 'italic', color = 'gray'
        ) # Añadir subtítulo

    # Ejes
    ax.set_xlabel(ax.get_xlabel(), fontsize = 12, fontweight = 'bold', color ="#003366")
    ax.set_ylabel(ax.get_ylabel(), fontsize = 12, fontweight = 'bold', color ="#003366")
    ax.tick_params(axis = 'x', labelsize = 8)
    ax.tick_params(axis = 'y', labelsize = 8)
    ax.set_xticklabels([label.get_text() for label in ax.get_xticklabels()],
                       fontweight = 'bold', fontsize = 8)

    # Leyenda
    legend = ax.legend(title = 'Método', title_fontsize = 9, fontsize = 8,
                       frameon = True, edgecolor = 'black')

    # Bordes
    sns.despine(ax = ax, left = True, bottom = False)

    # Añadir líneas de fondo (grid horizontal estilo ggplot)
    ax.yaxis.grid(True, which = 'major', linestyle = '-', linewidth = 0.5, color ='#D3D3D3')
    ax.xaxis.grid(True, which = 'major', linestyle = '-', linewidth = 0.5, color ='#D3D3D3')