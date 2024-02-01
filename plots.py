import matplotlib.pyplot as plt

def Spectra(wavenumbers, windspeeds, spectrum, title, xlabel, ylabel, fileTitle):
    for idx,windspeed in enumerate(windspeeds):
        plt.plot(wavenumbers, spectrum[idx],label=f"{windspeed} m/s")
    plt.title(title)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1e-15, 1e3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results/{fileTitle}.png')
    plt.show()
    plt.close()

def Spread(theta, n, spreads, title, id, fileTitle, Elfouhaily=False):
    if Elfouhaily:
        for idx, i in enumerate(n):
            plt.polar(theta, spreads[idx], label=f"{i} m/s")
    else:
        for idx, i in enumerate(n):
            plt.polar(theta, spreads[idx], label=f"{id}={i}")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results/{fileTitle}.png')
    plt.show()
    plt.close()