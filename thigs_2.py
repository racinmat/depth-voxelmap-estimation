

def colormap_plays():
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # Make a figure and axes with dimensions as desired.
    fig = plt.figure(figsize=(3, 8))
    ax1 = fig.add_axes([0.1, 0.1, 0.2, 0.8])

    # Set the colormap and norm to correspond to the data for which
    # the colorbar will be used.
    cmap = mpl.cm.cool
    norm = mpl.colors.Normalize(vmin=5, vmax=10)

    # ColorbarBase derives from ScalarMappable and puts a colorbar
    # in a specified axes, so it has everything needed for a
    # standalone colorbar.  There are many more kwargs, but the
    # following gives a basic continuous colorbar with ticks
    # and labels.
    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                    norm=norm,
                                    orientation='vertical')
    cb1.set_label('Some Units')

    plt.show()

if __name__ == '__main__':
    colormap_plays()