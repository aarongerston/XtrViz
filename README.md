# README #

## Installation ##

1. Create a virtual environment and ensure Git works in it. 
2. Install XtrUtils (required dependency) by running the following command:

`pip install git+https://github.com/aarongerston/XtrUtils.git`
3. Install XtrViz by running the following command:

`pip install git+https://github.com/aarongerston/XtrViz.git`

## Uninstallation

Run: `pip uninstall XtrViz`
    
## Xtrodes Offline Visualizer class example ##

    from XtrViz.plotter import Plotter

    # Prepare some data (pay attention to the axes, samples are on axis=0)
    fs = 4000  # Define sampling rate: 4000Hz
    data = np.random.rand(4000, 16)  # Artificial data set
    time = np.arange(0, len(data))/fs  # Artificial timestamp vector

    # Plot data:
    Plotter.plot_signals(time, data, fs,  # These 3 arguments are required
                         cols=(2, 4, 5, 6),  # Optionally specify columns to plot
                         annotations=None,  # If annotations is defined, plots triggers in separate plot
                         channel_names=('Channel %d' for d in (2, 4, 5, 6)),  # Channel names
                         title='Artificial data',  # Figure title
                         window=(100, 110.5),  # Plot only seconds 100-110.5
                         window_idc=None,  # Can also define window by samples instead of seconds
                         ylim=(-100, 85),  # Specify plot y-limits. Otherwise y-limits are automatically optimized.
                         calib_only=False  # Plots only calibration, if given in annotations.
                        )

    # Plot frequency spectrum of data:
    for col in (3, 4, 5):
        signal = data[:, col]
        Plotter.plot_spectrum(signal, ts, fs,  # These 3 arguments are required
                              fmin=0,  # Minimum frequency to plot
                              fmax=None,  # Maximum frequency to plot. Default: fs/2
                              window=(10, 100),  # Calculate frequency spectrum of signal segment 10sec-100sec
                              window_idc=None,  # Can also define window by samples instead of seconds
                              ax=None,  # Define axis to plot on. If None, plots in new figure window.
                              color=None,  # Define line color, if desired
                              alpha=None,  # Define line alpha, if desired
                              linewidth=2,  # Define line width
                              ylab=True,  # Include y-axis label
                              xlab=True,  # Include x-axis label
                              method='fft',  # Transform method
                              fft_smoothing=1,  # Smoothing factor of spectrum. 1 = No smoothing.
                              fft_window=None,  # FFT windowing method (e.g. 'hamming')
                              perc_overlap=50,  # % overlap of FFT windows
                              window_size=4096,  # Number of samples in each FFT window
                              label=None,  # Line label for legend
                              density=False  # If True, plots PSD instead of power
                             )

    # Plot spectrogram of data
    Plotter.spectrogram(data, ts, fs,  # These 3 arguments are required
                        band=None,  # Min. and max. frequency to plot. Default: (0, fs/2)
                        cols=(0, 13, 14, 15),  # Specify columns to plot
                        window=None,  # Specify time interval to plot, in seconds
                        channel_names=('col0', 'col13', 'col14', 'col15'),  # Column names
                        calib_only=False,  # Plot only calibration? annotations cannot be None.
                       )

    print('Done')