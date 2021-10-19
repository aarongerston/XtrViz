# External dependencies
import time
import numpy as np
import seaborn as sns
import matplotlib as mpl
import scipy.signal as sig
from typing import Union, Any
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import matplotlib.ticker as mtick
import matplotlib.patches as mpatch
import matplotlib.colors as mcolors

# Inter-dependencies
from XtrUtils.utils import Utils

# Modifications
mpl.use('Qt5Agg')


class Plotter(object):

    @staticmethod
    def maximize_fig() -> None:
        """
        Maximizes the most recent figure.
        """

        mng = plt.get_current_fig_manager()
        backend = plt.rcParams['backend']
        if backend == 'tkAgg':
            mng.window.state('zoomed')
        elif backend == 'WxAgg':
            mng.frame.Maximize(True)
        elif backend == 'Qt4Agg' or backend == 'Qt5Agg':
            mng.window.showMaximized()

    @staticmethod
    def plot_signals(ts: np.ndarray,
                     data: np.ndarray,
                     fs: float,
                     cols: Union[list, tuple] = None,
                     annotations = None,
                     channel_names: Union[list, tuple, np.ndarray] = None,
                     title: Union[str, None] = None,
                     window: Union[list, tuple, np.ndarray, None] = None,
                     window_idc: Union[list, tuple, np.ndarray, None] = None,
                     ylim: Union[list, tuple, None] = None,
                     calib_only: bool = False):
        """
        Plots signals in <data> against the timestamp vector. Each channel is a row. If triggers are available, an
        additional plot at the bottom of the figure plots triggers by value and color code.

        The legend can get crowded if there are many triggers. Thus, if only a subset of triggers are relevant, provide them
        in the <triggers> dict. This affects only what triggers are shown on the triggers plot (and corresponding legend).
        <triggers> must be a subset of C.CODES_ALL.

        If it is desired to plot only a subset of the data corresponding to specific triggers, provide the
        relevant triggers in the <show_only> dict. This affects what data is shown in all plots on the figure. <show_only>
        must be a subset of C.CODES_ALL.

        If it is desired to plot only a subset of the data corresponding to a specific time window, provide the start and
        end times in seconds as a 2-item list <window> or the start and end samples as a 2-item list <window_idc>.

        <title> displays text above all plots. No <title>, no text.

        :param ts: 1D np.ndarray timestamp vector.
        :param data: 2D np.ndarray data set. Rows are samples, columns are channels.
        :param fs: sampling rate (Hz).
        :param cols: (OPTIONAL) <data> column names) to plot. Default: all.
        :param channel_names: (OPTIONAL) list of channel names. Must be same length as number of columns in <cols> or
                              number of columns in the dataset if <cols> is None.
        :param annotations: (OPTIONAL) mne.annotations.Annotations list of triggers to plot on triggers plot (below data
                            plots). If None, triggers will not be plotted.
        :param title: (OPTIONAL) text to display above all plots.
        :param window: (OPTIONAL) 2-item list of start and end times (in seconds) to display.
        :param window_idc: (OPTIONAL) 2-item list of start and end indices (samples) to display.
        :param ylim: (OPTIONAL) y-limits to force on plots
        :param calib_only: True or False -- plot only calibration?
        """

        print('Plotting: %s' % title)

        # # Get relevant information
        # cols = aux.validate_channels(data, channels)
        # n_plots = len(cols) + int(bool(triggers))
        # ts = data['Timestamp (seconds)']

        # get start and end times of calibration session if calib_only == True
        # Overridden if <window> or <window_idc> are specified
        if calib_only:
            window_idc = Plotter.get_calib_idc(ts, annotations, window, window_idc)
        start, end = Utils.get_window(ts, window, window_idc)

        # Crop out undesired data columns and segments
        if cols:
            ncols = len(cols)
        else:
            ncols = data.shape[1]
            cols = [c for c in range(ncols)]
        data = data[start:end, cols]
        ts = ts[start:end]

        #
        if channel_names and (len(channel_names) == len(cols)):
            pass
        else:
            channel_names = [f'Column {c}' for c in cols]

        # Figure setup
        n_plots = ncols + int(bool(annotations))
        bottom = 0.1  # 0.05 + 0.05*bool(triggers)
        rect = (0.02, bottom, 0.98, 1)  # L bottom R top
        f, ax = plt.subplots(n_plots, tight_layout={'rect': rect})
        ax = np.atleast_1d(ax)

        # Plot
        n = 0
        for ncol, _ in enumerate(cols):

            # if "Channel" not in col:
            #     continue

            signal = data[:, ncol]
            ax[n].plot(ts, signal, linewidth=0.5, label=channel_names[ncol])
            ax[n].set_ylabel(channel_names[ncol])

            if n != n_plots - 1:
                ax[n].set_xticklabels([])

            n = n + 1

        # Plots only if triggers is not None
        if annotations is not None:
            trig_vec, trig_dict = Utils.get_trig_vec(annotations, ts)
            Plotter.plot_triggers(ts, trig_vec, trig_dict, fs, ax=ax[n_plots - 1])

        # Manually link all axes except for y-axis of trigger plot
        Plotter.link_xy(ax, triggers=annotations)

        # Maximize figure before adjusting sizes/locations
        Plotter.maximize_fig()

        # Other plot adjustments
        if title:
            f.suptitle(title)
        ax[n_plots - 1].set_xlim([min(ts), max(ts)])
        ax[n_plots - 1].set_xlabel('Time (s)')
        # ax[n_plots - 1].xaxis.set_major_formatter(Plotter.x_fmt(ts))
        Plotter.x_fmt(ax[n_plots - 1], ts)
        if Utils.islist(ylim) and (len(ylim) == 2):
            ax[n_plots - 1 - int(annotations is not None)].set_ylim(ylim)
            
    @staticmethod
    def get_calib_idc(ts, annotations, window = None, window_idc = None):
        if window is not None:
            print('Ignoring <calib_only> parameter since you specified <window>...')
        elif window_idc is not None:
            print('Ignoring <calib_only> parameter since you specified <window_idc>...')
        elif annotations is None:
            print('Cannot find calibration without annotations. Ignoring <calib_only> parameter...')
        else:
            calib_idc = [idx for idx, evt in enumerate(annotations.description) if 'Calibration step' in evt]
            calib_start = annotations.onset[calib_idc[0]]
            calib_end = annotations.onset[calib_idc[-1]]
            window_idc = (np.argmin(np.abs(ts - calib_start)), np.argmin(np.abs(ts - calib_end)))
            
        return window_idc

    @staticmethod
    def x_fmt(ax: plt.Axes, ts: np.ndarray) -> mtick.FuncFormatter:
        """
        Formats plot x-axis as 'mm:ss' or 'hh:mm:ss', depending on signal length.

        Example:
        >> ts = data['Timestamp (seconds)']
        >> ax = plt.gca()
        >> x_fmt(ax, ts)
        >> # OLD: ax.xaxis.set_major_formatter(x_fmt(ts))

        :param ax: axis whose x-axis to update.
        :param ts: timestamp vector.
        :return: mtick.FuncFormatter in the desired format.
        """

        if ts[-1] > 60 * 60:
            fmt = mtick.FuncFormatter(lambda sec, x: time.strftime('%H:%M:%S', time.gmtime(sec)))
        else:
            fmt = mtick.FuncFormatter(lambda sec, x: time.strftime('%M:%S', time.gmtime(sec)))

        ax.xaxis.set_major_formatter(fmt)

    @staticmethod
    def get_trig_from_value(value, trig_dict):

        if trig_dict == {}:
            return None

        if value == 0:
            return None

        return list(trig_dict.keys())[list(trig_dict.values()).index(value)]

    @staticmethod
    def plot_triggers(ts: np.ndarray,
                      trig_vec: np.ndarray,
                      trig_dict: dict,
                      fs: float,
                      ax: plt.Axes,
                      legend=True) \
            -> None:
        """
        Plots triggers by value and color on axis <ax>.

        :param ts: timestamp vector in seconds.
        :param trig_vec: trigger vector (same length as ts).
        :param trig_dict: dict mapping trigger events to unique numerics.
        :param fs: sampling rate (Hz).
        :param ax: axis to plot on.
        :param legend: (OPTIONAL) show legend, yes or no? Default: yes.
        """

        # Determine if any discontinuities in data
        pauses = Utils.find_breaks(ts, greater_than=2/fs)
        trig_vec[pauses + 1] = 0

        # Plot triggers
        ax.step(ts, trig_vec, color='black')
        ax.set_ylabel('Trigger')

        # Calculate and set y-lims
        possible_trigs = np.unique(trig_vec)
        max_possible_trigs = max(possible_trigs)
        min_possible_trigs = min(possible_trigs)
        ymax = 0 if max_possible_trigs == 0 else max_possible_trigs + 1
        ymin = 0 if min_possible_trigs == 0 else min_possible_trigs - 1
        ax.set_ylim(bottom=ymin, top=ymax)

        # Colorize triggers by value
        trig_idx = np.where(np.diff(trig_vec))[0]
        if trig_vec[0] != 0:
            trig_idx = np.insert(trig_idx, 0, 0)

        in_legend = []
        for n, idx in enumerate(trig_idx):

            # If trigger is on the last sample, do nothing
            if idx == len(ts):
                continue

            # Otherwise, color from trigger onset until subsequent trigger or end of signal
            trig_val = trig_vec[idx + 1]
            trig = Plotter.get_trig_from_value(trig_val, trig_dict)
            trig_start = ts[trig_idx[n]]
            trig_end = ts[trig_idx[n + 1]] if n != len(trig_idx) - 1 else ts[-1]
            label = trig if trig not in in_legend else ''
            in_legend.append(label)
            patch = mpatch.Rectangle((trig_start, 0), width=trig_end - trig_start, height=trig_val,
                                     color=Plotter.get_trig_color(trig_val, trig_dict), alpha=0.5, label=label)
            ax.add_patch(patch)

        # Adjust x-lims and x-labels
        ax.set_xlim(xmin=min(ts), xmax=max(ts))
        ax.set_xlabel('Time (s)')
        Plotter.x_fmt(ax, ts)
        # ax.xaxis.set_major_formatter(x_fmt(ts))

        # Add legend
        if legend:
            h, l = ax.get_legend_handles_labels()
            Plotter.add_trig_legend(ax.figure, l, h)

    @staticmethod
    def add_trig_legend(parent: Union[plt.Axes, plt.Figure], labels: list, handles: list = None, y_pos: float = None) \
            -> None:
        """
        Creates a legend for the trigger plot.

        :param parent: defines where to put the legend.
        :param labels: list of strings to use as legend values.
        :param handles: (OPTIONAL) patch or line handles to associate with <labels>.
        :param y_pos: (OPTIONAL) y position of the legend on the figure.
        """

        labels = [lab for lab in labels if lab != '']
        if not labels:
            return
        n_possible_trigs = len(set(labels))
        n_rows = int(np.ceil(n_possible_trigs / 6))
        n_cols = int(np.ceil(n_possible_trigs / n_rows))
        if handles:
            if y_pos is None:
                y_pos = 0.01
            parent.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, y_pos), fancybox=True, ncol=n_cols)
        else:
            if y_pos is None:
                y_pos = 0.04 if n_possible_trigs < 9 else 0.04
            parent.legend(labels, loc='upper center', bbox_to_anchor=(0.5, y_pos), fancybox=True, ncol=n_cols)

    @staticmethod
    def link_xy(ax: Union[np.ndarray, list, tuple], link: str = 'xy', triggers: Any = False) -> None:
        """
        Links plots in <ax> list of plt.Axes objects, by x, y, or both.

        :param ax: list of plt.Axes objects to link.
        :param link: defines which axes to link. Must be in: ('x', 'y', 'xy').
        :param triggers: is the last plot in the <ax> list a trigger plot? If so, its y-axis will not be linked with the
                         other plots, even if <link> is 'y' or 'xy'.
        """

        if link not in ['x', 'y', 'xy']:
            print('<link> parameter must be one of: [''x'', ''y'', ''xy'']')

        triggers = triggers not in (None, False)

        linkx = True
        linky = True

        if 'y' not in link:
            linky = False
        if 'x' not in link:
            linkx = False

        ax = np.array(ax).flatten()

        # Get min and max x- and y-limits form all plots to be linked
        last_plot_idx = len(ax) - int(triggers)  # - 1
        valid_plots = ax[:last_plot_idx] if last_plot_idx > 0 else [ax[0]]
        try:
            min_y = min([a.get_ylim()[0] for a in valid_plots])
            max_y = max([a.get_ylim()[1] for a in valid_plots])
            min_x = min(min([[min(l.get_xdata()) for l in a.lines if l.get_xdata() != []] for a in valid_plots]))
            max_x = max(max([[max(l.get_xdata()) for l in a.lines if l.get_xdata() != []] for a in valid_plots]))
        except:
            min_x = min([min(plot.get_xlim()) for plot in valid_plots])
            max_x = max([max(plot.get_xlim()) for plot in valid_plots])
            min_y = min([min(plot.get_ylim()) for plot in valid_plots])
            max_y = max([max(plot.get_ylim()) for plot in valid_plots])

        # Manually link all axes except for y-axis of trigger plot
        ax1 = ax[0]
        n_plots = len(ax)
        for n, a in enumerate(ax):
            if linkx:
                ax1._shared_x_axes.join(ax1, a)
            if linky and not (triggers and n == n_plots - 1):  # Don't link y-axis of trigger plot
                ax1._shared_y_axes.join(ax1, a)
        if linkx:
            ax1.set_xlim((min_x, max_x))
        if linky:
            ax1.set_ylim((min_y, max_y))

    @staticmethod
    def plot_spectrum(signal: np.ndarray,
                      ts: np.ndarray,
                      fs: float,
                      fmin: float = 0,
                      fmax: float = None,
                      fstep: float = 0.05,
                      window: Union[list, tuple] = None,
                      window_idc: Union[list, tuple] = None,
                      ax: plt.Axes = None,
                      color: str = None,
                      alpha: float = None,
                      linewidth: float = 2,
                      ylab: bool = True,
                      xlab: bool = True,
                      method: str = 'fft',
                      fft_smoothing: int = 1,
                      fft_window: Union[str, None] = None,
                      perc_overlap: float = 50,
                      window_size=4096,
                      label: Union[str, None] = None,
                      density: bool = False):
        """
        Plots power spectrum of <signal>.

        If <signal> contains nans, Lomb-Scargle is used in place of Welch to estimate power spectrum. For this reason,
        plot_spectrum() insists on a time series vector <ts> as well, to calculate the Lomb-Scargle power spectrum.

        <fmin> and <fmax> optionally specify the minimum and maximum frequencies to analyze and plot. <ftep> is the step
        size between frequency bins, and is relevant only in the case of Lomb-Scargle.

        If <ax>, a plt.Axes object, is given, the plot will be plotted on that axis. Otherwise, on a new plt.Figure.

        <color>, <alpha>, and <linewidth> optionally specify characteristics of the line plotted.

        <xlab> and <ylab> are booleans that specify whether to include x- and y-labels, respectively.

        :param signal: 1D signal.
        :param ts: 1D signal of timestamps corresponding to <signal>.
        :param fs: sampling rate (Hz).
        :param fmin: (OPTIONAL) minimum frequency to analyze and plot. Default: 0 (or minimum viable frequency).
        :param fmax: (OPTIONAL) maximum frequency to analyze and plot. Default: Fs/2.
        :param fstep: (OPTIONAL) step size between frequencies analyzed. Relevant only in case of Lomb-Scargle.
        :param ax: (OPTIONAL) plt.Axes object on which to plot on. If not given, plot opens in new plt.Figure.
        :param color: (OPTIONAL) line color.
        :param alpha: (OPTIONAL) line alpha.
        :param linewidth: (OPTIONAL) line width.
        :param ylab: (OPTIONAL) bool: should a y-label with units be added?
        :param xlab: (OPTIONAL) bool: should an x-label ('Frequency (Hz)') be added?
        :param method: (OPTIONAL) 'fft' or 'welch' or 'ls' (Lomb-Scargle)
        :param fft_smoothing: (OPTIONAL) size of window to smooth FFT. Only relevant if <method> = 'fft'.
                              Default: 1 (no smoothing).
        :param fft_window: (OPTIONAL) 'hamming' (average Hamming windows of <window_size> length) or None (perform FFT once
                           over whole signal; no averaging).
        :param perc_overlap: (OPTIONAL) percent overlap for FFT (if <fft_window> = 'hamming') and Welch estimations.
        :param window_size: (OPTIONAL) size of window to use for FFT (if <fft_window> = 'hamming') and Welch estimations.
                            Default: 50%.
        :param label: (OPTIONAL) label for legend. If None, no legend.
        """

        start, end = Utils.get_window(ts, window, window_idc)
        ts = ts[start:end]
        signal = signal[start:end]

        perc_overlap = 100 - perc_overlap

        non_nans = ~np.isnan(signal)
        non_nans_idc = np.where(non_nans)[0]
        if any(np.where(np.unique(np.diff(non_nans_idc)) != 1)) | (method == 'ls'):  # if there are nans, use Lomb-Scargle
            if fmin is None:
                fmin = fstep
            if fmax is None:
                fmax = fs / 2
            f = np.linspace(start=fstep if fmin == 0 else fmin, stop=fmax, num=int(fmax / fstep), endpoint=True)
            pxx = sig.lombscargle(ts.iloc[non_nans_idc], signal[non_nans], 2 * np.pi * f)
            if density:
                pxx = pxx / len(pxx)
                if ylab:
                    ylab = 'LS PSD (uV^2/Hz'
            elif ylab:
                ylab = 'LS power (uV^2)'
        else:  # if no nans, estimate psd with Welch or FFT
            if method == 'welch':
                scaling = 'density' if density else 'power'
                noverlap = int(window_size * perc_overlap / 100)
                if density:
                    scaling = 'density'
                [f, pxx] = sig.welch(signal, fs, window='hamming', nperseg=window_size,
                                     noverlap=noverlap, detrend=False, nfft=window_size,
                                     scaling=scaling, axis=-1, average='mean')
                if ylab:
                    ylab = 'Welch PSD (uV^2/Hz)' if density else 'Welch PSD (uV^2)'
            elif method == 'fft':
                if fft_window == 'hamming':
                    hamming_window = np.hamming(window_size)
                    noverlap = int(window_size * perc_overlap / 100)
                    n_windows = len(signal) // noverlap
                    windows = np.ndarray((n_windows, window_size), dtype=signal.dtype)
                    for i in range(n_windows):
                        head = int(i * noverlap)
                        tail = int(head + window_size)
                        windows[i] = signal[head:tail] * hamming_window
                    pxx = np.fft.rfft(windows, axis=-1)
                    pxx = np.mean(np.abs(pxx), axis=0)
                    f = np.fft.rfftfreq(len(hamming_window), 1 / fs)
                else:
                    pxx = np.fft.rfft(signal)
                    pxx = np.abs(pxx * pxx.conj()) / (len(signal) / 2)
                    f = np.fft.rfftfreq(len(signal), 1 / fs)  # *2*np.pi*C.FS

                # FFT -> PSD if <density> is True
                pxx = pxx / len(pxx) if density else pxx

                # Get frequencies from pxx
                idc = np.argsort(f)
                f = f[idc]
                pxx = pxx[idc]

                # Smooth result
                N = int(fft_smoothing)
                try:
                    pxx = np.convolve(pxx, np.ones(N) / N, mode='same')
                except Exception as e:
                    print(str(e))

                # Assign y-label
                if ylab:
                    ylab = 'PSD (uV^2/Hz)' if density else 'FFT (uV^2)'

            # Crop to desired frequencies
            fmin_idx = np.where(f > fmin)[0][0]
            if fmax is not None:
                fmax_idx = np.where(f < fmax)[0][-1]
            else:
                fmax_idx = len(f)
            f = f[fmin_idx:fmax_idx]
            pxx = pxx[fmin_idx:fmax_idx]

        # Plot
        if ax is None:
            plt.figure()
            plt.plot(f, pxx, color=color, alpha=alpha, linewidth=linewidth, label=label)
            if ylab:
                plt.ylabel(ylab)
            if xlab:
                plt.xlabel('Frequency (Hz)')
            plt.xlim(fmin, fmax)
            plt.gca().set_ylim(0, plt.gca().get_ylim()[1])
        else:
            ax.plot(f, pxx, color=color, alpha=alpha, linewidth=linewidth, label=label)
            if ylab:
                ax.set_ylabel(ylab)
            if xlab:
                ax.set_xlabel('Frequency (Hz)')
            ax.autoscale()
            ax.set_xlim(fmin, fmax)
            ax.set_ylim(0, ax.get_ylim()[1])
            plt.draw()

        # plt.show(block=False)
        plt.legend()
        Plotter.maximize_fig()

    @staticmethod
    def add_patches(ts: np.ndarray,
                    onsets: Union[list, np.ndarray, tuple, int],
                    offsets: Union[list, np.ndarray, tuple, int],
                    ax: plt.Axes,
                    color='red',
                    label=None,
                    height=99999,
                    alpha=0.25) \
            -> list:
        """
        Adds patches to plot <ax>, defined by <onsets> and <offsets> indices or lists of indices where the patches should
        begin and end, respectively.

        :param ts: full 1D timestamp vector of data.
        :param onsets: index or list of indices of patch onsets.
        :param offsets: index or list of indices of patch offsets.
        :param ax: plt.Axes object on which to plot patches.
        :param C: Constants.
        :param color: (OPTIONAL) color.
        :param label: (OPTIONAL) text for legend.
        :param height: (OPTIONAL) height of patch. If specified, the patch will extend from -1*<height> to <height>.
        :param alpha: (OPTIONAL) patch opacity.
        :return: list of mpatch.Rectangle patch objects.
        """

        if not Utils.islist(onsets):
            onsets = [onsets]
        if not Utils.islist(offsets):
            offsets = [offsets]
        if len(onsets) != len(offsets):
            print('Failed add patches. Must have same number of onsets and offsets.')
            return []

        yl = ax.get_ylim()

        patches = []
        for onset, offset in zip(onsets, offsets):
            patch = mpatch.Rectangle((ts[onset], -1 * height), width=ts[offset] - ts[onset], height=2 * height,
                                     color=color, alpha=alpha, label=label)
            patches.append(patch)
            ax.add_patch(patch)
            label = None

        ax.set_ylim(yl)

        return patches

    @staticmethod
    def get_trig_color(trig_val: Union[int, str], trig_dict: dict) -> Union[str, None]:
        """
        Returns the color assigned to trigger with value <trig_val>.

        :param trig_val: value assigned to trigger according to C.CODES_ALL.
        :param C: Constants.
        :return: Matplotlib colormap color.
        """

        if not trig_dict:
            print('Must provide valid trig_dict.')
            return None

        if trig_val == 0:
            return None

        # trig_str = Plotter.get_trig_from_value(trig_val, trig_dict)
        try:
            possible_vals = [val for val in list(trig_dict.values()) if val != 0]
            n_color = possible_vals.index(trig_val)
        except ValueError:
            try:
                possible_vals = [val for val in list(trig_dict.keys()) if val != 0]
                n_color = possible_vals.index(trig_val)
            except ValueError:
                return None

        # Convert Seaborn to matplotlib colormap
        rgb2hex = lambda x: '#{:02x}{:02x}{:02x}'.format(x[0], x[1], x[2])
        n_possible_trigs = len(possible_vals)

        # seaborn way:
        colors = sns.color_palette('bright', n_colors=n_possible_trigs, as_cmap=False)
        hex = [rgb2hex([int(c * 255) for c in color]) for color in colors]
        cmap = mcolors.ListedColormap(hex)
        color = cmap.colors[n_color]

        # # matplotlib way:
        # cmap = plt.get_cmap('gist_rainbow')
        # color = rgb2hex([int(c * 255) for c in cmap(n_color / n_possible_trigs)[:-1]])

        return color

    @staticmethod
    def spectrogram(data: np.ndarray,
                    ts: np.ndarray,
                    fs: float,
                    band: Union[tuple, list] = None,
                    cols: Union[tuple, list] = None,
                    window: Union[tuple, list] = None,
                    channel_names: Union[tuple, list] = None,
                    calib_only: bool = False,
                    annotations=None):
        """
        Plots separate spectrogram for each channel of <data>.

        :param data: 2D np.ndarray data set. Rows are samples, columns are channels.
        :param ts: 1D np.ndarray timestamp vector.
        :param fs: Sampling rate (Hz).
        :param band: 2-element array of floats defining the frequency band of interest.
        :param cols: (OPTIONAL) list of <data> columns to plot. Default: all columns.
        :param window: (OPTIONAL) 2-item list of start and end times (in seconds) to display.
        :param channel_names: (OPTIONAL) list of channel names. Must be same length as number of columns in <cols> or
                              number of columns in the dataset if <cols> is None.
        :param calib_only: (OPTIONAL) True or False -- plot only calibration? Default: False.
        :param annotations: (OPTIONAL) mne.annotations.Annotations list of triggers. Required if <calib_only> is True.
        """

        if calib_only:
            if annotations is None:
                print('Annotations required.')
                return
            window_idc = Plotter.get_calib_idc(ts, annotations, window)
        else:
            window_idc = None
        start, end = Utils.get_window(ts, window, window_idc)

        if cols:
            channels = cols
        else:
            channels = [col for col in range(data.shape[1])]
            
        if not channel_names or len(channel_names) != len(channels):
            channel_names = [f'Column {col}' for col in channels]

        # Spectrogram parameters
        window_size = 4096
        perc_overlap = 50
        noverlap = int(window_size * perc_overlap / 100)

        f, ax = plt.subplots(len(channels), tight_layout={'rect': (0.02, 0.02, 0.98, 1)})
        Plotter.maximize_fig()
        ax = np.atleast_1d(ax)

        for nch, ch in enumerate(channels):
            f, t, Sxx = sig.spectrogram(data[start:end, ch], fs=fs, window='hamming', nperseg=window_size,
                                        noverlap=noverlap, nfft=None, scaling='density')
            if band:
                band = (np.argwhere(f > band[0])[0][0], np.argwhere(f < band[1])[-1][0])
                f = f[band[0]:band[1]]
                Sxx = Sxx[band[0]:band[1]]

            if len(t) > 1000:
                Sxx = sig.decimate(Sxx, q=int(len(t) / 1000))
                t = sig.decimate(t, q=int(len(t) / 1000))
            # t = np.arange(ts.iloc[start_idx], ts.iloc[end_idx], (ts.iloc[end_idx] - ts.iloc[start_idx]) / len(t))
            t = t + window[0]

            # Plot
            ax[nch].pcolormesh(t, f, np.log(Sxx), shading='gouraud', norm=mpc.Normalize(0, 5))
            ax[nch].set_ylabel(channel_names[nch])
            if nch == len(channels) -1:
                Plotter.x_fmt(ax[nch], t)
            else:
                ax[nch].set_xticks(())
