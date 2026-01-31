# Copyright (C) Meridian Innovation Ltd. Hong Kong, 2020. All rights reserved.

import time
import os
import logging
import math
import itertools
from functools import partial
from pathlib import Path
import operator
import numpy as np
import cv2 as cv
from serial.tools import list_ports
from serial import Serial, SerialException
from senxor.mi48 import MI48
from senxor.interfaces import MI_VID, MI_PIDs, USB_Interface
from senxor.colormaps import colormaps

def connect_senxor(src=None, name=None):
    """
    Return an MI48 instance corresponding to the SenXor module connected to `src`

    `src` can be either the name of a virtual comport, e.g. COM6, or a sequential
    number, e.g. 0, 1, etc.
    if `name` (stirng) is not None, it will be assigned to mi48.name instance, else
    the name of the virtual comport will be assigned to the mi48.name.

    Return None, if no connection to SenXor can be established.
    """
    cam_index, port_name = None, None
    try:
        src = int(src)
        cam_index = src
    except ValueError:
        port_name = src.upper()
    except TypeError:
        pass
    mi48 = None
    connected_port = None
    port_names = []
    for p in list_ports.comports():
        if p.vid == MI_VID and p.pid in MI_PIDs:
            port = p.description.split()[-1][1:-1]
            port_names.append(port)
            if port_name is not None and port_name != port: continue
            if cam_index is not None and cam_index != len(port_names)-1: continue
            try:
                ser = Serial(p.device)
            except SerialException:
                # port already open
                if port_name is not None:
                    logging.warning(f'{port_name} seems already open')
                if cam_index is not None:
                    logging.warning(f'Thermal image source {cam_index}'
                                     ' seems already open')
                continue
            usb = USB_Interface(ser)
            connected_port = port
            if name is None: name = connected_port
            mi48 = MI48([usb,usb], name=name, read_raw=False)
    return mi48, connected_port, port_names

def data_to_frame(data, array_shape, hflip=False):
    """
    Convert 1D array into nH x nV 2D array corresponding to the FPA.

    Use this func to change orientation to forward looking camera with `hflip`.
    """
    # Note that the data coming for the EVK is stored as a 1D array.
    # the data.reshape() reconstructs the 2D FPA array shape; 
    # Note the data ordering is 'F' (fortran-like).
    nc, nr = array_shape
    if hflip:
        # The flipping below realises horisontal flip, assuming that 
        # the USB port faces the ceiling or the sky, to correct for 
        # left/right flip in the camera, if necessary.
        frame = np.flip(data.reshape(array_shape, order='F').T, 1)
    else:
        frame = data.reshape(array_shape, order='F').T
    return frame.copy()

def remap(data, new_range=(0, 255), curr_range=None, to_uint8=True):
    """
    Remap data from one range to another; return float16.

    This function is critical for working with temperature data and
    maintaining accuracy.

    The mapping is a linear transformation:

        l1, h1 = curr_range

        l2, h2 = new_range

        x = (data - l1) / (h1 - l1)

        out = l2 + x * (h2 - l1)

    If `curr_range` is not specified, assume it is defined by the data limits.
    If `to_uint8` is true, return an uint8, instead of float16. This is
    useful in conjuction with `new_range` being (0, 255), to prepare for
    many OpneCV routines which accept only uint8.
    """
    lo2, hi2 = new_range
    #
    if curr_range is None:
        lo1 = np.min(data)
        hi1 = np.max(data)
    else:
        lo1, hi1 = curr_range
    #
    # The relpos below represents the relative position of _data in the
    # current range.
    # We could potentially manipulate relpos by some function to 
    # realise non-linear remapping
    relpos = (data - lo1) / float(hi1 - lo1)
    out = lo2 + relpos * (hi2 - lo2)
    #
    if to_uint8:
        return out.astype('uint8')
    else:
        return out.astype('float16')

def get_default_outfile(src_id=None, ext='csv'):
    """Yield a timestamped filename with specified extension."""
    ts = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    if src_id is not None:
        filename = "{}-{}.{}".format(src_id, ts, ext)
    else:
        filename = "{}.{}".format(ts, ext)
    return filename


#fib = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 600]
def get_colormap(colormap='rainbow2', nc=None):
    """
    Return a 256-color LUT corresponding to `colormap`.

    `colormap` is either from open cv or explicitly defined above.
    If `nc` is not None, return a quantized colormap with `nc` different colors.
    """
    try:
        # use defualt opencv maps or explicitly defined above
        cmap = colormaps[colormap]
    except KeyError:
        cmap = colormaps["rainbow2"]
    if nc is not None:
        # some names appear in OpenCV (int) and custom LUTs
        if isinstance(cmap, int):
            # return non-quantized CV cmap
            return cmap
        # we need to create a LUT with 256 entries, and these entries
        # are indexes in the actual color map; there are `nc` such indexes
        nmax = 256
        # number of indexes per color
        ipc = nmax // nc
        # If nmax is not multiple of nc, then we have to patch up the LUT.
        # Below, we choose to patch up with the highest index
        delta = nmax % nc
        lut = [int((j // ipc) / (nc-1) * (nmax-1)) for j in range(nmax-delta)]
        lut += [nmax-1,] * delta
        cmap = np.array([cmap[i] for i in lut], dtype='uint8')
    return cmap


def cv_render(data, title='', resize=(800, 620), colormap='jet',
              interpolation=cv.INTER_CUBIC, display=True, n_colors=None):
    """
    Render and display a 2D numpy array data of type uint8, using OpenCV.
    
    Color the image using any of the supported OpenCV colormaps.
    Resize the image, ensuring the aspect ratio is maintained.
    Use cubic interpolation when upsizing.
    
    If `display` is true, render the image in an OpenCV-controled window.
    Else, return the OpenCV image object.
    """
    # colormap may be either a colormap list or a string
    cmap = get_colormap(colormap, n_colors)
    cvcol = cv.applyColorMap(data, cmap)
    if isinstance(resize, tuple) or isinstance(resize, list):
        cvresize =  cv.resize(cvcol, dsize=resize,
                            interpolation=interpolation)
    else:
        cvresize =  cv.resize(cvcol, dsize=None, fx=resize, fy=resize,
                            interpolation=interpolation)
    if display:
        cv.imshow(title, cvresize)
    return cvresize

def cv_filter(data, parameters=None, use_median=True, use_bilat=True,
                     use_nlm=False):
    """
    Spatial filtering based on a sequence of Meidan, Bilateral and NLM.

    Requires uint8 and returns uint8 data.

    For best results, set both Bilateral and Non-Local Means
    flags to true.

    See OpenCV for significance and values of the optional parameters
    """
    # default median filter parameters
    par = {'blur_ks': 5}
    # default bilateral filter parameters
    par.update({'d': 7, 'sigmaColor': 23, 'sigmaSpace': 23})
    # default nlmeans filter parameters
    par.update({'h': 5, 'templateWindowSize': 5, 'searchWindowSize': 11})
    # update parameters from caller
    if parameters is not None:
        par.update(parameters)
    # do the filtering
    filtered = data
    if use_median:
        #t0 = time.time()
        filtered = cv.medianBlur(filtered, par['blur_ks'])
        #print('Median cost [ms]: {:8.4f}'.format(time.time() - t0))
    if use_bilat:
        #t0 = time.time()
        filtered = cv.bilateralFilter(filtered, d=par.get('d'),
                                    sigmaColor=par.get('sigmaColor'),
                                    sigmaSpace=par.get('sigmaSpace'))
        #print('Bilateral cost [ms]: {:8.4f}'.format(time.time() - t0))

    # nlmeans works only on uint8; the result is also uint8
    # so we must normalise and then renormalise
    if use_nlm:
        t0 = time.time()
        filtered = cv.fastNlMeansDenoising(filtered, None, h=par.get('h'),
                        templateWindowSize=par.get('templateWindowSize'),
                        searchWindowSize=par.get('searchWindowSize'))
        print('NLMeans cost [ms]: {:8.4f}'.format(time.time() - t0))
    return filtered

def clip_frame(frame, minval=None, maxval=None, c0=0.0, c1=0.0):
    """
    Clip the lowest and highest of the `frame`.

    The output range is shrunk from top and bottom by a fraction --
    `c0` and `c1` respectively, of the temperature range, where the temperature
    range is defined ither from the intrinsic frame range, or, by the `minval`
    and `maxval` if provided.
    """
    try:
        _range  = maxval - minval
    except TypeError:
        minval, maxval = frame.min(), frame.max()
        _range  = maxval - minval
    m0 = minval + c0 * _range
    m1 = maxval - c1 * _range
    return np.clip(frame, m0, m1)


class TrueAverageFilter:

    def __init__(self, depth):
        self.depth = depth
        self.buf = np.zeros(shape=(depth,62,80))
        self.counter = 0
        self.ix = 0
        self.av = 0

    def update(self, new):
        if self.counter < self.depth: self.counter += 1
        self.buf[self.ix] = new
        self.av = np.sum(self.buf, axis=0)
        self.av = self.av / self.counter
        self.ix += 1
        if self.ix > self.depth - 1: self.ix = 0
        return self.av

    def __call__(self, new):
        return self.update(new)


class RollingAverageFilter:

    def __init__(self, N=4):
        """
        Rolling average filter over ``N`` frames.

        Usage:

            # establish rolling average over 20 frames
            min_filter = RollingAverageFilter(N=20)
            ...

            min_temp = min_filter(measured_min)
        """
        self.N = N
        self.count = 0
        self.av = 0
        self.update = self._update_0

    def _update_0(self, new):
        self.count += 1
        self.av += 1. / self.count * (new - self.av)
        if not self.count < self.N:
            self.update = self._update_1

    def _update_1(self, new):
        self.av += 1. / self.N * (new - self.av)

    def clear(self):
        self.__init__(self.N)

    def __call__(self, new):
        self.update(new)
        return self.av


class FibonacciAverageFilter:

    fib = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55]

    def __init__(self, initial, N=6, i_start=1):
        """
        Fibonacci-weighted average filter over ``N`` frames.

        Usage:

            # establish rolling average over 20 frames
            min_filter = RollingAverageFilter(initial_min, N=20)
            ...

            min_temp = min_filter(measured_min)
        """
        self.N = N
        self.s = i_start
        self.p = i_start + N
        self.frames = [initial] * self.N
        w = np.array(self.fib[self.s: self.p])
        self.weights = np.asarray(w)/np.sum(w)

    def __call__(self, new):
        """Update the rolling average estimate"""
        self.frames = self.frames[1:] + [new]
        #print(len(self.frames))
        self.frames[-1] = np.sum([w*f for w,f in
                            zip(self.weights, self.frames)], axis=0)

        #print(self.frames[-1].shape)
        return self.frames[-1]


class KeyboardHandler:
    """
    Add a handler for a specific key to allow interactive change of
    parameters which are stored in a dictionary.
    """
    def __init__(self, pardict):
        """
        Keyboard handler that allows for interactive change of parameters.

        pardict (dict): a dictionary with parameter key-val pairs to be
                        affected
        """
        self.parameters = pardict
        self.actions = {}
        self.triggers = {}

    def register(self, key, parname, action='toggle', bounds=None, trigger=None):
        """
        Associate a key-press with certain modification of a parameter.

        Parameters:

            key: key (toggle) or a pair of keys (increment/decrement)
            parname:  name of parameter from ``pardict`` to be affected
            action: 'toggle' by default; if integer, then treat as delta
                    by which to increment/decrement by pair of keys
            bounds: limits for increment/decrement action
            trigger: a function that is passed the updated parameter value
        """
        if action=='toggle':
            self.actions[key] = (parname, partial(operator.not_), bounds)
        else:
            delta = -action
            self.actions[key[0]] = (parname, partial(operator.add, delta), bounds)
            delta = action
            self.actions[key[1]] = (parname, partial(operator.add, delta), bounds)
        if trigger is not None:
            try:
                for k in key:
                    self.triggers[k] = partial(trigger[0], *trigger[1:])
            except AttributeError:
                    self.triggers[key] = partial(trigger[0], *trigger[1:])

    def __call__(self, key):
        """Execute the action associated with pressing of the given key"""
        try:
            pname, func, bounds = self.actions[key]
            pval = self.parameters[pname]
            newval = func(pval)
            if bounds is None:
                self.parameters[pname] = newval
            else:
                self.parameters[pname] = newval
                if newval < bounds[0]:
                    self.parameters[pname] = bounds[0]
                if newval > bounds[1]:
                    self.parameters[pname] = bounds[1]
            # do whatever requested after the parameter update
            try:
                self.triggers[key](self.parameters[pname])
            except KeyError:
                # no further action triggered by the pressed key
                pass
        except KeyError:
            # wrong key pressed
            pass


class TestData:
    nc, nr = 80, 62
    nh = 80
    def __init__(self):
        """Create a dictionary to store all data.

        The dictionary key is decided upon adding items.
        When adding data, we can pass either a tupple (Vdd, Tsx, Frames), or
        a 2D array of shape N_frames, n_header+n_col*n_rows.
        In the latter case, the assumption is that header[:, 1] and header[:, 2]
        are Vdd and Tsx. These are parsed to produce the correct units (V, and degC)
        (Vdd, Tsx, frame) is the stored dictionary value.
        """
        self.data = {}

    def update(self, key, data):
        """Add data as a tupple (Vdd, Tsx, Frames) or a 2D array from np.loadtxt"""
        try:
            Vdd, Tsx, frames = data
        except ValueError:
            frames = data[:, -self.nc * self.nr:]
            Vdd = data[:, 2]   # * 1.e-4
            Tsx = data[:, 3]   # 100 + KELVIN0
        self.data[key] = Vdd, Tsx, frames

    def get(self, key):
        """Retrieve data for a given key"""
        return self.data[key]


def quick_segment(data, param=None):
    """
    Perform a quick hot-on-cold segmentation and return the contour lines,
    the mask of the hot contours, and their statistics, as a 3-tuple.

    Input `data` must be a 2D frame.
    Parameters control the bilateral filtering, median bluring for
    smoothing the contours, and how to do the hot-on-cold thresholding.
    """
    p = {
        'use_bilat': True,
        'bilat_d': 7,
        'bilat_sigmaColor': 23,
        'bilat_sigmaSpace': 23,
        #
        'use_median': True,
        'median_ksize': 3,
        #
        'thresholding': 'adaptive',  # or Otsu?
        'adaptth_blockSize': 97,
        'adaptth_C': -29,
        #
        'contour_minArea': -9,
    }
    if param is not None: p.update(param)
    img = data.copy()
    if p['use_bilat']:
        # reduce noise
        img = cv.bilateralFilter(img.astype(np.float32), d=p['bilat_d'],
                                  sigmaColor=p['bilat_sigmaColor'],
                                  sigmaSpace=p['bilat_sigmaSpace'])
    # abserr = np.abs(bilat - bilat.mean())
    # segment the image
    img = cv.adaptiveThreshold(remap(img), 1, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv.THRESH_BINARY,
                                blockSize=p['adaptth_blockSize'],
                                C=p['adaptth_C'])
    if p['use_median']:
        # smooth the contours of the segments
        img = cv.medianBlur(img, ksize=p['median_ksize'])

    # extract the contours and return their line, mask and statistics
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cntr_stats = get_contour_stats(data, contours, minArea=p['contour_minArea'])
    contours, masks, cntrstats = zip(*cntr_stats)
    return contours, masks, cntrstats


def get_contour_stats(data, contours, minArea=None, min_sdev=None,
                      mean_range=None, sortby='mean'):
    """
    Return a list of tupples: [(contour, mask, metrics)].

    The 'metrics' is a dictionary with the following keys:
        centroid, area, mean, median, sdev, min, max, spread,
        center, center_9, center_5.
        
    The 'area' is with a sign: negative = hot on cold; positive = cold on hot.
    The center, center_9 and center_5 are the values of the pixel at 
    the 'centroid' coordinates, or the mean of the 5 or 9 pixels centered
    at the 'centroid' coordinates of the frame.
    """
    output = []
    # first identify warm-on-cold contours that have at least
    # minArea number of pixels
    for i, c in enumerate(contours):
        # rough estimate of area with sign, to distinguish b/w
        # hot on cold (-ve) and cold on hot (+ve)
        area = cv.contourArea(c, oriented=True)
        # work only with hot on cold contours, assuming hot is foreground
        if minArea is None or area < minArea:
            # create a filled mask for the current contour
            mask = np.zeros(data.shape, dtype='uint8')
            cv.drawContours(mask, contours, i, 1, cv.FILLED)
            # get some metrics of the data within the current contour
            metrics = {}
            M = cv.moments(c)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            centroid = (cx, cy)
            centre_9_ix_iy = np.array([[cx+i, cy+j] for i in [-1, 0, +1] for j in [-1, 0, +1]], ndmin=2, dtype=int)
            centre_5_ix_iy = np.array([[cx-1, cy], [cx, cy], [cx+1, cy], [cx, cy-1], [cx, cy+1], ], ndmin=2, dtype=int)
            metrics['centroid'] = centroid
            metrics['area'] = int(math.copysign(len( mask[mask != 0]), area))
            metrics['mean'] = data[mask != 0].mean()
            metrics['median'] = np.median(data[mask != 0])
            metrics['sdev'] = data[mask != 0].astype(np.float32).std()
            metrics['min'] = data[mask != 0].min()
            metrics['max'] = data[mask != 0].max()
            metrics['spread'] = np.ptp(data[mask != 0])
            # note that when we work with indexes, y-index comes first (row),
            # then x-index (column)
            metrics['center_9'] = data[centre_9_ix_iy[:,1], centre_9_ix_iy[:,0]].mean()
            metrics['center_5'] = data[centre_5_ix_iy[:,1], centre_5_ix_iy[:,0]].mean()
            metrics['center'] = data[cy, cx]
            output.append((c, mask, metrics))
    # now check other features of the contours and exclude them
    # if not matching
    exclude = []
    if output:
        for i, (c, mask, metrics) in enumerate(output):
            if min_sdev is not None and metrics['sdev'] < min_sdev:
                exclude.append(i)
            if mean_range is not None and\
               (mean_range[0] > metrics['mean'] or\
                mean_range[1] < metrics['mean']):
                exclude.append(i)
    if exclude:
        output = [output[i] for i in range(len(output)) if i not in exclude]
    # return the contours sorted by the desired metric
    output = sorted(output, key=lambda L: L[2][sortby], reverse=True)
    return output

def get_ipx_1D(icol_irow, n=9, ncols=80):
    """
    Return the 1-D vector indexes of the `n` pixels centered on `icol_irow`
    pixel of the 2-D frame.
    """
    ipc, ipr = icol_irow
    ipx = ncols * ipr + ipc-1

    # special cases first
    if n == 1:
        ipx = [ipx]
        return ipx
    if n == 3:
        ipx = [ipx, ipx-1, ipx+1]
        return ipx
    if n == 5:
        ipx = [ipx, ipx-1, ipx+1, ipx-ncols, ipx+ncols]
        return ipx
    if n == 6:
        ipx = [ipx, ipx-ncols-1, ipx-ncols+1, ipx+ncols-1, ipx+ncols+1]
        return ipx

    # if n == (2q+1)^2
    q, r = int(np.sqrt(n) // 2), int(np.sqrt(n) % 2)
    assert r == 1
    offs = range(-q, q+1)
    ix_offs = [coloffs + ncols * rowoffs for (rowoffs, coloffs)
               in itertools.product(offs, offs)]
    ipx = [ipx + offs for offs in ix_offs]
    return ipx

def get_spot_offsets(n=9):
    q, r = int(np.sqrt(n) // 2), int(np.sqrt(n) % 2)
    assert r == 1
    offs = range(-q, q+1)
    return np.array(list(itertools.product(offs,offs)))

def get_spot_in_frame(centre=(40,31), n=9):
    offs = get_spot_offsets()
    return np.array(centre) + offs

def stptime2float(x, fmt="%Y-%m-%dT%H:%M:%S.%f%z"):
    """
    Convert the time string into a numpy float.

    This function may be used as a converter, when reading e.g. output 
    from the SenXorViewer file to a numpy array, via np.loadtxt.
    However, this is not recommended.
    Instead, read the frame data separately:
        data = np.loadtext(filename, usecols=range(n, n+4960), delimiter=',')
    and construct a pandas dataframe for the n columns of header related stuff
    plus select pixels as necessary.
    """
    dt = datetime.datetime.strptime(x, fmt)
    return np.datetime64(dt).astype(float)


def compose_display(img_list):
    """
    Compose a single image out of a list of opencv-rendered images of the same size
    """
    if len(img_list) == 4:
        top_img = np.hstack(img_list[:2])
        bot_img = np.hstack(img_list[2:])
        img = np.vstack((top_img, bot_img))
        return img
    if len(img_list) == 6:
        top_img = np.hstack(img_list[:3])
        bot_img = np.hstack(img_list[3:])
        img = np.vstack((top_img, bot_img))
        return img
    if len(img_list) == 8:
        top_img = np.hstack(img_list[:4])
        bot_img = np.hstack(img_list[4:])
        img = np.vstack((top_img, bot_img))
        return img
    img = np.hstack(img_list)
    return img


def annotate(image, isd, scale=1):
    """
    Annotate the `image` with elements of input data structure `isd`.

    Currently supported annotation components are: contours, texts, rectangles.
    On the assumption that `isd` elements are computed within the original
    thermal frame, we must rescale them to the resolution of `image`.
    This can easily be accomplished by the `scale` parameter. However,
    if the isd components are obtained at different resolutions, e.g. from
    visual and thermal stream, then scale must remain 1, and the actual
    scaling must be done outside of this routine, at the time of composing the `isd`.
    """
    # contours are arrays of points; must be scaled to the resolution of frame.
    for contour in isd['contours']:
        cv.drawContours(image, [contour * scale], contourIdx=0, color=GREEN, thickness=2)

    # texts require coordinates; adjusting the fontsize may be necessary
    for text, coord in zip(isd['texts'], isd['text_coords']):
        coord = (coord[0] * scale, coord[1] * scale)
        cv.putText(image, text, coord, CVFONT, CVFONT_SIZE * 2./scale, GREEN, 2)

    # boxes are rectangles; must be scaled to the frame size
    for pts in isd['rectangles']:
        pts = [p * scale for p in pts]
        cv.rectangle(image, (pts[0], pts[1]), (pts[2], pts[3]), GREEN, 2)

    return image


class Display:
    """
    This class helps to compose a display of a list of rendered images,
    optionally locating the window at a specified location on the screen.
    """

    def __init__(self, options, composer=compose_display):
        """
        Decide how to organize rendered images on the display.
        `options` is a dictionary:

            * `window_coord` -- in x,y pixels,
            * `window_title` -- as a string.
        """
        self.coord = options['window_coord']
        self.title = options['window_title'].upper()
        self.composer = composer
        self.dir = Path(options.get('directory', 'images'))
        try:
            os.mkdir(self.dir)
        except FileExistsError:
            pass

    def __call__(self, img_list):
        self.img = self.composer(img_list)
        cv.imshow(self.title, self.img)
        if self.coord is not None:
            cv.moveWindow(self.title, *self.coord)

    def save(self, filename):
        """
        Save the image to a file.

        Filename includes extension, which determines the output format,
        but excludes the directory (it is set during class initialization).
        """
        filepath = str(self.dir / filename)
        cv.imwrite(filepath, self.img)


class HotSpot:
    """
    Object to calculate and store the thermal and geometrical metrics/statistics
    of a hot spot.
    The object has an `osd_attributes` member, which lists the accessible attributes
    for composing the output structured data dictionary.
    """
    def __init__(self, ix, frame, contour, mask, stats, p):
        # parameters
        self.p = p
        # index
        self.ix = ix
        self.ny, self.nx = frame.shape
        # contour
        self.contour = contour
        # mask corresponding to contour
        self.mask = mask
        # start composing output structure data based on the
        # contour statistics
        self.osd = stats.copy()   # copy, because we'll change it here
        # box returned below is (center(x,y), (width, height), angle of rotation)
        min_area_box = cv.minAreaRect(contour)
        # calculate area as width * height in pixels (int)
        self.bbox_area = min_area_box[1][0] * min_area_box[1][1]
        # get a list of points from the given (possibly rotated) box
        # min_area_box is [list of virtexes]
        self.min_area_bbox = np.asarray(cv.boxPoints(min_area_box), dtype=int)
        # establish the background temperature relevant to the hotspot
        # by defining an extended box that captures the relevant background emission
        # note we pass the rotated box not the list of virtexes
        self._extend_bbox(min_area_box)
        # background mask
        self._bg_mask()
        # Background temperature; it may be better to do some averaging
        self.bg = np.mean(sorted(frame[~self.bg_mask])[:12])
        self.bg_min = frame[~self.bg_mask].min()
        # update the osd dictionary with local calculations in addition to contour stats
        self.osd.update({
            'contour': self.contour,              # list of points 2D
            'min_area_bbox': self.min_area_bbox,  # list of virtexes
            'extended_bbox': self.extended_bbox,  # list of virtexes
            'extended_bbox_rotated': self.extended_bbox_rotated,  # list of virtexes
            'bbox_area': self.bbox_area,          # area in pixels
            'bg_min': self.bg_min,                # background temperature from min of
                                                  # extended box around hot spot
            'bg': self.bg                         # background temperature from the
                                                  # mean of the 12 lowest values in
                                                  # the extended box around hot spot
            })
        self.out_frames = {
            'hs_mask': self.mask * 255,           # hot spot mask
            'bg_mask': self.bg_mask * 255,        # background area overlapping hot spot
        }

    def _extend_bbox(self, min_area_box, e=None):
        """
        Extend the bounding box of the hot spot by `e` number of pixels on each side
        `min_area_box` is of the form: [(x_center, y_center), width, heights, angle]
        and is always a rectangle.
        `e` is the extension by which to enlarge on each side of the box.
        """
        if e is None: e = self.p['bbox_extension']
        w1 = min_area_box[1][0] + 2 * e
        h1 = min_area_box[1][1] + 2 * e
        # clear angle of rotation, because we can't easily slice for background
        # temperature estimation, but use the same center and the new width
        # box = (min_area_box[0], (w1, h1), min_area_box[2])
        box_rotated = (min_area_box[0], (w1, h1), min_area_box[2])
        box = (min_area_box[0], (w1, h1), 0)
        # transform the box definition to a [list of virtexes]
        # below we get an array of points which can be visualized as 'contours'
        self.extended_bbox_rotated = np.asarray(cv.boxPoints(box_rotated), dtype=int)
        self.extended_bbox = np.asarray(cv.boxPoints(box), dtype=int)

    def _bg_mask(self):
        """
        Return an array with the frame size:
            True outside the extended box of the hot spot
            False within the extended box
        """
        # Get the top-left and bottom right corner of the box
        # Note that because it comes from a rotated box, the order of
        # the points in the array does not correspond to any specific enumeration
        # Therefore, we cannot rely on indexing to extract top-left and bottom-right
        x0 = np.asarray(self.extended_bbox)[:,0].min()
        x1 = np.asarray(self.extended_bbox)[:,0].max()
        y0 = np.asarray(self.extended_bbox)[:,1].min()
        y1 = np.asarray(self.extended_bbox)[:,1].max()
        # The extended indexes may go below 0 and beyond frame size.
        # Here we constrain them to 0 and nx or ny
        (x0, y0) = max(x0, 0), max(y0, 0)
        (x1, y1) = min(x1, self.nx-1), min(y1, self.ny-1)
        self.bg_mask = np.ones(self.mask.shape, dtype=bool)
        self.bg_mask[y0:y1, x0:x1] = False


class ColdSpot:
    """
    Object to calculate and store the thermal and geometrical metrics/statistics
    of a cold spot.
    The object has an `osd_attributes` member, which lists the accessible attributes
    for composing the output structured data dictionary.
    """
    def __init__(self, ix, frame, contour, mask, stats, p):
        # parameters
        self.p = p
        # index
        self.ix = ix
        self.ny, self.nx = frame.shape
        # contour
        self.contour = contour
        # mask corresponding to contour
        self.mask = mask
        # start composing output structure data based on the
        # contour statistics
        self.osd = stats.copy()   # copy, because we'll change it here
        # box returned below is (center(x,y), (width, height), angle of rotation)
        min_area_box = cv.minAreaRect(contour)
        # calculate area as width * height in pixels (int)
        self.bbox_area = min_area_box[1][0] * min_area_box[1][1]
        # get a list of points from the given (possibly rotated) box
        # min_area_box is [list of virtexes]
        self.min_area_bbox = np.asarray(cv.boxPoints(min_area_box), dtype=int)
        # establish the background temperature relevant to the hotspot
        # by defining an extended box that captures the relevant background emission
        # note we pass the rotated box not the list of virtexes
        self._extend_bbox(min_area_box)
        # background mask
        self._bg_mask()
        # Background temperature; it may be better to do some averaging
        #
        self.bg = np.mean(sorted(frame[~self.bg_mask])[12:])
        self.bg_max = frame[~self.bg_mask].max()
        # update the osd dictionary with local calculations in addition to contour stats
        self.osd.update({
            'contour': self.contour,              # list of points 2D
            'min_area_bbox': self.min_area_bbox,  # list of virtexes
            'extended_bbox': self.extended_bbox,  # list of virtexes
            'extended_bbox_rotated': self.extended_bbox_rotated,  # list of virtexes
            'bbox_area': self.bbox_area,          # area in pixels
            'bg_max': self.bg_max,                # background temperature from max of
                                                  # extended box around cold spot
            'bg': self.bg                         # background temperature from the
                                                  # mean of the 12 highest values in
                                                  # the extended box around cold spot
            })
        self.out_frames = {
            'hs_mask': self.mask * 255,           # hot spot mask
            'bg_mask': self.bg_mask * 255,        # background area overlapping hot spot
        }

    def _extend_bbox(self, min_area_box, e=None):
        """
        Extend the bounding box of the hot spot by `e` number of pixels on each side
        `min_area_box` is of the form: [(x_center, y_center), width, heights, angle]
        and is always a rectangle.
        `e` is the extension by which to enlarge on each side of the box.
        """
        if e is None: e = self.p['bbox_extension']
        w1 = min_area_box[1][0] + 2 * e
        h1 = min_area_box[1][1] + 2 * e
        # clear angle of rotation, because we can't easily slice for background
        # temperature estimation, but use the same center and the new width
        # box = (min_area_box[0], (w1, h1), min_area_box[2])
        box_rotated = (min_area_box[0], (w1, h1), min_area_box[2])
        box = (min_area_box[0], (w1, h1), 0)
        # transform the box definition to a [list of virtexes]
        # below we get an array of points which can be visualized as 'contours'
        self.extended_bbox_rotated = np.asarray(cv.boxPoints(box_rotated), dtype=int)
        self.extended_bbox = np.asarray(cv.boxPoints(box), dtype=int)

    def _bg_mask(self):
        """
        Return an array with the frame size:
            True outside the extended box of the hot spot
            False within the extended box
        """
        # Get the top-left and bottom right corner of the box
        # Note that because it comes from a rotated box, the order of
        # the points in the array does not correspond to any specific enumeration
        # Therefore, we cannot rely on indexing to extract top-left and bottom-right
        x0 = np.asarray(self.extended_bbox)[:,0].min()
        x1 = np.asarray(self.extended_bbox)[:,0].max()
        y0 = np.asarray(self.extended_bbox)[:,1].min()
        y1 = np.asarray(self.extended_bbox)[:,1].max()
        # The extended indexes may go below 0 and beyond frame size.
        # Here we constrain them to 0 and nx or ny
        (x0, y0) = max(x0, 0), max(y0, 0)
        (x1, y1) = min(x1, self.nx-1), min(y1, self.ny-1)
        self.bg_mask = np.ones(self.mask.shape, dtype=bool)
        self.bg_mask[y0:y1, x0:x1] = False


class CVSegment:
    """
    A class for quick segmentation based on simple, Otsu, or Adaptive threshold.
    """
    def _adaptive_threshold(self, frame, *args, **kwargs):
        threshold = None
        binary = cv.adaptiveThreshold(frame, *args, **kwargs)
        return threshold, binary

    def _otsu_threshold(self, frame, otsu_threshold_delta, *args, **kwargs):
        """
        Otsu's algorithm of threshold selection is based on a histogram analysis.
        In thermal imaging, it seems that more stable result may be obtained by
        shifting the threshold a bit. This is achieved by the `otsu_threshold_delta`.
        """
        threshold, binary = cv.threshold(frame, *args, **kwargs)
        threshold += otsu_threshold_delta
        # print(threshold - otsu_threshold_delta, threshold)
        threshold, binary = cv.threshold(frame, thresh=threshold,
                                         maxval=1, type=cv.THRESH_BINARY)
        return threshold, binary

    def _contour(self, data, binary):
        contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = get_contour_stats(data, contours, minArea=self.p['contour_minArea'])
        return contours

    def __init__(self, p):
        self.p = p
        if p['threshold_type'] == 'simple':
            self.threshold = partial(cv.threshold, thresh=p['threshold'],
                                     maxval=1, type=cv.THRESH_BINARY)
        if p['threshold_type'] == 'otsu':
            self.threshold = partial(self._otsu_threshold,
                                    otsu_threshold_delta=p['otsu_threshold_delta'],
                                    thresh=p['threshold'], maxval=1,
                                    type=cv.THRESH_BINARY+cv.THRESH_OTSU)
        if p['threshold_type'] == 'adaptive':
            self.threshold = partial(self._adaptive_threshold,
                maxValue=1, adaptiveMethod=cv.ADAPTIVE_THRESH_MEAN_C,
                thresholdType=cv.THRESH_BINARY,
                blockSize=p['threshold_blocksize'], C=p['threshold_C'])

    def __call__(self, frame, frui8=None):
        # binarise
        if frui8 is None:
            frui8 = remap(frame)
        self.frui8 = frui8
        threshold, binary = self.threshold(frui8)
        self.binary = binary
        # contours of the hot spots and get the corresonding masks and stats
        contours = (self._contour(frame, binary))
        self.hotspots = [HotSpot(i, frame, c[0], c[1], c[2], self.p)\
                    for i, c in enumerate(contours)]
        # formulate the intermediate processing frames and output structured data
        self.out_frames = {'normed': self.frui8, 'binary': self.binary}
        self.osd = {'n_hotspots': len(self.hotspots)}


class CVSegmentCH:
    """
    A class for quick segmentation on both hot-on-cold and cold-on-hot split.
    """
    def _adaptive_threshold(self, frame, *args, **kwargs):
        threshold = None
        binary = cv.adaptiveThreshold(frame, *args, **kwargs)
        return threshold, binary

    def _otsu_threshold(self, frame, otsu_threshold_delta, *args, **kwargs):
        """
        Otsu's algorithm of threshold selection is based on a histogram analysis.
        In thermal imaging, it seems that more stable result may be obtained by
        shifting the threshold a bit. This is achieved by the `otsu_threshold_delta`.
        """
        threshold, binary = cv.threshold(frame, *args, **kwargs)
        threshold += otsu_threshold_delta
        # print(threshold - otsu_threshold_delta, threshold)
        threshold, binary = cv.threshold(frame, thresh=threshold,
                                         maxval=1, type=cv.THRESH_BINARY)
        return threshold, binary

    def _contour(self, data, binary):
        contours, hierarchy = cv.findContours(binary, cv.RETR_TREE,
                                              cv.CHAIN_APPROX_SIMPLE)
        contours = get_contour_stats(data, contours, minArea=self.p['contour_minArea'])
        return contours

    def __init__(self, p):
        self.p = p
        if p['threshold_type'] == 'simple':
            self.threshold = partial(cv.threshold, thresh=p['threshold'],
                                     maxval=1, type=cv.THRESH_BINARY)
        if p['threshold_type'] == 'otsu':
            self.threshold = partial(self._otsu_threshold,
                                    otsu_threshold_delta=p['otsu_threshold_delta'],
                                    thresh=p['threshold'], maxval=1,
                                    type=cv.THRESH_BINARY+cv.THRESH_OTSU)
        if p['threshold_type'] == 'adaptive':
            self.threshold = partial(self._adaptive_threshold,
                maxValue=1, adaptiveMethod=cv.ADAPTIVE_THRESH_MEAN_C,
                thresholdType=cv.THRESH_BINARY,
                blockSize=p['threshold_blocksize'], C=p['threshold_C'])

    def __call__(self, frame, frui8=None):
        # binarise
        if frui8 is None:
            frui8 = remap(frame)
        self.frui8 = frui8
        hs_threshold, hs_binary = self.threshold(frui8)
        cs_threshold, cs_binary = self.threshold(255-frui8)
        self.hs_binary = hs_binary
        self.cs_binary = cs_binary
        # contours of the hot spots and get the corresonding masks and stats
        contours = (self._contour(frame, hs_binary))
        self.hotspots = [HotSpot(i, frame, c[0], c[1], c[2], self.p)\
                    for i, c in enumerate(contours)]
        # contours of the cold spots and get the corresonding masks and stats
        contours = (self._contour(frame, cs_binary))
        self.coldspots = [ColdSpot(i, frame, c[0], c[1], c[2], self.p)\
                    for i, c in enumerate(contours)]
        # formulate the intermediate processing frames and output structured data
        self.out_frames = {'normed': self.frui8,
                           'hs_binary': self.hs_binary,
                           'cs_binary': self.cs_binary}
        self.osd = {
            'n_hotspots': len(self.hotspots),
            'n_coldspots': len(self.coldspots)
        }


