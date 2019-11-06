import PIL
import matplotlib
import numpy as np
from matplotlib import cm

from . import maps


def draw_ellipse_from_catalog(catalog, shape, wcs, transpose_method=PIL.Image.FLIP_TOP_BOTTOM, **kwargs):
    img = PIL.Image.new("RGBA", shape[-2:][::-1])
    draw = PIL.ImageDraw.Draw(img, "RGBA")

    pix = maps.catalog2pix(catalog, shape, wcs).T
    # rad = 8
    width = kwargs['width'] if 'width' in kwargs else 2
    snr_min = np.floor((catalog['snr'].min()))

    use_cm = 'use_cm' in kwargs and kwargs['use_cm']
    if use_cm:
        cm_range = kwargs['cm_range'] if 'cm_range' in kwargs else [catalog['jy'].min(), catalog['jy'].max()]
        cm_style = kwargs['cm_style'] if 'cm_style' in kwargs else 'coolwarm' 
        cmap = cm.get_cmap(cm_style)
        norm = matplotlib.colors.Normalize(vmin=np.log(cm_range[0]), vmax=np.log(cm_range[1]))
        smap = cm.ScalarMappable(norm=norm, cmap=cmap)
    else:
        smap = None

    def draw_color(flux, smap):
        if not smap:
            return 'black'
        else:
            return smap.to_rgba(np.log(flux), bytes=True)

    for idx in range(pix.shape[0]):
        antialias = 1 if width < 1 else 4
        dec = pix[idx][0]
        ra = pix[idx][1]
        rad = 8 + int((catalog['snr'][idx] - snr_min) / 2)
        color = draw_color(catalog['jy'][idx], smap)
        draw_ellipse(img, (ra - rad, dec - rad, ra + rad, dec + rad),
                     outline=color, width=width, antialias=antialias)

    return img if not transpose_method else img.transpose(transpose_method)


def draw_ellipse(image, bounds, width=1, outline='black', antialias=1):
    """
    Took it from pixell.enplot ...    

    Improved ellipse drawing function, based on PIL.ImageDraw.
    Improved from
    http://stackoverflow.com/questions/32504246/draw-ellipse-in-python-pil-with-line-thickness"""
    bounds = np.asarray(bounds)
    # Create small coordinate system around ellipse, with a
    # margin of width on each side
    esize = bounds[2:] - bounds[:2] + 2 * width
    ebounds = bounds - bounds[[0, 1, 0, 1]] + width
    # Use a single channel image (mode='L') as mask.
    # The size of the mask can be increased relative to the imput image
    # to get smoother looking results.
    mask = PIL.Image.new(size=tuple(esize * antialias), mode='L', color='black')
    draw = PIL.ImageDraw.Draw(mask)
    # draw outer shape in white (color) and inner shape in black (transparent)
    for offset, fill in (width / -2.0, 'white'), (width / 2.0, 'black'):
        a = (ebounds[:2] + offset) * antialias
        b = (ebounds[2:] - offset) * antialias
        draw.ellipse([a[0], a[1], b[0], b[1]], fill=fill)
    # downsample the mask using PIL.Image.LANCZOS
    # (a high-quality downsampling filter).
    mask = mask.resize(esize, PIL.Image.LANCZOS)
    # paste outline color to input image through the mask
    image.paste(outline, tuple(bounds[:2] - width), mask=mask)
