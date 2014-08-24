import sys
import os
import numpy as np
import scipy.signal as signal
import scipy.ndimage as ndimage
import Image
import ImageDraw
import ImageFont
import string
import argparse
import cPickle

ASPECT_RATIO = 1.75
WHITE_THRESHOLD = 63

def to_array(img):
    width, height = img.size
    if img.mode in ('1', 'L'):
        n_channels = 1
    elif img.mode == 'RGB':
        n_channels = 3
    else:
        n_channels = 4

    img = np.ndarray(shape=(width * height * n_channels, ), buffer=img.tobytes(), dtype=np.uint8)
    img = np.asarray(img, dtype=np.float64)
    if n_channels == 1:
        img = img.reshape((height, width))
    else:
        img = img.T
        img = img.reshape((n_channels, height, width))
    return img 

def to_grayscale(img):
    return img.convert('L')

def downsample(img, new_width):
    width, height = img.size
    new_height = int(round(height * float(new_width) / width / ASPECT_RATIO))
    return img.resize((new_width, new_height), Image.ANTIALIAS)

def normalize(img_arr):
    min_val = img_arr.min()
    max_val = img_arr.max()
    if min_val == max_val:
        return img_arr
    delta = float(max_val - min_val)
    img_arr = (img_arr - min_val) / delta
    return img_arr

def get_char_densities():
    density_file = 'char_densities.pkl'
    try:
        with open(density_file, 'r') as f:
            chars, masses = cPickle.load(f)
            return chars, masses
    except Exception:
        print >> sys.stderr, 'Creating character density file'

    printable = set(string.printable) - set(string.whitespace)
    printable.add(' ')
    printable = ''.join(printable)

    font = ImageFont.truetype('DejaVuSansMono.ttf', 144)

    tmp_img = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(tmp_img)
    width, height = draw.textsize(printable, font=font)
    char_width = width / len(printable)
    height += 100

    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), printable, font=font)
    img_arr = to_array(to_grayscale(img))

    mask = img_arr > WHITE_THRESHOLD
    tmp_masses = mask.sum(axis=0).reshape((len(printable), char_width))
    tmp_masses = normalize(tmp_masses.sum(axis=1))
    masses = zip(printable, tmp_masses)
    masses.sort(key = lambda x: x[1])
    chars, masses = zip(*masses)

    with open(density_file, 'wb') as f:
        cPickle.dump((chars, masses), f, cPickle.HIGHEST_PROTOCOL)

    return chars, masses

def get_ascii(img, chars, masses):
    delta = masses[1] - masses[0]
    masses = normalize(np.hstack([-delta, masses]))[1:]
    bins = np.digitize(img.ravel(), masses, right=True).reshape(img.shape)
    chars = np.array(chars)
    ascii_art = [''.join(chars[row]) for row in bins]
    return ascii_art

def get_edges(gray_img):
    img_arr = to_array(gray_img)
    stdev = min(gray_img.size) * 0.003
    if stdev >= 1:
        img_arr = ndimage.gaussian_filter(img_arr, stdev, order=0)
    grad_x = ndimage.sobel(img_arr, axis=1)
    grad_y = ndimage.sobel(img_arr, axis=0)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    edge_arr = np.empty(img_arr.shape, dtype=np.uint8)
    (255 * normalize(grad_magnitude)).round(out=edge_arr)
    edges = Image.fromarray(edge_arr, mode='L')
    return edges

def asciify(img, linewidth):
    chars, masses = get_char_densities()
    img = to_grayscale(img)
    img = get_edges(img)
    img = downsample(img, linewidth)
    img = normalize(to_array(img))
    ascii_art = get_ascii(img, chars, masses)
    return ascii_art

def main():
    parser = argparse.ArgumentParser(description=
        """
        Make ASCII art!
        """)
    parser.add_argument('-w', '--width', type=int, dest='width', default=80,
        help='ASCII image linewidth, default 80')
    parser.add_argument('image_file', help='Path to image file')
    args = parser.parse_args()

    img = Image.open(args.image_file)
    ascii_art = asciify(img, args.width)
    print '\n'.join(ascii_art)

if __name__ == '__main__':
    main()
