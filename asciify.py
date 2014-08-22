import numpy as np
import Image
import ImageDraw
import ImageFont
import string

ASPECT_RATIO = 1.75

def to_array(img):
    width, height = img.size
    if img.mode in ('1', 'L'):
        n_channels = 1
    elif img.mode == 'RGB':
        n_channels = 3
    else:
        n_channels = 4

    img = np.array(list(img.getdata())).T
    if n_channels == 1:
        img = img.reshape((height, width))
    else:
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
    delta = float(max_val - min_val)
    img_arr = (img_arr - min_val) / delta
    return img_arr

def get_char_densities():
    printable = set(string.printable) - set(string.whitespace)
    printable.add(' ')
    printable = ''.join(printable)

    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 144)

    tmp_img = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(tmp_img)
    width, height = draw.textsize(printable, font=font)
    char_width = width / len(printable)
    height += 100

    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), printable, font=font)

    img = to_array(to_grayscale(img))

    masses = []
    for i in xrange(len(printable)):
        char = img[:, i*char_width:(i+1)*char_width]
        mask = char > 63
        mass = mask.sum()
        masses.append((printable[i], mass))
    masses.sort(key = lambda x: x[1])
    normalizer = float(max(masses, key = lambda x: x[1])[1])
    chars, masses = zip(*[(c, val / normalizer) for c, val in masses])

    return chars, masses

def to_ascii(img, chars, masses):
    delta = masses[1] - masses[0]
    masses = normalize(np.hstack([-delta, masses]))[1:]
    bins = np.digitize(img.ravel(), masses, right=True).reshape(img.shape)
    chars = np.array(chars)
    for row in bins:
        print ''.join(chars[row])

def main():
    chars, masses = get_char_densities()
    print masses
    img = Image.open('me.jpg')
    img = downsample(to_grayscale(img), 80)
    img = normalize(to_array(img))
    to_ascii(img, chars, masses)

if __name__ == '__main__':
    main()
