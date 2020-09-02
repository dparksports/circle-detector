import numpy as np
from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa
from scipy.ndimage import gaussian_filter
from scipy import ndimage, misc
import matplotlib.pyplot as plt
import cv2

def show(img):
    plt.imshow(img)
    plt.show()

def log(img):
    img16 = np.asarray(img, dtype=np.float16)
    print('img16:', img16.shape, img16)

def minmax(img):
    print('minmax:', np.min(img), np.max(img), np.median(img), np.mean(img), img.shape)

def log_nonzero(img):
    img16 = np.asarray(img, dtype=np.float16)
    rows, cols = np.nonzero(img16)
    nonzeros = img16[rows,cols]

    print('nonzeros:', np.mean(nonzeros), np.median(nonzeros), nonzeros.shape)
    # print('nonzeros:', nonzeros)

def print_xyr(img):
    scaled = (255 * (img - np.min(img)) / np.ptp(img)).astype(int)
    # show(scaled)

    imgint = np.asarray(scaled, dtype=np.uint8)
    show(imgint)

    # circles = cv2.HoughCircles(imgint, cv2.HOUGH_GRADIENT, 1.2, 2*50)
    circles = cv2.HoughCircles(image=imgint,
                               method=cv2.HOUGH_GRADIENT,
                               dp=1.2,
                               minDist=2 * 50,
                               param1=50,
                               param2=50,
                               minRadius=3,
                               maxRadius=50
                               )
    if circles is not None:
        output = np.zeros((img.shape[0], imgint.shape[1]))
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            print('x,y,r:', x, y, r)
            # cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            # cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

        # show(np.hstack([img, output]))
    print('circles:', circles)

def white(img):
    show(img)
    whites = np.array(np.where(img == img.max()))
    show(whites)
    print('whites:', whites)

def draw_circle(img, row, col, rad):
    rr, cc, val = circle_perimeter_aa(row, col, rad)
    valid = (
        (rr >= 0) &
        (rr < img.shape[0]) &
        (cc >= 0) &
        (cc < img.shape[1])
    )

    img[rr[valid], cc[valid]] = val[valid]

    zeros = np.zeros((img.shape[0], img.shape[0]), dtype=np.float)
    zeros[rr[valid], cc[valid]] = val[valid] * 255
    print_xyr(zeros)

def noisy_circle(size, radius, noise):
    img = np.zeros((size, size), dtype=np.float)

    # Circle
    row = np.random.randint(size)
    col = np.random.randint(size)
    rad = radius
    draw_circle(img, row, col, rad)
    log_nonzero(img)
    # show(img)

    # Noise
    img += noise * np.random.rand(*img.shape)
    log_nonzero(img)
    # show(img)
    return (row, col, rad), img

def find_circle(img):
    scaled = (255 * (img - np.min(img)) / np.ptp(img)).astype(int)
    minmax(scaled)
    # show(scaled)

    imgint = np.asarray(scaled, dtype=np.uint8)
    minmax(imgint)
    show(imgint)

    # show(imgint)
    # median = ndimage.median_filter(imgint, size=3)
    # median = ndimage.median_filter(imgint, size=1)
    # minmax(median)
    # show(median)

    # show(imgint)
    # gaussian = gaussian_filter(imgint, sigma=6)
    # gaussian = gaussian_filter(imgint, sigma=2)
    # minmax(gaussian)
    # show(gaussian)

    # circles = cv2.HoughCircles(imgint, cv2.HOUGH_GRADIENT, 1.2, 100)
    circles = cv2.HoughCircles(image=imgint,
                               method=cv2.HOUGH_GRADIENT,
                               dp=1.2,
                               minDist=2 * 50,
                               param1=50,
                               param2=50,
                               minRadius=3,
                               maxRadius=50
                               )

    if circles is not None:
        output = np.zeros((imgint.shape[0], imgint.shape[1]))
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            print('x,y,r:', x, y, r)
        print('circles:', circles)

    gaussian = gaussian_filter(imgint, sigma=5)
    minmax(gaussian)
    show(gaussian)



    # for i in range(9):
    #     gaussian = gaussian_filter(imgint, sigma=i)
    #     minmax(gaussian)
    #     show(gaussian)

    # dst = np.zeros((imgint.shape[0], imgint.shape[1]))
    # dst = cv2.fastNlMeansDenoising(imgint, dst, h=30, templateWindowSize=7, searchWindowSize=21)
    # dst = cv2.fastNlMeansDenoising(imgint, dst, h=1, templateWindowSize=7, searchWindowSize=21)
    # minmax(dst)
    # show(dst)

    # Fill in this function
    return 100, 100, 30


def iou(params0, params1):
    row0, col0, rad0 = params0
    row1, col1, rad1 = params1

    shape0 = Point(row0, col0).buffer(rad0)
    shape1 = Point(row1, col1).buffer(rad1)

    return (
        shape0.intersection(shape1).area /
        shape0.union(shape1).area
    )


np.set_printoptions(threshold=np.inf)
results = []
for _ in range(1000):
    # params, img = noisy_circle(200, 50, 2)
    params, img = noisy_circle(200, 50, 2)
    detected = find_circle(img)
    results.append(iou(params, detected))
results = np.array(results)
print((results > 0.7).mean())
