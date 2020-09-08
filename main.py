import numpy as np
from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa
from skimage.transform import hough_circle, hough_circle_peaks
from scipy.ndimage import gaussian_filter
from scipy.ndimage import convolve
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
    # show(imgint)

    # circles = cv2.HoughCircles(imgint, cv2.HOUGH_GRADIENT, 1.2, 2*50)
    circles = cv2.HoughCircles(image=imgint,
                               method=cv2.HOUGH_GRADIENT,
                               dp=1.2,
                               minDist=2 * 50,
                               param1=50,
                               param2=50,
                               minRadius=10,
                               maxRadius=50
                               )
    if circles is not None:
        output = np.zeros((img.shape[0], imgint.shape[1]))
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            print('x,y,r:', y, x, r)
            # cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            # cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

        # show(np.hstack([img, output]))
    # print('circles:', circles)

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
    rad = np.random.randint(10, max(10, radius))
    draw_circle(img, row, col, rad)
    # log_nonzero(img)
    # show(img)

    # Noise
    img += noise * np.random.rand(*img.shape)
    # log_nonzero(img)
    # show(img)
    return (row, col, rad), img

def find_circle(params, img):
    scaled = (255 * (img - np.min(img)) / np.ptp(img)).astype(int)
    # show(scaled)
 
    imgint = np.asarray(scaled, dtype=np.int16)
    imgint -= 2 * np.median(imgint).astype(int)
    # show(imgint)
    imgint[imgint < 0] = 0
    # show(imgint)

    hist, _ = np.histogram(imgint, bins=np.max(imgint))
    # hist = hist.astype(np.int32)
    index = 0
    for value in hist:
        index += 1
        if value > 20:
            imgint[imgint < index] = 0
            hist, _ = np.histogram(imgint, bins=np.max(imgint))

    # show(imgint)

    nonzeroindices = np.nonzero(imgint)
    imgint[nonzeroindices] = 255
    imgint = 255 - imgint
    # show(imgint)

    imguint = np.asarray(imgint, dtype=np.uint8)
    img_cv = cv2.resize(imguint,(200,200))

    best_iou = 0
    best_circle = None
    guess_dp = 1.0
    minimum_circle_size = 10      #this is the range of possible circle in pixels you want to find
    maximum_circle_size = 50     #maximum possible circle size you're willing to find in pixels

    max_guess_accumulator_array_threshold = 100     #minimum of 1, no maximum, (max 300?) the quantity of votes 
    guess_accumulator_array_threshold = max_guess_accumulator_array_threshold
    while guess_accumulator_array_threshold > 1:
        guess_radius = maximum_circle_size

        while guess_dp < 9:
            while True:
                circles = cv2.HoughCircles(img_cv, 
                        cv2.HOUGH_GRADIENT, 
                        dp=guess_dp,               #resolution of accumulator array.
                        minDist=100,                #number of pixels center of circles should be from each other, hardcode
                        param1=50,
                        param2=guess_accumulator_array_threshold,
                        minRadius=minimum_circle_size,    #HoughCircles will look for circles at minimum this size
                        maxRadius=maximum_circle_size     #HoughCircles will look for circles at maximum this size
                        )
                if circles is not None:
                    circles = np.round(circles[0, :]).astype("int")
                    for (x,y,r) in circles:
                        detected = y, x, r
                        score = iou(params, detected)
                        if score > best_iou:
                            best_iou = score
                            best_circle = detected
                            print("best_circle:", detected, iou(params, detected))
                        
                    break

                guess_radius -= 2 
                if guess_radius < minimum_circle_size:
                    break;
            guess_dp += 1.5
        guess_accumulator_array_threshold -= 2


    if best_iou > 0:
        return best_circle
    return None


def iou(params0, params1):
    row0, col0, rad0 = params0
    row1, col1, rad1 = params1

    shape0 = Point(row0, col0).buffer(rad0)
    shape1 = Point(row1, col1).buffer(rad1)

    result = shape0.intersection(shape1).area / shape0.union(shape1).area
    # print('iou:', result)

    return (result)


results = [1,2,3]
np.set_printoptions(threshold=np.inf)

for _ in range(100):
    # params, img = noisy_circle(200, 50, 2)
    params, img = noisy_circle(200, 50, 2)
    print('params:', params)

    detected = find_circle(params, img)
    if detected is not None:
        results.append(iou(params, detected))
results = np.array(results)
print((results > 0.7).mean())

