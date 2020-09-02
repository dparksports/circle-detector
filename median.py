import numpy as np
from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa
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
            print('x,y,r:', x, y, r)
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
    log_nonzero(img)
    # show(img)

    # Noise
    img += noise * np.random.rand(*img.shape)
    log_nonzero(img)
    # show(img)
    return (row, col, rad), img

def find_circle(img):
    # show(img)
    # noise = 1
    # img -= noise * np.random.rand(*img.shape)
    # show(img)

    scaled = (255 * (img - np.min(img)) / np.ptp(img)).astype(int)
    imgint = np.asarray(scaled, dtype=np.int16)
    # show(imgint)

    minmax(imgint)
    imgint -= 2 * np.median(imgint).astype(int)
    # imgint -= 2 * np.mean(imgint).astype(int)
    # show(imgint)

    # imgint -= np.max(imgint)
    # show(imgint)

    imgint[imgint < 0] = 0
    # show(imgint)

    imguint = np.asarray(imgint, dtype=np.uint8)
    # show(imguint)

    # for i in range(5):
    #     gaussian = gaussian_filter(imgint, sigma=i)
    #     show(gaussian)
    #
    gaussian = gaussian_filter(imguint, sigma=2)
    # show(gaussian)

    # circles = cv2.HoughCircles(image=gaussian,
    #                            method=cv2.HOUGH_GRADIENT,
    #                            dp=1.2,
    #                            minDist=2 * 10,
    #                            param1=50,
    #                            param2=50,
    #                            minRadius=10,
    #                            maxRadius=50
    #                            )

    circles = cv2.HoughCircles(gaussian, cv2.HOUGH_GRADIENT, 1, 200, param1=30, param2=45, minRadius=10, maxRadius=50)
    circles = cv2.HoughCircles(imguint, cv2.HOUGH_GRADIENT, 1, 200, param1=30, param2=45, minRadius=10, maxRadius=50)
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")

        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            print('x,y,r:', x, y, r)
            return x, y, r
    return None


def find_circle_v3(img):
    scaled = (255 * (img - np.min(img)) / np.ptp(img)).astype(int)
    imgint = np.asarray(scaled, dtype=np.uint8)
    show(imgint)

    sobel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    G_x = convolve(imgint, sobel)
    G_y = convolve(imgint, np.fliplr(sobel).transpose())
    show(G_x)
    show(G_y)

    G = pow((G_x*G_x + G_y*G_y),0.5)
    show(G)
    G[G<128] = 0
    show(G)

    show(imgint)
    laplacian = np.array([[1,1,1],[1,-8,1],[1,1,1]])
    L = convolve(imgint, laplacian)
    show(L)

    M,N = L.shape
    temp = np.zeros((M+2,N+2))                                                  #Initializing a temporary image along with padding
    temp[1:-1,1:-1] = L                                                         #result hold the laplacian convolved image
    result = np.zeros((M,N))                                                    #Initializing a resultant image along with padding
    for i in range(1,M+1):
        for j in range(1,N+1):
            if temp[i,j]<0:                                                     #Looking for a negative pixel and checking its 8 neighbors
                for x,y in (-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1):
                        if temp[i+x,j+y]>0:
                            result[i-1,j-1] = 1

    output = np.array(np.logical_and(result,G),dtype=np.uint8)
    show(output)


def find_circle_v2(img):
    scaled = (255 * (img - np.min(img)) / np.ptp(img)).astype(int)
    # minmax(scaled)
    # show(scaled)

    imgint = np.asarray(scaled, dtype=np.uint8)
    # minmax(imgint)
    show(imgint)

    # gray = cv2.GaussianBlur(imgint, (5, 5), 0);
    # show(gray)

    gray = cv2.medianBlur(imgint, 5)
    show(gray)

    # gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
    #                              cv2.THRESH_BINARY, 11, 3.5)
    # gray = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # gray = cv2.adaptiveThreshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # thresh1 = cv2.adaptiveThreshold(imgint, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    #                                 cv2.THRESH_BINARY, 199, 5)
    # show(thresh1)
    # thresh2 = cv2.adaptiveThreshold(imgint, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                 cv2.THRESH_BINARY, 199, 5)
    # show(thresh2)

    # kernel = np.ones((3, 3), np.uint8)
    # gray = cv2.erode(gray, kernel, iterations=1)
    # show(gray)
    #
    # gray = cv2.dilate(gray, kernel, iterations=1)
    # show(gray)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 200, param1=30, param2=45, minRadius=3, maxRadius=50)
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")

        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            print('x,y,r:', x, y, r)


def find_circle_v1(img):
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
for _ in range(20):
    # params, img = noisy_circle(200, 50, 2)
    params, img = noisy_circle(200, 50, 2)
    detected = find_circle(img)
    if detected is not None:
        results.append(iou(params, detected))
results = np.array(results)
print((results > 0.7).mean())
