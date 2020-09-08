import numpy as np
from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa
from scipy.ndimage import gaussian_filter
from scipy.ndimage import convolve
from scipy import ndimage, misc
import matplotlib.pyplot as plt
import cv2
from PIL import Image

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
    # show(imgint)

    # mask_gray = cv2.normalize(src=imguint, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    # image = Image.fromarray(imgint.astype(np.uint8))
    # grayImage = cv2.cvtColor(imguint, cv2.COLOR_GRAY2BGR)

    imguint = np.asarray(imgint, dtype=np.uint8)
    img_cv = cv2.resize(imguint,(200,200))
    # img_cv = 255 - img_cv
    # show(img_cv)

    # contours, hierarchy = cv2.findContours(img_cv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contour_img = cv2.drawContours(img_cv, contours, -1, (255,255,255), 3)
    # cv2.imshow('contours', contour_img)
    # show(img_cv)

    # iContours = 0
    # for contour in contours:
    #     cv2.drawContours(imgint, contours, iContours, (255, 255, 255), 3)
    #     iContours += 1    


    # gray = cv2.medianBlur(img_cv, 3)
    # show(gray)
    # hist, _ = np.histogram(gray, bins=np.max(gray))
    # show(threshed)

    circles = cv2.HoughCircles(img_cv, cv2.HOUGH_GRADIENT, 1, 10,
                               param1=200, param2=100,
                               minRadius=10, maxRadius=50)


    # threshed = cv2.adaptiveThreshold(imguint, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
    # gray = cv2.medianBlur(threshed, 3)
    # hist, _ = np.histogram(threshed, bins=np.max(threshed))
    # show(threshed)

    # ret, thresh = cv2.threshold(scaled, 172, 255, 0)
    # show(thresh)

    # contours, hierarchy = cv2.findContours(mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # iContours = 0
    # for contour in contours:
    #     cv2.drawContours(imgint, contours, iContours, (255, 255, 255), 3)
    #     iContours += 1    
    # show(imgint)

    found_circles = []
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
                        detected = x, y, r
                        found_circles.append(detected)
                        print("iou:", detected, iou(params, detected))
                        
                    break

                guess_radius -= 2 
                if guess_radius < minimum_circle_size:
                    break;
            guess_dp += 1.5
        guess_accumulator_array_threshold -= 2


    if len(found_circles) > 0:
        return found_circles[0]
    # if circles is not None:
    #     circles = np.round(circles[0, :]).astype("int")
    #     for (x, y, r) in circles:
    #         print('x,y,r:', x, y, r)
    #         return x, y, r
    # else:
    #    show(mask_gray)
    return None


def find_circlev4(img):
    # show(img)
    # noise = 1
    # img -= noise * np.random.rand(*img.shape)
    # show(img)

    scaled = (255 * (img - np.min(img)) / np.ptp(img)).astype(int)
    imgint = np.asarray(scaled, dtype=np.int16)
    # show(imgint)

    # minmax(imgint)
    imgint -= 2 * np.median(imgint).astype(int)
#    imgint -= 2 * np.mean(imgint).astype(int)
    # show(imgint)

    # imgint -= np.max(imgint)
    # show(imgint)

    imgint[imgint < 0] = 0
    # show(imgint)

    imguint = np.asarray(imgint, dtype=np.uint)
    # show(imguint)

    # kernel = np.ones((3, 3), np.uint8)
    # dilate = cv2.dilate(imguint, kernel, iterations=3)
    # show(dilate)

    # closing = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
    # show(closing)

    # for i in range(7):
    #     gaussian = gaussian_filter(closing, sigma=i)
    #     show(gaussian)

    # gaussian = gaussian_filter(closing, sigma=4)
    # show(gaussian)

    uint_img = np.array(imguint).astype('uint8')
    gray = cv2.cvtColor(uint_img, cv2.COLOR_GRAY2BGR)
    mask_gray = cv2.normalize(src=imguint, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    guess_dp = 1.0
    minimum_circle_size = 10      #this is the range of possible circle in pixels you want to find
    maximum_circle_size = 50     #maximum possible circle size you're willing to find in pixels

    max_guess_accumulator_array_threshold = 100     #minimum of 1, no maximum, (max 300?) the quantity of votes 
    guess_accumulator_array_threshold = max_guess_accumulator_array_threshold
    while guess_accumulator_array_threshold > 1:
        guess_radius = maximum_circle_size

        while guess_dp < 9:
            while True:
                circles = cv2.HoughCircles(mask_gray, 
                        cv2.HOUGH_GRADIENT, 
                        dp=guess_dp,               #resolution of accumulator array.
                        minDist=100,                #number of pixels center of circles should be from each other, hardcode
                        param1=50,
                        param2=guess_accumulator_array_threshold,
                        minRadius=minimum_circle_size,    #HoughCircles will look for circles at minimum this size
                        maxRadius=maximum_circle_size     #HoughCircles will look for circles at maximum this size
                        )
                if circles is not None:
                    print(circles)
                    break

                guess_radius -= 2 
                if guess_radius < minimum_circle_size:
                    break;
            guess_dp += 1.5
        guess_accumulator_array_threshold -= 2


    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            print('x,y,r:', x, y, r)
            return x, y, r
    # else:
    #    show(mask_gray)
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

    result = shape0.intersection(shape1).area / shape0.union(shape1).area
    # print('iou:', result)

    return (result)


results = [1,2,3]
np.set_printoptions(threshold=np.inf)

for _ in range(1000):
    # params, img = noisy_circle(200, 50, 2)
    params, img = noisy_circle(200, 50, 2)
    print('params:', params)

    detected = find_circle(params, img)
    if detected is not None:
        results.append(iou(params, detected))
results = np.array(results)
print((results > 0.7).mean())

