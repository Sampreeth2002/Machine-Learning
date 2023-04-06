import cv2

img = cv2.imread("logo.png")
gray = cv2.imread("logo.png",cv2.IMREAD_GRAYSCALE)
cv2.imshow("Dog Image",gray)

cv2.waitKey(5000)
cv2.destroyAllWindows()