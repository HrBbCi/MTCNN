from mtcnn.mtcnn import MTCNN
import cv2
img = cv2.imread("test_image.jpg")
detector = MTCNN()
print(detector.detect_faces(img))