import cv2

img = Scalar(0);
Rect r(10, 10, 100, 100);
Mat smallImg = img(r);

Mat img = imread("image.jpg");
IplImage img1 = img;
CvMat m = img;

Mat img = imread("scibowl.png"); // loading a 8UC3 image
Mat grey;
cvtColor(img, grey, CV_BGR2GRAY);

src.convertTo(dst, CV_32F);
