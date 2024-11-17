import cv2

def white_balance(image):
    # Create a SimpleWB object and balance the white
    white_balancer = cv2.xphoto.createSimpleWB()
    result = white_balancer.balanceWhite(image)
    return result

if __name__ == "__main__":
    image = cv2.imread("Photos/Green Background/bloc_3.jpeg")
    if image is None:
        print("Could not open or find the image!")
    else:
        white_balanced_image = white_balance(image)
        cv2.imshow("Original Image", image)
        cv2.imshow("White Balanced Image", white_balanced_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()