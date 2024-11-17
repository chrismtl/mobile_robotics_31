import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('Photos/Green Background/bloc_1.jpeg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the original image
plt.figure(figsize=(8, 6))
plt.imshow(image_rgb)
plt.axis('off')
plt.title('Original Image')
plt.show()

# Convert BGR to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#print(hsv_image)

# Define ranges for red color in HSV
lower_red1 = np.array([0, 50, 250])
upper_red1 = np.array([10, 100, 255])
lower_red2 = np.array([170, 50, 250])
upper_red2 = np.array([180, 100, 255])

# Define ranges for blue color in HSV
lower_blue1 = np.array([100, 20, 225])
upper_blue1 = np.array([110, 30, 255])
lower_blue2 = np.array([100, 100, 225])
upper_blue2 = np.array([110, 130, 255])

# Define ranges for green color in HSV
lower_green = np.array([40, 100, 150])
upper_green = np.array([75, 255, 255])

# Define ranges for yellow color in HSV
lower_yellow = np.array([109, 0, 225])
upper_yellow = np.array([111, 7, 255])

# Function to calculate centroids with sensitivity parameter
def calculate_centroids(mask, min_area_threshold):
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []
    for contour in contours:
        # Calculate area of contour
        area = cv2.contourArea(contour)
        if area > min_area_threshold:
            # Calculate centroid using moments
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centroids.append((cx, cy))

    return centroids

# Create masks for each color
red_mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
red_mask = cv2.bitwise_or(red_mask1, red_mask2)
blue_mask1 = cv2.inRange(hsv_image, lower_blue1, upper_blue1)
blue_mask2 = cv2.inRange(hsv_image, lower_blue2, upper_blue2)
blue_mask = cv2.bitwise_or(blue_mask1, blue_mask2)
yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

# Apply masks to the original image
segmented_red = cv2.bitwise_and(image, image, mask=red_mask)
segmented_blue = cv2.bitwise_and(image, image, mask=blue_mask)
segmented_green = cv2.bitwise_and(image, image, mask=green_mask)
segmented_yellow = cv2.bitwise_and(image, image, mask=yellow_mask)

# Convert segmented images from BGR to RGB for display
segmented_red_rgb = cv2.cvtColor(segmented_red, cv2.COLOR_BGR2RGB)
segmented_blue_rgb = cv2.cvtColor(segmented_blue, cv2.COLOR_BGR2RGB)
segmented_green_rgb = cv2.cvtColor(segmented_green, cv2.COLOR_BGR2RGB)
segmented_yellow_rgb = cv2.cvtColor(segmented_yellow, cv2.COLOR_BGR2RGB)

# Sensitivity parameter (adjust as needed)
min_area_threshold = 300  # Adjust this value based on your sensitivity needs

# Calculate centroids for each color with sensitivity parameter
centroids_red = calculate_centroids(red_mask, min_area_threshold)
centroids_blue = calculate_centroids(blue_mask, min_area_threshold)
centroids_green = calculate_centroids(green_mask, min_area_threshold)
centroids_yellow = calculate_centroids(yellow_mask, min_area_threshold)

# Display the segmented images with centroids
plt.figure(figsize=(15, 6))

plt.subplot(2, 2, 1)
plt.imshow(segmented_red_rgb)
plt.scatter([cx for cx, cy in centroids_red], [cy for cx, cy in centroids_red], color='r', marker='o', s=30)
plt.axis('off')
plt.title('Red Segmentation')

plt.subplot(2, 2, 2)
plt.imshow(segmented_blue_rgb)
plt.scatter([cx for cx, cy in centroids_blue], [cy for cx, cy in centroids_blue], color='b', marker='o', s=30)
plt.axis('off')
plt.title('Blue Segmentation')

plt.subplot(2, 2, 3)
plt.imshow(segmented_green_rgb)
plt.scatter([cx for cx, cy in centroids_green], [cy for cx, cy in centroids_green], color='g', marker='o', s=30)
plt.axis('off')
plt.title('Green Segmentation')

plt.subplot(2, 2, 4)
plt.imshow(segmented_yellow_rgb)
plt.scatter([cx for cx, cy in centroids_yellow], [cy for cx, cy in centroids_yellow], color='y', marker='o', s=30)
plt.axis('off')
plt.title('Yellow Segmentation')

plt.tight_layout()
plt.show()


# Print the coordinates of the centroids
print("Red centroids:")
for i, (cx, cy) in enumerate(centroids_red):
    print(f"Red Object {i+1}: Center at ({cx}, {cy})")

print("\nBlue centroids:")
for i, (cx, cy) in enumerate(centroids_blue):
    print(f"Blue Object {i+1}: Center at ({cx}, {cy})")

print("\nGreen centroids:")
for i, (cx, cy) in enumerate(centroids_green):
    print(f"Green Object {i+1}: Center at ({cx}, {cy})")

print("\nYellow centroids:")
for i, (cx, cy) in enumerate(centroids_yellow):
    print(f"Yellow Object {i+1}: Center at ({cx}, {cy})")
