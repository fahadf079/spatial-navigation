import cv2
import numpy as np
import glob

# Checkerboard dimensions (inner corners, not squares)
CHECKERBOARD = (9, 6)

# Prepare object points like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

images = glob.glob('calib_images/*.jpg')
print(f"Found {len(images)} images")

successful = 0
img_shape = None

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_shape = gray.shape[::-1]

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        successful += 1

print(f"Checkerboard detected in {successful} out of {len(images)} images")

if successful < 5:
    print("Not enough successful detections to calibrate reliably. Need at least 5-10.")
else:
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )

    print("\nCamera matrix (focal length + optical center):")
    print(camera_matrix)

    print("\nDistortion coefficients:")
    print(dist_coeffs)

    np.savez('camera_calibration.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    print("\nSaved calibration data to camera_calibration.npz")