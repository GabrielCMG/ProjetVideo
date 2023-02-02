import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=1000,
                      qualityLevel=0.01,
                      minDistance=5,
                      blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (1000, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

roi = cv2.selectROI("", old_gray)

img = old_gray[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]

name_detector = 'Shi-Tomasi'

if name_detector == 'Shi-Tomasi':
    # Shi-Tomasi corner detector
    p0 = cv2.goodFeaturesToTrack(img, mask=None, **feature_params)
elif name_detector == 'Harris':
    # Harris detector
    blockSize = 2
    apertureSize = 3
    k = 0.04
    thresh = 102
    # Detecting corners
    p0 = cv2.cornerHarris(img, blockSize, apertureSize, k)
    p0_norm = np.empty(p0.shape, dtype=np.float32)
    cv2.normalize(p0, p0_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    p0_norm_scaled = cv2.convertScaleAbs(p0_norm)

    p0_tmp = []

    for i in range(p0_norm.shape[0]):
        for j in range(p0_norm.shape[1]):
            if int(p0_norm[i, j]) > thresh:
                p0_tmp.append([j, i])

    p0 = p0_tmp
    p0 = np.array(p0, dtype='float32')
    p0 = np.reshape(p0, [p0.shape[0], 1, 2])
else:
    # Detection des contours par filtre de Canny
    kp = cv2.Canny(img, 100, 200)

    p0_tmp = []
    thresh = 150

    for i in range(kp.shape[0]):
        for j in range(kp.shape[1]):
            if int(kp[i, j]) > thresh:
                p0_tmp.append([j, i])

    p0 = p0_tmp
    p0 = np.array(p0, dtype='float32')
    p0 = np.reshape(p0, [p0.shape[0], 1, 2])

# Change position of pixel
p0[:, :, 0] += roi[0]
p0[:, :, 1] += roi[1]

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
while 1:
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

    # Update frame
    x1, y1, x2, y2 = np.quantile(good_new[:, 0], 0.1), np.quantile(good_new[:, 1], 0.1), \
        np.quantile(good_new[:, 0], 0.9), np.quantile(good_new[:, 1], 0.9)

    print(x1, x2, y1, y2)

    frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0))

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        # mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 2, (255, 0, 0), -1)

    # img = cv2.add(frame, mask)

    cv2.imshow('frame', frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
