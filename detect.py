from __future__ import division

import cv2
import numpy as np


def standard_hough(img, init_vote):
    # Hough transform wrapper to return a list of points like PHough does
    lines = cv2.HoughLines(img, 1, np.pi/180, init_vote)
    points = [[]]
    for l in lines:
        for rho, theta in l:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*a)
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*a)
            points[0].append((x1, y1, x2, y2))
    return points


def scale_points(x1, y1, x2, y2, horizon_height):
    # update x2 to match a scaled y2 to the drawing horizon
    if x1 == x2:
        return x2
    m = (y2-y1)/(x2-x1)
    scaled_x2 = ((horizon_height-y1)/m) + x1
    return int(scaled_x2)


def scale_line(x1, y1, x2, y2, horizon_height):
    # scale the farthest point of the segment to be on the drawing horizon
    if y1 < y2:
        x1 = scale_points(x2, y2, x1, y1, horizon_height)
        y1 = horizon_height
    else:
        x2 = scale_points(x1, y1, x2, y2, horizon_height)
        y2 = horizon_height

    return x1, y1, x2, y2


def base_distance(x1, y1, x2, y2, width):
    # compute the point where the give line crosses the base of the frame
    # return distance of that point from center of the frame
    if x2 == x1:
        return (width*0.5) - x1
    m = (y2-y1)/(x2-x1)
    c = y1 - m*x1
    base_cross = -c/m
    return (width*0.5) - base_cross


def draw_bound(frame, points, draw_horizon, theta_cross, line_width=5):
    # draw lane bound in blue if it is not being crossed, in red otherwise
    if points is not None:
        x1, y1, x2, y2 = points[0], points[1], points[2], points[3]
        theta = np.abs(np.arctan2((y2-y1), (x2-x1)))
        x1, y1, x2, y2 = scale_line(x1, y1, x2, y2, draw_horizon)
        if theta_cross < theta:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), line_width)
        else:
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), line_width)


def main(video_path, prob_hough=True, debug=False):
    cap = cv2.VideoCapture(video_path)

    hough_vote = 50
    horizon = 160  # horizon height to define the ROI relative to the road
    draw_horizon = 170  # horizon height to extend short lines drawing
    roiy_begin = horizon
    roix_begin = 0
    roi_theta = 0.30  # angle WRT horizon: lines with smaller angle than this are discarded when looking for lane boundaries
    theta_cross = 1.00  # angle WRT horizon: lines with theta_cross < angle are being crossed by the car

    while cap.isOpened():
        ret, frame = cap.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        roiy_end = frame.shape[0]
        roix_end = frame.shape[1]
        roi = img[roiy_begin:roiy_end, roix_begin:roix_end]  # work only on the lower part of the frame
        blur = cv2.medianBlur(roi, 5)
        contours = cv2.Canny(blur, 60, 120)

        if prob_hough:
            lines = cv2.HoughLinesP(contours, 1, np.pi/180, hough_vote, minLineLength=30, maxLineGap=100)
        else:
            lines = standard_hough(contours, 50)

        if lines is not None:
            # find nearest lines to center
            lines = lines+np.array([0, horizon, 0, horizon]).reshape((1,1,4))  # scale points from ROI coordinates to full frame coordinates
            left_bound = None
            right_bound = None
            for l in lines:
                # find the rightmost line of the left half of the frame and the leftmost line of the right half
                for x1, y1, x2, y2 in l:
                    theta = np.abs(np.arctan2((y2-y1), (x2-x1)))  # line angle WRT horizon
                    if theta > roi_theta:  # ignore lines with a small angle WRT horizon
                        dist = base_distance(x1, y1, x2, y2, frame.shape[1])
                        if left_bound is None and dist < 0:
                            left_bound = (x1, y1, x2, y2)
                            left_dist = dist
                        elif right_bound is None and dist > 0:
                            right_bound = (x1, y1, x2, y2)
                            right_dist = dist
                        elif left_bound is not None and 0 > dist > left_dist:
                            left_bound = (x1, y1, x2, y2)
                            left_dist = dist
                        elif right_bound is not None and 0 < dist < right_dist:
                            right_bound = (x1, y1, x2, y2)
                            right_dist = dist

            draw_bound(frame, left_bound, draw_horizon, theta_cross)
            draw_bound(frame, right_bound, draw_horizon, theta_cross)

            if debug:
                # draw all lines detected by Hough Transform
                for l in lines:
                    for x1, y1, x2, y2 in l:
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        cv2.imshow('', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main('data/V4_cross2.mp4')
