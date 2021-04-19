import cv2

import track
import detect
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=str, help="Video path")
    return parser.parse_args()


def main(video_path):
    cap = cv2.VideoCapture(video_path)

    ticks = 0

    lt = track.LaneTracker(2, 0.1, 500)
    ld = detect.LaneDetector(180)
    while cap.isOpened():
        precTick = ticks
        ticks = cv2.getTickCount()
        dt = (ticks - precTick) / cv2.getTickFrequency()

        _, frame = cap.read()
        if frame is None:
            break

        predicted = lt.predict(dt)

        lanes = ld.detect(frame)

        if predicted is not None:
            cv2.line(frame,
                     (predicted[0][0], predicted[0][1]),
                     (predicted[0][2], predicted[0][3]),
                     (0, 0, 255), 5)
            cv2.line(frame,
                     (predicted[1][0], predicted[1][1]),
                     (predicted[1][2], predicted[1][3]),
                     (0, 0, 255), 5)

        if lanes is not None:
            lt.update(lanes)

        cv2.imshow('', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    args = parse_args()
    main(args.path)
