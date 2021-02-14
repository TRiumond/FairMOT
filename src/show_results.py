import pandas as pd
import cv2
import os

cap = cv2.VideoCapture("../models/fra.vs.cro.first.goal.mp4")
result_path = '../models/results/'
paths = 'cqavjvcl-%d.csv'
c = 1
while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    df = pd.read_csv(os.path.join(result_path, paths % c))
    for _, i in df.iterrows():
        item = i.to_dict()

        if item['id'] == 341:
            print(item)
            bbox = [item['x'], item['y'], item['h'], item['w']]
            x1, y1, h, w = bbox
            x1, y1, x2, y2 = tuple(map(int, ( x1-w, y1-h, x1 + w/2, y1 + h/2)))
            # y[:, 0] = (x[:, 0] - x[:, 2] / 2)
            # y[:, 1] = (x[:, 1] - x[:, 3] / 2)
            # y[:, 2] = (x[:, 0] + x[:, 2] / 2)
            # y[:, 3] = (x[:, 1] + x[:, 3] / 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
            cv2.putText(frame, '{}'.format(item['id']), (x1, y1 + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255),
                         thickness=1)
    c += 1
    # Display the resulting frame
    cv2.imshow('frame', frame)
    cv2.waitKey(10)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
