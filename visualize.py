import cv2 as cv
import numpy as np


def draw_trajectories(img, trajectories):
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    for t in trajectories:
        coords = [t.coords[0]] + [x for x in t.coords for _ in range(2)] + [t.coords[-1]]
        line = np.array(coords).reshape((-1, 2, 2))
        line = np.int32(line + 0.5)
        cv.polylines(vis, line, 1, (100, 210, 100))
        cv.circle(vis, (int(coords[-1][0]), int(coords[-1][1])), 2, (40, 40, 250), -1)

    return vis


def draw_flow(img, flow, step):
    h, w = img.shape[:2]
    y, x = np.mgrid[0:h:step, 0:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(vis, lines, 0, (255, 0, 0))

    return vis
