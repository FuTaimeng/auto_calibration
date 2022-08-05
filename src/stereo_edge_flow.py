import cv2
import numpy as np
import os
import random
import sys


def write_ply(verts, fn):
    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        end_header
    '''
    verts = verts.reshape(-1, 3)
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f')


def contourFilter(in_edge, edge_lower_th=50, edge_upper_th=100):
    contours, hierarchy = cv2.findContours(in_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # edge_col = cv2.cvtColor(edge1, cv2.COLOR_GRAY2BGR)
    # edge_c = cv2.drawContours(edge_col, contours, -1, (0,0,255), 1)
    edge = np.zeros_like(in_edge)
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # cv2.drawContours(edge_col,[box],0,(0,0,255),1)
        w, h = rect[1]
        l = len(cnt)
        flag = False
        if w > edge_upper_th or h > edge_upper_th:
            flag = True
        elif w > edge_lower_th or h > edge_lower_th:
            if l < (w+h)*4:
                flag = True
        if flag:
            for pt in cnt:
                edge[pt[0][1], pt[0][0]] = 255
    
    return edge


def plot(in_img, pts, scale=1, radius=2, color=(0,0,255), thickness=2):
    pts = pts.reshape(-1, 2)
    img = cv2.resize(in_img, None, fx= scale, fy=scale)
    for pt in pts:
        cv2.circle(img, (int(pt[0]*scale), int(pt[1]*scale)), radius, color, thickness)
    return img


def plot_match(lf, lf_pts, rt, rt_pts, scale=1):
    w = lf.shape[1]
    lf_pts = lf_pts.reshape(-1, 2)
    rt_pts = rt_pts.reshape(-1, 2)
    img = cv2.hconcat((lf, rt))
    img = cv2.resize(img, None, fx=scale, fy=scale)
    for i in range(len(lf_pts)):
        lp = (int(lf_pts[i][0]*scale), int(lf_pts[i][1]*scale))
        rp = (int((rt_pts[i][0]+w)*scale), int(rt_pts[i][1]*scale))
        cv2.line(img, lp, rp, (255,0,0), 1)
        cv2.circle(img, lp, 2, (0,0,255), 2)
        cv2.circle(img, rp, 2, (0,0,255), 2)
    return img


def plot_flow(in_edge_lf, lf_pts, in_rt, rt_pts, scale=1):
    edge_lf = cv2.merge((np.zeros_like(in_edge_lf), in_edge_lf, in_edge_lf))
    img = cv2.addWeighted(edge_lf, 1, in_rt, 1, 0)
    img = cv2.resize(img, None, fx=scale, fy=scale)
    for i in range(len(lf_pts)):
        lp = (int(lf_pts[i][0][0]*scale), int(lf_pts[i][0][1]*scale))
        rp = (int(rt_pts[i][0][0]*scale), int(rt_pts[i][0][1]*scale))
        cv2.line(img, lp, rp, (255,0,0), 1)
        cv2.circle(img, rp, 2, (0,0,255), 2)
    return img


def mkdir(dir, clean=False):
    if not os.path.exists(dir):
        os.mkdir(dir)
    elif clean:
        for fname in os.listdir(dir):
            # print("del {}/{}".format(dir, fname))
            os.remove("{}/{}".format(dir, fname))


if __name__ == "__main__":
    path = "/home/tymon/DataSSD/shimizu/scene1"
    if len(sys.argv) == 2:
        path = sys.argv[1]
    max_points = 1000
    sample = 1

    left_path = path + "/image/left"
    right_path = path + "/image/right"
    thermal_path = path + "/image/thermal"
    cloud_path = path + "/stereo_clouds_edge"

    mkdir(cloud_path, clean=True)

    n_img = len(os.listdir(left_path))
    for i in range(0, n_img*sample, sample):
        print("Processing {}/{} ...".format(i, n_img*sample))
        th = cv2.imread("{}/{}.png".format(thermal_path, i), -1)
        lf = cv2.imread("{}/{}.png".format(left_path, i), -1)
        rt = cv2.imread("{}/{}.png".format(right_path, i), -1)

        scale =  float(th.shape[0]) / float(lf.shape[0])
        small_lf = cv2.resize(lf, None, fx=scale, fy=scale)
        small_rt = cv2.resize(rt, None, fx=scale, fy=scale)

        blur_lf = cv2.GaussianBlur(small_lf, (13,13), 0)
        blur_rt = cv2.GaussianBlur(small_rt, (13,13), 0)

        canny_lf = cv2.Canny(blur_lf, 50, 100)
        canny_rt = cv2.Canny(blur_rt, 50, 100)

        edge_lf = contourFilter(canny_lf, 30, 100)
        edge_rt = contourFilter(canny_rt, 30, 100)

        canny_th = cv2.Canny(th, 10, 30)

        edge_th = contourFilter(canny_th, 50, 100)

        gray_lf = cv2.cvtColor(lf, cv2.COLOR_BGR2GRAY)
        mask_lf = cv2.resize(edge_lf, (lf.shape[1], lf.shape[0]), interpolation=cv2.INTER_CUBIC)
        feature_params = dict(maxCorners=5000,
                            qualityLevel=0.01,
                            minDistance=10,
                            blockSize=15)
        lf_pts_good = cv2.goodFeaturesToTrack(gray_lf, mask=mask_lf, **feature_params)
        # lf_pts_good = np.array([], dtype=np.float32).reshape(-1, 1, 2)

        good_mask = plot(np.ones_like(edge_lf), lf_pts_good*scale, radius=0, thickness=20)
        residual_edge_lf = cv2.bitwise_and(edge_lf, edge_lf, mask=good_mask)

        # lf_pts_others = (cv2.findNonZero(residual_edge_lf) / scale).astype(np.float32)
        # lf_pts_others = lf_pts_others[[i%3==0 for i in range(len(lf_pts_others))]]
        lf_pts_others = np.array([], dtype=np.float32).reshape(-1, 1, 2)

        lf_pts = np.concatenate((lf_pts_good, lf_pts_others))
        print("good:{}, others:{}, total:{}".format(len(lf_pts_good), len(lf_pts_others), len(lf_pts)))

        rt_pts, flow_mask, err = cv2.calcOpticalFlowPyrLK(lf,
                                                        rt,
                                                        lf_pts,
                                                        None,
                                                        winSize=(101,101),
                                                        maxLevel=5)

        lf_pts = lf_pts[flow_mask.astype(bool)].reshape(-1,1,2)
        rt_pts = rt_pts[flow_mask.astype(bool)].reshape(-1,1,2)
        print("flow:{}".format(len(lf_pts)))

        F, ransac_mask = cv2.findFundamentalMat(lf_pts, rt_pts, cv2.FM_RANSAC)

        lf_pts = lf_pts[ransac_mask.astype(bool)].reshape(-1,1,2)
        rt_pts = rt_pts[ransac_mask.astype(bool)].reshape(-1,1,2)
        print("ransac:{}".format(len(lf_pts)))

        match = plot_match(lf, lf_pts, rt, rt_pts, 1)
        large_edge_lf = cv2.resize(edge_lf, (lf.shape[1], lf.shape[0]), interpolation=cv2.INTER_CUBIC)
        flow = plot_flow(large_edge_lf, lf_pts, rt, rt_pts, 1)

        P1 = np.loadtxt("{}/image/P1.dat".format(path))
        P2 = np.loadtxt("{}/image/P2.dat".format(path))
        K = P1[:3, :3]
        pts_4d = cv2.triangulatePoints(P1, P2, lf_pts, rt_pts).T
        pts_3d = pts_4d[:, :3] / pts_4d[:, 3:4]

        dist_lower_th = 0.5
        dist_upper_th = 50
        p3d_mask = [pts_3d[i][2]>dist_lower_th and pts_3d[i][2]<dist_upper_th for i in range(len(pts_3d))]
        pts_3d = pts_3d[p3d_mask]
        print("3d:{}".format(len(pts_3d)))

        if len(pts_3d) > max_points:
            sel = np.random.choice(len(pts_3d), size=max_points, replace=False)
            pts_3d = pts_3d[sel, :]

        write_ply(pts_3d, "{}/{}.ply".format(cloud_path, i))
        print("cloud:{}".format(len(pts_3d)))

