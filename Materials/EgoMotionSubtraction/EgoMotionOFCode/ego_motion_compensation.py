import numpy as np
import glob
import cv2


def rotational_ego_motion(yaw, yaw_prev, yaw_rate_prev, focal_length, frame, is_unreal=True):
    u = None
    v = None

    yaw_estimation = yaw
    if yaw_prev is None:
        yaw_prev = yaw_estimation
    else:
        if yaw_rate_prev is None:
            yaw_rate_prev = yaw_estimation - yaw_prev
            yaw_prev = yaw
        else:
            if is_unreal:
                yaw_rate_value = yaw_estimation - yaw_prev
            else:
                yaw_rate_value = yaw_rate_estimate(yaw_rate_prev, yaw_prev, yaw)

            f_c = focal_length

            # add rotational(yaw) ego-motion subtraction
            # calc matrix of H
            # x = 0 1 2 ... frame.shape[0]
            # y = 0 1 2 ... frame.shape[1]
            # axis = 0, down
            # axis = 1, right
            width = frame.shape[1]
            height = frame.shape[0]

            h_array = np.arange(-int(height / 2), int(height / 2) + 1)
            w_array = np.arange(-int(width / 2), int(width / 2) + 1)

            h1 = (np.matmul(np.delete(h_array, int(height / 2)).reshape(height, 1),
                            np.delete(w_array, int(width / 2)).reshape(1, width))) / f_c
            h2 = - f_c * np.ones((height, width)) - \
                 (np.repeat(np.square(np.delete(w_array, int(width / 2))).reshape(1, width) / f_c, height,
                            axis=0))
            h3 = (np.repeat(np.arange(-int(height / 2), int(height / 2)).reshape(height, 1), width, axis=1))
            h4 = f_c * np.ones((width, height)) + \
                 np.transpose(np.repeat(np.square(np.delete(h_array, int(height / 2))).reshape(height, 1) / f_c, width,
                                        axis=1))
            h5 = -h1
            h6 = -np.repeat(np.arange(-int(width / 2), int(width / 2)).reshape(1, int(width)), height, axis=0)

            # rotational ego-motion
            u = h2 * np.deg2rad(yaw_rate_value)
            v = h5 * np.deg2rad(yaw_rate_value)

            yaw_prev = yaw_estimation
            yaw_rate_prev = yaw_rate_value

    return u, v, yaw_prev, yaw_rate_prev


def yaw_rate_estimate(yaw_rate_prev, yaw_angle_prev, yaw_angle):
    dt = 1.3
    # counterclockwise is postive for yaw_rate
    alpha = 0.5

    if yaw_angle < 0:
        yaw_angle = yaw_angle + 360
    if yaw_angle_prev < 0:
        yaw_angle_prev = yaw_angle_prev + 360

    yaw_error = yaw_angle - yaw_angle_prev
    if abs(yaw_error) < 180:
        yaw_value_est = alpha * yaw_rate_prev + (1 - alpha) * yaw_error / dt
    else:
        if yaw_error > 0:
            yaw_value_est = alpha * yaw_rate_prev + (1 - alpha) * (yaw_error - 360) / dt
        else:
            yaw_value_est = alpha * yaw_rate_prev + (1 - alpha) * (360 + yaw_error) / dt

    return yaw_value_est


def visualize_flow(img_in, flow_in, decimation=15, scale=10):
    """
    Function to visualize optical flow over the original image
    :param img_in: rgb image - if none is provided, the flow is shown alone
    :param flow_in: this is the dense optical flow
    :param decimation: how dense the arrows are shown in the vector field
    :param scale: scale of the magnitude in the vector field
    :param method: vector field or hsv plot
    :return: void
    """
    img_out = np.copy(img_in)
    # quiver plot
    y = list(range(int(img_out.shape[0])))[0::decimation]
    x = list(range(int(img_out.shape[1])))[0::decimation]
    xv, yv = np.meshgrid(x, y)
    u = scale * flow_in[yv, xv, 0]
    v = scale * flow_in[yv, xv, 1]
    start_points = np.array([xv.flatten(), yv.flatten()]).T.astype(int).tolist()
    end_points = np.array([xv.flatten() - u.flatten(), yv.flatten() - v.flatten()]).T.astype(int).tolist()
    for i in range(len(start_points)):
        cv2.arrowedLine(img_out, tuple(start_points[i]), tuple(end_points[i]), [255, 100, 30], thickness=1)
    return img_out


if __name__ == '__main__':

    # focal length of your camera
    focal_length = 256
    j = 0
    yaw_prev, yaw_rate_prev = None, None
    # yaw angle for UGV
    # yaw_list = np.rad2deg([-0.0528073, -0.0610591, -0.0594091, -0.0656803, -0.0646896, -0.0594091, -0.0597391, -0.0610591,
    #             -0.0617192, -0.060729, -0.0590788, -0.0600694, -0.0610591, -0.0603991, -0.0630397, -0.0630397,
    #             -0.0630397, -0.0630397])

    # yaw_list = np.rad2deg([0.434715, 0.913947, 1.06643, 1.29284, 1.43311])

    # yaw angle for unreal engine
    yaw_list = [84, 87, 90, 93, 96, 99, 102]

    for i in glob.glob("data/test_unreal1/image/*.jpg"):
        # read images and pose
        frame_rgb = cv2.imread(i)
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)

        ego_vx, ego_vy, yaw_prev, yaw_rate_prev = rotational_ego_motion(yaw_list[j], yaw_prev, yaw_rate_prev,
                                                                        focal_length, frame)
        if j > 0:
            flow_out = cv2.calcOpticalFlowFarneback(prev=frame_prev,
                                                    next=frame,
                                                    flow=None,
                                                    pyr_scale=0.5,
                                                    levels=3,
                                                    winsize=15,
                                                    iterations=3,
                                                    poly_n=5,
                                                    poly_sigma=1.2,
                                                    flags=0)
            if ego_vx is None:
                pass
            else:
                flow_comp = np.zeros_like(flow_out)
                flow_ego = np.zeros_like(flow_out)
                # ego_vy = ego_vy.T
                flow_trans_x = flow_out[:, :, 0] - ego_vx
                flow_trans_y = flow_out[:, :, 1] - ego_vy
                flow_comp[:, :, 0] = flow_trans_x
                flow_comp[:, :, 1] = flow_trans_y
                flow_ego[:, :, 0] = ego_vx
                flow_ego[:, :, 1] = ego_vy

                flow_total = visualize_flow(frame_rgb, flow_out, decimation=30, scale=3)
                flow_comp = visualize_flow(frame_rgb, flow_comp, decimation=30, scale=3)
                flow_ego = visualize_flow(frame_rgb, flow_ego, decimation=30, scale=3)
                cv2.imshow('total optical flow', flow_total)
                cv2.imshow('compensated optical flow', flow_comp)
                cv2.waitKey(100)
                cv2.imwrite('comp_{}.jpg'.format(j), flow_comp)
                cv2.imwrite('total_{}.jpg'.format(j), flow_total)
                cv2.imwrite('ego_{}.jpg'.format(j), flow_ego)

        frame_prev = frame
        j += 1
