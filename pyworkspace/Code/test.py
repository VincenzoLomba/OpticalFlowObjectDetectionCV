
import numpy as np
import cv2
import os

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
            height, width = frame.shape[:2]

            # Calcolo delle matrici per la compensazione
            h_array = np.arange(-int(height/2), int(height/2) + 1)
            w_array = np.arange(-int(width/2), int(width/2) + 1)

            h1 = np.matmul(
                np.delete(h_array, int(height/2)).reshape(height, 1),
                np.delete(w_array, int(width/2)).reshape(1, width)
            ) / f_c

            h2 = -f_c * np.ones((height, width)) - np.repeat(
                np.square(np.delete(w_array, int(width/2))).reshape(1, width) / f_c, 
                height, 
                axis=0
            )

            h5 = -h1

            # Calcolo del flusso indotto dal movimento ego
            u = h2 * np.deg2rad(yaw_rate_value)
            v = h5 * np.deg2rad(yaw_rate_value)

            yaw_prev = yaw_estimation
            yaw_rate_prev = yaw_rate_value

    return u, v, yaw_prev, yaw_rate_prev

def yaw_rate_estimate(yaw_rate_prev, yaw_angle_prev, yaw_angle):
    dt = 1.3
    alpha = 0.5

    # Normalizza gli angoli tra 0-360 gradi
    yaw_angle = yaw_angle % 360
    yaw_angle_prev = yaw_angle_prev % 360

    # Calcola l'errore di yaw
    yaw_error = yaw_angle - yaw_angle_prev
    if abs(yaw_error) > 180:
        yaw_error = yaw_error - 360 if yaw_error > 0 else yaw_error + 360

    # Filtro alpha-beta per la stima della velocità
    return alpha * yaw_rate_prev + (1 - alpha) * yaw_error / dt

def visualize_flow(img_in, flow_in, decimation=20, scale=2):  # Ridotti decimation e scale
    img_out = np.copy(img_in)
    h, w = img_out.shape[:2]
    
    # Genera la griglia di punti (più rada)
    y = np.arange(0, h, decimation)
    x = np.arange(0, w, decimation)
    xv, yv = np.meshgrid(x, y)
    
    # Estrae i vettori di flusso (scala ridotta)
    u = scale * flow_in[yv, xv, 0]
    v = scale * flow_in[yv, xv, 1]
    
    # Disegna le frecce (punta più corta e spessore ridotto)
    for i in range(xv.size):
        start = (int(xv.flat[i]), int(yv.flat[i]))
        end = (int(xv.flat[i] - u.flat[i]), int(yv.flat[i] - v.flat[i]))
        cv2.arrowedLine(
            img_out, start, end, 
            (255, 100, 30), 
            thickness=1,       # Spessore della linea
            tipLength=0.3      # Lunghezza della punta ridotta
        )
    
    return img_out
def process_video(input_path, output_path, yaw_list, focal_length=256):
    # Configurazione I/O video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Errore: Impossibile aprire il video di input")
        return

    # Ottieni parametri video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Inizializza video writer (codec MJPG per AVI)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Variabili di stato
    j = 0
    yaw_prev, yaw_rate_prev = None, None
    frame_prev = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Converti a scala di grigi per l'elaborazione
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calcola movimento ego
        if j >= len(yaw_list):
            print("Attenzione: yaw_list troppo corta!")
            break
            
        ego_vx, ego_vy, yaw_prev, yaw_rate_prev = rotational_ego_motion(
            yaw_list[j], yaw_prev, yaw_rate_prev, focal_length, gray
        )

        if frame_prev is not None and ego_vx is not None:
            # Calcola flusso ottico
            flow = cv2.calcOpticalFlowFarneback(
                prev=frame_prev, next=gray, flow=None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )

            # Compensa il moto ego
            #flow_compensated = flow - np.dstack((ego_vx, ego_vy))
            flow_compensated = flow

            # Visualizza e salva
            result_frame = visualize_flow(frame, flow_compensated)
            out.write(result_frame)
            
            # Mostra anteprima
            cv2.imshow('Output Preview', result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_prev = gray.copy()
        j += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Percorso del video di input
    input_video = "C:\\Users\\vince\\OneDrive\\Documenti\\Università\\Magistrale\\Second Year\\Topic Highlights\\Final Project\\pyworkspace\\Code\\Data\\rgb_output.avi"
    
    # Percorso del video di output
    output_video = "C:\\Users\\vince\\OneDrive\\Documenti\\Università\\Magistrale\\Second Year\\Topic Highlights\\Final Project\\pyworkspace\\Code\\Data\\GO.avi"
    
    # Lista valori yaw (adattare alla lunghezza del video)
    yaw_values = [84 + i*3 for i in range(300)]  # Esempio per 300 frame
    
    process_video(input_video, output_video, yaw_values)
    print("Elaborazione completata con successo!")

""""
"
import logger as log
import os
from opticalFlow import loadVideoFrames
import numpy as np

# Loading video frames
print(os.sep)
inputVideoPath = "C:\\Users\\vince\\Downloads\\rgb_output.avi"  # Replace with your video path
log.setActive("LOADER")
frames, fps, width, height = loadVideoFrames(inputVideoPath)
if not frames: exit()

vels = np.load("C:\\Users\\vince\\Downloads\\camera_velocities.npy")
print(vels.shape)
linear_x = vels[:, 0]
linear_y = vels[:, 1]
linear_z = vels[:, 2]
angular_x = vels[:, 3]
angular_y = vels[:, 4]
angular_z = vels[:, 5]
print("linear_x", linear_x.shape)
print("linear_y", linear_y.shape)
print("linear_z", linear_z.shape)
print("angular_x", angular_x.shape)
print("angular_y", angular_y.shape)
print("angular_z", angular_z.shape)

print(f"Loaded {len(frames)} frames with resolution {width}x{height} at {fps} FPS.")

"""