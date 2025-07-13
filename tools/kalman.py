import cv2
import numpy as np

'''
Kiến thức nền:
Kalman Filter gồm:
    1. Dự đoán trạng thái X(t+1) và sai số của trạng thái đó P(t+1)
    2. Đo lường trạng thái thật Z(t+1)
    3. Kết hợp (1), (2) để ước lượng trạng thái X(t+1) và sai số của trạng thái P(t+1)


F: ma trận chuyển đổi trạng thái => dùng để chuyển đổi trạng thái (t) sang (t + 1)
P: ma trận phương sai của ước lượng
Q: ma trận phương sai của dự đoán 
R: ma trận phương sai của đo lường

P càng lớn => ước lượng sai nhiều => không thể tin tưởng vào kết quả của Kalman
Q càng lớn => P lớn => K lớn => Dự đoán sai nhiều => tin vào đo lường hơn
R càng lớn => K nhỏ => Đo lường sao nhiều => tin vào dự đoán hơn

'''

class Kalman:
    def __init__(self, method='xy-xyv'):
        dt = 1
        if method == 'xy-xyv':
            state_num_param = 4
            '''
            measure: Px, Py
            state (t): Px, Py, Vx, Vy
            => state (t+1)
                Px = Px + Vx*dt
                Py = Py + Vy*dt
                Vx = Vx
                Vy = Vy
            '''
            self.F = np.matrix([ [1, 0, dt, 0],
                                [0, 1, 0,  dt],
                                [0, 0, 1,  0],
                                [0, 0, 0,  1],
                                ])
        elif method == 'xyv-xyva':

            '''
            measure: Px, Py, Vx, Vy
            state (t): Px, Py, Vx, Vy, Ax, Ay
            => state (t+1)
                Px = Px + Vx*dt + 0.5*Ax*dt^2
                Py = Py + Vy*dt + 0.5*Ay*dt^2
                Vx = Vx + Ax*dt
                Vy = Vy + Ax*dt
                Ax = Ax
                Ay = Ay
            '''

            state_num_param = 6
            self.F = np.matrix([ [1, 0, dt, 0,  0.5*dt**2, 0        ],
                                 [0, 1, 0,  dt, 0,         0.5*dt**2],
                                 [0, 0, 1,  0,  dt,        0        ],
                                 [0, 0, 0,  1,  0,         dt       ],
                                 [0, 0, 0,  0,  1,         0        ],
                                 [0, 0, 0,  0,  0,         1        ],
                                ])


        self.P = np.eye(state_num_param) * 1
        
        self.H = np.eye(state_num_param)[:state_num_param-2]
        
        self.I = np.matrix(np.eye(state_num_param))
        
        self.R = np.eye(state_num_param-2) * 1 # phương sai của đo lường
        self.Q = np.eye(state_num_param) * 0.1 # phương sai của dự đoán

        self.state = np.zeros(state_num_param).reshape((-1, 1))
        self.missing_time = 0

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state
    
    def update(self, measure):
        measure = np.matrix(measure)
        self.y = measure - self.H@self.state
        self.K = self.P @ self.H.T @ np.linalg.pinv(self.H @ self.P @ self.H.T + self.R)

        self.state = self.state + self.K@self.y
        self.P = (self.I - self.K@self.H)@self.P

        return self.state

    def get_estimate(self):
        return self.state[:2]  # Return only x, y
    
def createimage(w,h):
    img = np.ones((w,h,3),np.uint8)*255
    return img

if __name__ == '__main__':
    method = 'xyv-xyva'
    kalman = Kalman(method=method)

    with open('frames/center_noise.txt', 'r') as f:
        data = f.readlines()
    with open('frames/center.txt', 'r') as f:
        gt = f.readlines()

    dt = 1
    old_measure = None
    frame = createimage(200, 700)

    predicts = []
    measures = []
    gts = []

    for k, dta in enumerate(data):
        dta = np.array(dta.strip().split()).astype(int)
        frame_id = dta[0]
        measure = dta[1:] # xyxy
        _gt = np.array(gt[k].strip().split()).astype(int)[1:]

        if method == 'xy-xyv':
            measure = measure.reshape((2, 1))

        elif method == 'xyv-xyva':
            if old_measure is None:
                measure = np.vstack([measure, np.zeros(2)])
            else:
                deltaP = measure - old_measure[:2].reshape((-1))
                mV = deltaP/dt
                measure = np.vstack([measure, mV])

            measure = measure.reshape((-1, 1))

        prediction = kalman.predict()
        kalman.update(measure)
        x, y = int(measure[0]), int(measure[1])
        gtx, gty = int(_gt[0]), int(_gt[1])
        pred_x, pred_y = int(prediction[0]), int(prediction[1])
        corrected_x, corrected_y = int(kalman.state[0]), int(kalman.state[1])

        predicts.append((corrected_x, corrected_y))
        measures.append((x, y))
        gts.append((gtx, gty))

        prediction = kalman.predict()
        kalman.update(measure)  

        old_measure = measure
        # print(f'{gtx} {gty} | {pred_x} {pred_y} | {x} {y}')

    def mse(pt1, pt2):
        x1, y1 = pt1
        x2, y2 = pt2
        return ((x1 - x2)**2 + (y1 - y2)**2) / 2

    frame = createimage(200, 700)
    measure_errors = []
    predict_errors = []
    for i in range(1, len(predicts)):
        pred = predicts[i]
        meas = measures[i]
        gt = gts[i]

        opred = predicts[i-1]
        omeas = measures[i-1]
        ogt = gts[i-1]

        measure_errors.append(mse(meas, gt))
        predict_errors.append(mse(pred, gt))

        frame = cv2.line(frame, opred, pred, (255, 0, 0), 1) 
        frame = cv2.line(frame, omeas, meas, (0, 0, 255), 1) 
        frame = cv2.line(frame, ogt, gt, (0, 255, 0), 1) 

    cv2.imwrite('result.jpg', frame)
    print(f'Measure MSE: {np.mean(measure_errors):.2f} | Predict MSE: {np.mean(predict_errors):.2f}')

