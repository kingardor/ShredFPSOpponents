import cv2
import time
import numpy as np
from mss import mss
from PIL import Image
from core import utils
from utils import normalize
import requests
import ast
from videocaptureasync import VideoCaptureAsync
import random
from args import imgsize, mon

class Detection:
    ''' Class to hold all detection related member methods and variables. '''

    def __init__(self):
        ''' Called when class object is created. '''

        self.IMAGE_H, self.IMAGE_W = imgsize, imgsize

        
        self.classes = utils.read_coco_names('./data/coco.names')
        self.num_classes = len(self.classes)
                
        self.colors = self.color_generator()
        self.windowName = 'Inference'
        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
    
    def color_generator(self):
        ''' Method to create a list of randomly generated colors. '''

        color_list = list()
        for i in range(0, len(self.classes)):
            color_list.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        return color_list
    
    def plot_results(self, result, frame):
        ''' Method to plot results on a frame. '''

        for i in range(0, len(result)):
            if not result[i][0] is None:
                
                boxes = result[i][2]
                scores = result[i][1]
                labels = result[i][0]

                if (not labels is None):
                    
                    for i in range(0, len(boxes)):

                        x1 = int(boxes[i][0])
                        y1 = int(boxes[i][1])
                        x2 = int(boxes[i][2])
                        y2 = int(boxes[i][3])

                        probability = "{0:.2f}".format(scores[i])
                        name = self.classes[labels[i]]
                        if not name == 'Enemy':
                            continue
                        color = (0, 0, 255)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.rectangle(frame, (x1, y1), (x1 + (len(name) + len(probability)) * 10, 
                                    y1 - 10) , color, -1, cv2.LINE_AA)
                        cv2.putText(frame, name + ':' + probability, (x1, y1), 
                                    cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        
        return frame

    def inference(self):
        ''' Method to perform detections on a source. '''

        
        sct = mss()
        
        return_value = True
        
        while True:
            sct.get_pixels(mon)
            frame = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
            frame = np.array(frame)

            n_frame, frame = normalize(frame, self.IMAGE_W)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            prev_time = time.time()

            response = requests.post('http://127.0.0.1:8000/detect', data=n_frame.tostring())
            result = ast.literal_eval(ast.literal_eval(response.content.decode('utf-8'))['Response'])

            frame = self.plot_results(result, frame)        
                                    
            curr_time = time.time()
            exec_time = curr_time - prev_time
            info = "time: %.2f ms" %(1000*exec_time)

            print('Time taken per frame: ', info, end='\r')
            
            cv2.imshow(self.windowName, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'): 
                cv2.destroyAllWindows()
                break

if __name__ == '__main__':

    detection = Detection.__new__(Detection)
    detection.__init__()
    detection.inference()
