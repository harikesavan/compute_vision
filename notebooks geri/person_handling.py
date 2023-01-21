import cv2 as cv
import math
import numpy as np

def path_to_center(path):
    x = path["xmin"]+(path["xmax"]-path["xmin"])/2
    y = path["ymin"]+(path["ymax"]-path["ymin"])/2
    x = int(x)
    y = int(y)
    return np.array([x,y])

class PersonHandler:
    
    def __init__(self):
        self.persons = []

    def handle_paths(self,paths,time_stamp):
        
        self.print_persons()
        for index, path in paths.iterrows():
            p_cent = path_to_center(path)
            min_dist_person = None
            min_dist = 100000

            for person in self.persons:
                dist_to_person = person.dist_to_person(p_cent[0],p_cent[1])
                if dist_to_person < min_dist:
                    min_dist = dist_to_person
                    min_dist_person = person 
            
            
            if min_dist_person == None:
                p = Person(15,p_cent,time_stamp)
                self.persons.append(p)

            if min_dist_person != None and min_dist_person.check_boundery_conditions(p_cent[0],p_cent[1], time_stamp):
                min_dist_person.move(path,time_stamp)
            else:
                p = Person(15,p_cent,time_stamp)
                self.persons.append(p)

    def render_paths(self,img):
        for person in self.persons:
            person.render_path(img)
    
    def print_persons(self):
        print(len(self.persons))


                





class Person:
    def __init__(self, v_max=4, p_t1=np.array([-1,-1]), t1 =0 ):
        self.v_max = v_max
        self.t1 = t1
        self.path = [p_t1]

    def dist_to_person(self,x,y):
        dx = x-self.path[-1][0]
        dy = y-self.path[-1][1]
        dist = math.sqrt(dx**2+dy**2)
        return dist

    def check_boundery_conditions(self, x,y, time_stamp):
        dist = self.dist_to_person(x,y)
        
        #after 8 seconds default to new person
        if ((time_stamp-self.t1)/30)>6:
            return False

        dt = int(math.log((time_stamp-self.t1)+1))+1

        if dist <= self.v_max*dt:
            return True
        else:
            return False

    def move(self, path, time_stamp):      
        x = int(path["xmin"]+(path["xmax"]-path["xmin"])/2)
        y = int(path["ymin"]+(path["ymax"]-path["ymin"])/2)
        xy = np.array([x,y])
        self.path.append(xy)
        self.t1 = time_stamp    
    
    def print_path(self):
        for p in self.path:
            print(p)

    def render_path(self, img):
        p0 = [-1,-1]
        
        for p in self.path:
            if p0[0] == -1:
                p0 = p
            else:         
                    
                cv.line(img,p0,p,(255,0,0),2)
                p0 = p

if __name__ == "__main__":
    import torch
    import cv2 as cv
    import numpy as np

    #load the torch model
    model = torch.hub.load('../yolov5', 'custom', path='../yolov5/best.pt', source='local')
    #set min confidence 
    model.conf = 0.05

    #other options are 
        #iou = 0.45  # NMS IoU threshold
        #agnostic = False  # NMS class-agnostic
        #multi_label = False  # NMS multiple labels per box
        #classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
        #max_det = 1000  # maximum number of detections per image
        #amp = False  # Automatic Mixed Precision (AMP) inference
    
    person_handler = None
    person_handler = PersonHandler()
    video_path = "../data_target_tracking/video_024.mp4"
    cap = cv.VideoCapture(video_path)
    time_stamp = 0
    out = cv.VideoWriter("test_vid",cv.VideoWriter_fourcc(*'MP4V'), 30, (640,512))

    while (cap.isOpened()):
        time_stamp += 1
        ret, frame = cap.read()  
        if ret:
            result = model(frame)
            img = np.array(result.render()).squeeze()
            person_handler.handle_paths(result.pandas().xyxy[0],time_stamp)
            person_handler.render_paths(img)
            out.write(img)

            
            cv.imshow("Test",img)
            #out.write(img)
            if cv.waitKey(25) & 0xFF == ord('q'):
                
                break
        else:
            break

    out.release()
    cap.release()
    #out.release()
    cv.destroyAllWindows()
