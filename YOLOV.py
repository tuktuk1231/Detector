import cv2 as cv
Conf_threshold = 0.98
MMS_threshold = 0.4
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
class_name = []

with open('arm.names', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]
net = cv.dnn.readNet('arm_6000.weights', 'arm.cfg')

net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)



cap = cv.VideoCapture('videoplayback.mp4')

while True:
    ret, frame = cap.read()
    if ret==False:
        break
    classes, scores, boxes = model.detect(frame, Conf_threshold, MMS_threshold)
    for (classid, score, box) in zip(classes, scores, boxes):
        color=COLORS[int(classid) % len(COLORS)]
        label = '%s: %f' % (class_name[classid[0]], scores)
        cv.rectangle(frame, box, color, 1)
        cv.putText(frame, label, (box[0], box[1]-10), cv.FONT_HERSHEY_COMPLEX, 0.5, color, 2)

    cv.imshow('frame', frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv.destroyAllWindows()