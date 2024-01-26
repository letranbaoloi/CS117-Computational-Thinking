import cv2
classNames = ['Boot', 'Glove', 'Hardhat', 'Vest']
def plot_boxes(labels, cord, frame):
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
        bgr = (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 1)
        label = f"{int(row[4]*100)}"
        cv2.putText(frame, classNames[i], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    return frame

def draw_one_class(results, className, originFrame):
    labels, cord = results.xyxyn[0][:, -1].to('cpu').numpy(), results.xyxyn[0][:, :-1].to('cpu').numpy()
    label_store = []
    cord_store = []
    for i in range(len(labels)):
        label = int(labels[i]) # i= 0 label = 2
        if classNames[label] == className:
            label_store.append(label)
            cord_store.append(cord[i])
   
    newframe = plot_boxes(label_store, cord_store, originFrame)
    
    return newframe

def convert_to_classID(selected_class):
    #['Person', 'Vest', 'Blue Helmet', 'Red Helmet', 'White Helmet', 'Yellow Helmet']
    class_choice = []
    for c in selected_class:
        if c == 'Person':
            class_choice.append(0)
        elif c == "Vest":
             class_choice.append(1)
        elif c == "Blue":
             class_choice.append(2)
        elif c == "Red":
             class_choice.append(3)
        elif c == "White":
             class_choice.append(4)
        elif c == "Yellow":
             class_choice.append(5)
    return class_choice
    