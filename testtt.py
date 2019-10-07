import tensorflow as tf
import os
import cv2 
from PIL import Image
import numpy as np
import image_handling 
from array import array
#from object_detectionBurn import ObjectDetection as object_detectionBurn
#from object_detectionMold import ObjectDetection as object_detectionMold
from object_detection import ObjectDetection
import time


#-------------new--------------------
class TFObjectDetection(ObjectDetection):
    """Object Detection class for TensorFlow
    """

    def __init__(self, graph_def, labels):
        super(TFObjectDetection, self).__init__(labels)
        self.graph = tf.Graph()
        with self.graph.as_default():
            input_data = tf.placeholder(tf.float32, [1, None, None, 3], name='Placeholder')
            tf.import_graph_def(graph_def, input_map={"Placeholder:0": input_data}, name="")

    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float)[:, :, (2, 1, 0)]  # RGB -> BGR
        output_tensor = sess3.graph.get_tensor_by_name('model_outputs:0')
        outputs = sess3.run(output_tensor, {'Placeholder:0': inputs[np.newaxis, ...]})
        return outputs[0]

# Load a TensorFlow model
graph_def = tf.GraphDef()
with tf.gfile.FastGFile('model.pb', 'rb') as f:
    graph_def.ParseFromString(f.read())
    
# Load labels
with open('labels.txt', 'r') as f:
    labels = [l.strip() for l in f.readlines()]

od_model = TFObjectDetection(graph_def, labels)





# Remove some warnings from output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'








#----------------------------放置攝影機長寬等，關掉鏡頭後要儲存的影片設定------------------
cap = cv2.VideoCapture('input.mp4')
# 声明编码器和创建VideoWrite对象
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter('output.avi',fourcc, 30.0, (width, height))


# checks whether frames were extracted 
success = 1


# These names are part of the model and cannot be changed.分類模型的設定
output_layer = 'loss:0'
input_node = 'Placeholder:0'


 



    
graphObjectDetectBoth = tf.Graph()    
with graphObjectDetectBoth.as_default():
    input_data = tf.placeholder(tf.float32, [1, None, None, 3], name='Placeholder')
    tf.import_graph_def(graph_def, input_map={"Placeholder:0": input_data}, name="")
with tf.Session(graph=graphObjectDetectBoth) as sess3:



    while success:    
        start_time = time.time() # start time of the loop    
        # vidObj object calls read 
        # function extract frames 
        success, image = cap.read()
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            
        
        # Load the test images from a file

        #image = Image.open(image)
        #image = array(img).reshape(1, 227 * 227 * 3)


        # Update orientation based on EXIF tags, if the file has orientation info
    

        # Convert to OpenCV (Open Source Computer Vision) format
    

        # If the image has either w or h greater than 1600 we resize it down respecting
        # aspect ratio such that the largest dimension is 1600


        

    


        cv2.imshow('imageFinal',image)

            
        image = Image.fromarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # npmarray to jpgfile 
        #
        predictions = od_model.predict_image(image) 
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for i in predictions:
            if(i['probability'] > 0.3):
                x = int(i['boundingBox']['left']*640)
                y = int(i['boundingBox']['top']*480)
                w = i['boundingBox']['width']*640
                h = i['boundingBox']['height']*480
                wid = int(x+w)
                hie = int(y+h)
                text = i['tagName']+str('%.3f' %  (i['probability']*100))+"%"
                image = np.array(image)
                image = cv2.rectangle(image, (x, y), (wid, hie), (0, 255, 0), 5)             
                image = cv2.putText(image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,255,100), 2)
                cv2.imshow('imageFinal',image)
                #存成影片
                out.write(image)
            print(i)
        #fps = cap.get(cv2.CAP_PROP_FPS)
        #print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
        print("FPS: ", 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop    


                    
        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

           

print("width:",width, "height:", height)
# 釋放攝像頭資源
cap.release()
out.release()
cv2.destroyAllWindows() 



