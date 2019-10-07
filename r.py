import tensorflow as tf
import os
import cv2 
from PIL import Image
import numpy as np
import image_handling 
from array import array
from object_detectionBurn import ObjectDetection as object_detectionBurn
from object_detectionMold import ObjectDetection as object_detectionMold



#MODEL_FILENAME = 'modelDetect.pb'
#LABELS_FILENAME = 'labelsDetect.txt'




#---------------------物件偵測麵包烤焦的模型-----------------------
class TFObjectDetection(object_detectionBurn):
    """Object Detection class for TensorFlow
    """
    def __init__(self, graph_objectDetectBurn, labels):
        super(TFObjectDetection, self).__init__(labels)
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.import_graph_def(graph_objectDetectBurn, name='')
            
    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float)[:,:,(2,1,0)] # RGB -> BGR

        #with tf.Session(graph=self.graph) as sess:
        output_tensor = sess2.graph.get_tensor_by_name('model_outputs:0')
        outputs = sess2.run(output_tensor, {'Placeholder:0': inputs[np.newaxis,...]})
        return outputs[0]

# Load a TensorFlow model
graph_objectDetectBurn = tf.GraphDef()
with tf.gfile.FastGFile('modelDetectBurn.pb', 'rb') as f:
    graph_objectDetectBurn.ParseFromString(f.read())

# Load labels
with open('labelsDetectBurn.txt', 'r') as f:
    labels = [l.strip() for l in f.readlines()]

od_modelBurn = TFObjectDetection(graph_objectDetectBurn, labels)






#--------------發霉物件偵測----------------------------------

class TFObjectDetectionMold(object_detectionMold):
    """Object Detection class for TensorFlow
    """

    def __init__(self, graph_objectDetectMold, labels):
        super(TFObjectDetectionMold, self).__init__(labels)
        self.graph = tf.Graph()
        with self.graph.as_default():
            input_data = tf.placeholder(tf.float32, [1, None, None, 3], name='Placeholder')
            tf.import_graph_def(graph_objectDetectMold, input_map={"Placeholder:0": input_data}, name="")
            #tf.import_graph_def(TFObjectDetectionMold, name='')

    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float)[:, :, (2, 1, 0)]  # RGB -> BGR

        #with tf.Session(graph=self.graph) as sess:
        output_tensor = sess3.graph.get_tensor_by_name('model_outputs:0')
        outputs = sess3.run(output_tensor, {'Placeholder:0': inputs[np.newaxis, ...]})
        return outputs[0]

# Load a TensorFlow model
graph_objectDetectMold = tf.GraphDef()
with tf.gfile.FastGFile('model_mold.pb', 'rb') as f:
    graph_objectDetectMold.ParseFromString(f.read())

# Load labels
with open('labels_mold.txt', 'r') as f:
    labels = [l.strip() for l in f.readlines()]

od_modelMold = TFObjectDetectionMold(graph_objectDetectMold, labels)








# Remove some warnings from output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



#--------------------分類的模型--------------------

graph_classification = tf.GraphDef()
labels = []
# Import the TF graph
with tf.gfile.FastGFile('modelClassification.pb', 'rb') as f:
    graph_classification.ParseFromString(f.read())
    tf.import_graph_def(graph_classification, name='')

# Create a list of labels
with open('labelsClassification.txt', 'rt') as lf:
    for l in lf:
        labels.append(l.strip())





#----------------------------放置攝影機長寬等，關掉鏡頭後要儲存的影片設定------------------
cap = cv2.VideoCapture(0)
# 声明编码器和创建VideoWrite对象
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter('output.avi',fourcc, 40.0, (width, height))


# checks whether frames were extracted 
success = 1


# These names are part of the model and cannot be changed.分類模型的設定
output_layer = 'loss:0'
input_node = 'Placeholder:0'


with tf.Session() as sess:
    prob_tensor = sess.graph.get_tensor_by_name(output_layer)


    graphObjectDetectBurn = tf.Graph()   
    graphObjectDetectMold = tf.Graph() 
    #麵包烤焦模型
    with graphObjectDetectBurn.as_default():
        tf.import_graph_def(graph_objectDetectBurn, name='') # graph_def2 loaded somewhere
    with tf.Session(graph=graphObjectDetectBurn) as sess2:
        #tf.import_graph_def(graph_def2, name='')
      
      
        
        #麵包發霉模型
        with graphObjectDetectMold.as_default():
            #input_data = tf.placeholder(tf.float32, [1, None, None, 3], name='Placeholder')
            #tf.import_graph_def(graph_objectDetectMold, name='')
            input_data = tf.placeholder(tf.float32, [1, None, None, 3], name='Placeholder')
            tf.import_graph_def(graph_objectDetectMold, input_map={"Placeholder:0": input_data}, name="")
        with tf.Session(graph=graphObjectDetectMold) as sess3:



            while success:        
                # vidObj object calls read 
                # function extract frames 
                success, image = cap.read()
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


                
                # Load the test images from a file

                #image = Image.open(image)
                #image = array(img).reshape(1, 227 * 227 * 3)


                # Update orientation based on EXIF tags, if the file has orientation info
                image = image_handling.update_orientation(image)

                # Convert to OpenCV (Open Source Computer Vision) format
                image = image_handling.convert_to_opencv(image)

                # If the image has either w or h greater than 1600 we resize it down respecting
                # aspect ratio such that the largest dimension is 1600
                image = image_handling.resize_down_to_1600_max_dim(image)



                # We next get the largest center square
                h, w = image.shape[:2]
                min_dim = min(w,h)
                max_square_image = image_handling.crop_center(image, min_dim, min_dim)

                # Resize that square down to 256x256
                augmented_image = image_handling.resize_to_256_square(max_square_image)

                # The compact models have a network size of 227x227, the model requires this size
                network_input_size = 224

                # Crop the center for the specified network_input_Size
                augmented_image = image_handling.crop_center(augmented_image, network_input_size, network_input_size)

                
                predictions = sess.run(prob_tensor, {input_node: [augmented_image] })

                # Print the highest probability label
                highest_probability_index = np.argmax(predictions)
                print('Classified as: ' + labels[highest_probability_index])

                #有增加label字的frame


                #分類的機率
                # And print out each of the results mapping labels with their probabilities
                label_index = 0
                for p in predictions[0]:
                    truncated_probablity = np.float64(np.round(p,8))
                    print (labels[label_index] + str(truncated_probablity))
                    label_index += 1

                    

                #若是波蘿機率大於蔥麵包就執行波蘿模型的物件偵測 
                if(predictions[0][0]>predictions[0][1]):

                    #image = Image.open(image)
                    image = Image.fromarray(image) # npmarray to jpgfile 
                    #with tf.gfile.FastGFile('modelDetect.pb', 'rb') as f:
                        #graph_def.ParseFromString(f.read())
                    #tf.import_graph_def(graph_def2, name='')
                    predictions = od_modelBurn.predict_image(image)
                    for i in predictions:
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
                #蔥麵包機率大於波蘿麵包
                elif(predictions[0][0] < predictions[0][1]):
                    image = Image.fromarray(image)
                    predictions = od_modelMold.predict_image(image)
                    for i in predictions:
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
                        print("b")


                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

           

print("width:",width, "height:", height)
# 釋放攝像頭資源
cap.release()
out.release()
cv2.destroyAllWindows() 



