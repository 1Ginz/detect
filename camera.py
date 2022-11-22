# import time
#
# from PIL import ImageTk, Image
#
# import numpy as np
# import cv2
#
# import numpy
# from tensorflow.keras.models import load_model
# from tensorflow.keras.utils import Sequence
# from keras.preprocessing import image
#
# #############################################
# #physical_devices = tf.config.list_physical_devices("GPU")
# #tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[0], True)
#
# threshold = 0.75 #THRESHOLD của Xác Suất
# font = cv2.FONT_HERSHEY_SIMPLEX
# ##############################################
#
# # SETUP CAMERA
# cap = cv2.VideoCapture(0)
# cap.set(3, 640) # Chiều rộng cửa sổ
# cap.set(4, 480) # Chiều dài cửa sổ
# cap.set(10, 180) # Độ sáng
# # IMPORT TRAINED MODEL
# # model = load_model('model.h5')
# model = load_model('traffic_classifier.h5')
#
# def preprocessing(img):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img = cv2.equalizeHist(img)
#     img = img / 255
#     return img
#
# def getCalssName(classNo):
#     if classNo == 0:
#         return 'Tốc độ tối đa 20 km/h'
#     elif classNo == 1:
#         return 'Tốc độ tối đa 30 km/h'
#     elif classNo == 2:
#         return 'Tốc độ tối đa 50 km/h'
#     elif classNo == 3:
#         return 'Tốc độ tối đa 60 km/h'
#     elif classNo == 4:
#         return 'Tốc độ tối đa 70 km/h'
#     elif classNo == 5:
#         return 'Tốc độ tối đa 80 km/h'
#     elif classNo == 6:
#         return 'End of Speed Limit 80 km/h'
#     elif classNo == 7:
#         return 'Tốc độ tối đa 100 km/h'
#     elif classNo == 8:
#         return 'Tốc độ tối đa 120 km/h'
#     elif classNo == 9:
#         return 'Không vượt'
#     elif classNo == 10:
#         return 'No passing for vechiles over 3.5 metric tons'
#     elif classNo == 11:
#         return 'Right-of-way at the next intersection'
#     elif classNo == 12:
#         return 'Priority road'
#     elif classNo == 13:
#         return 'Đường ưu tiên'
#     elif classNo == 14:
#         return 'Dừng lại'
#     elif classNo == 15:
#         return 'No vechiles'
#     elif classNo == 16:
#         return 'Vechiles over 3.5 metric tons prohibited'
#     elif classNo == 17:
#         return 'Không quay đầu'
#     elif classNo == 18:
#         return 'Cẩn thật'
#     elif classNo == 19:
#         return 'Chỗ ngoặt nguy hiểm vòng bên trái'
#     elif classNo == 20:
#         return 'Chỗ ngoặt nguy hiểm vòng bên phải'
#     elif classNo == 21:
#         return 'Double curve'
#     elif classNo == 22:
#         return 'Đường gập ghềnh'
#     elif classNo == 23:
#         return 'Đường trơn trượt'
#     elif classNo == 24:
#         return 'Road narrows on the right'
#     elif classNo == 25:
#         return 'Đường đang thi công'
#     elif classNo == 26:
#         return 'Traffic signals'
#     elif classNo == 27:
#         return 'Pedestrians'
#     elif classNo == 28:
#         return 'Children crossing'
#     elif classNo == 29:
#         return 'Bicycles crossing'
#     elif classNo == 30:
#         return 'Beware of ice/snow'
#     elif classNo == 31:
#         return 'Wild animals crossing'
#     elif classNo == 32:
#         return 'End of all speed and passing limits'
#     elif classNo == 33:
#         return 'Turn right ahead'
#     elif classNo == 34:
#         return 'Turn left ahead'
#     elif classNo == 35:
#         return 'Ahead only'
#     elif classNo == 36:
#         return 'Go straight or right'
#     elif classNo == 37:
#         return 'Go straight or left'
#     elif classNo == 38:
#         return 'Keep right'
#     elif classNo == 39:
#         return 'Keep left'
#     elif classNo == 40:
#         return 'Roundabout mandatory'
#     elif classNo == 41:
#         return 'End of no passing'
#     elif classNo == 42:
#         return 'End of no passing by vechiles over 3.5 metric tons'
#
# classes = {1: 'Tốc độ tối đa (20km/h)',
#            2: 'Tốc độ tối đa (30km/h)',
#            3: 'Tốc độ tối đa (50km/h)',
#            4: 'Tốc độ tối đa (60km/h)',
#            5: 'Tốc độ tối đa (70km/h)',
#            6: 'Tốc độ tối đa (80km/h)',
#            7: 'End of speed limit (80km/h)',
#            8: 'Tốc độ tối đa (100km/h)',
#            9: 'Tốc độ tối đa (120km/h)',
#            10: 'Không được vượt',
#            11: 'No passing veh over 3.5 tons',
#            12: 'Right-of-way at intersection',
#            13: 'Đường ưu tiên',
#            14: 'Nhường đường',
#            15: 'Dừng lại',
#            16: 'No vehicles',
#            17: 'Veh > 3.5 tons prohibited',
#            18: 'Không vào   ',
#            19: 'Cẩn thận',
#            20: 'Chỗ ngoặt nguy hiểm vòng bên trái',
#            21: 'Chỗ ngoặt nguy hiểm vòng bên phải',
#            22: 'Double curve',
#            23: 'Đường gập ghềnh',
#            24: 'Đường trơn trượt',
#            25: 'Road narrows on the right',
#            26: 'Đường đang thi công',
#            27: 'Biển báo giao thông',
#            28: 'Pedestrians',
#            29: 'Trẻ em qua đường',
#            30: 'Bicycles crossing',
#            31: 'Beware of ice/snow',
#            32: 'Wild animals crossing',
#            33: 'End speed + passing limits',
#            34: 'Rẽ phải phía trước',
#            35: 'Rẽ trái phía trước',
#            36: 'Đi thẳng',
#            37: 'Go straight or right',
#            38: 'Go straight or left',
#            39: 'Keep right',
#            40: 'Keep left',
#            41: 'Roundabout mandatory',
#            42: 'End of no passing',
#            43: 'End no passing veh > 3.5 tons'}
#
# def classify(image):
#     image = image.resize((30, 30))
#     image = numpy.expand_dims(image, axis=0)
#     image = numpy.array(image)
#     # pred = model.predict_classes([image])[0]
#     # sign = classes[pred + 1]
#     if(image.shape[0] == None):
#         return
#
#
#     predict_x = model.predict([image])
#     classes_x = numpy.argmax(predict_x, axis=1)
#     sign = classes[classes_x[0] + 1]
#     return sign
#
# def returnRedness(img):
# 	yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
# 	y,u,v=cv2.split(yuv)
# 	return v
#
# def threshold(img,T=150):
# 	_,img=cv2.threshold(img,T,255,cv2.THRESH_BINARY)
# 	return img
#
# def findContour(img):
# 	contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# 	return contours
#
# def findBiggestContour(contours):
# 	m = 0
# 	c = [cv2.contourArea(i) for i in contours]
# 	return contours[c.index(max(c))]
#
# def boundaryBox(img,contours):
# 	x,y,w,h=cv2.boundingRect(contours)
# 	img=cv2.rectangle(img,(x - 7,y-7),(x+w+7,y+h+7),(0,255,0),2)
# 	sign=img[y:(y+h) , x:(x+w)]
# 	return img,sign
#
# while True:
#     _, frame = cap.read()
#     redness = returnRedness(frame)  # step 1 --> specify the redness of the image
#     thresh = threshold(redness)
#     imgOrignal = frame
#     try:
#         contours = findContour(thresh)
#         big = findBiggestContour(contours)
#         if cv2.contourArea(big) > 3000:
#             img, sign = boundaryBox(frame, big)
#             cv2.imshow('frame', img)
#             cv2.imshow('sign', sign)
#             imgOrignal = sign
#         else:
#             pass
#
#     except:
#         pass
#
#     img = Image.fromarray(imgOrignal, 'RGB')
#     detected = classify(img)
#     print("Detected value: " + str(detected))
#     # Xử lý ảnh
#     # img = np.asarray(imgOrignal)
#     # img = cv2.resize(img, (32, 32))
#     # # img = preprocessing(img)
#     # cv2.imshow("Processed Image", img)
#     # img = img.reshape(1, 32, 32, 1)
#     # cv2.putText(imgOrignal, "Biển: ", (20, 35), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
#     # cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
#
#     # Tiến hành dự đoán kết quả
#     # predictions = model.predict(img)
#     # classIndex = np.argmax(predictions, axis=1)
#     # probabilityValue = np.amax(predictions)
#     # if probabilityValue > threshold:
#     #     print(getCalssName(classIndex))
#     #     cv2.putText(imgOrignal, str(classIndex) + " " + str(getCalssName(classIndex)), (120, 35),
#     #             font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
#     #     cv2.putText(imgOrignal, str(round(probabilityValue * 100, 2)) + " %", (180, 75),
#     #             font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
#     # cv2.imshow("Demo", imgOrignal)
#
#     if cv2.waitKey(1) and 0xFF == ord('q'):
#         break