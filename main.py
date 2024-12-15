from ultralytics import YOLO
import cv2 
from sort.sort import *
from utils import get_car, read_license_plate, write_csv

# Load models
coco_model = YOLO("yolov8n.pt")# This is for detect cars
license_plate_detector = YOLO("./license_plate_detector.pt")

# Results
results = {}

# Load the video
#cap = cv2.VideoCapture("./videonico.mp4")
cap = cv2.VideoCapture("./sample.mp4")
#cap = cv2.VideoCapture("./videonegro.mp4")

# Read frames
frame_nmr = -1
ret = True
vehicles = [2, 3, 5, 7]# id's objects of the coco model
tracker = Sort()

while ret:
	frame_nmr += 1
	ret, frame = cap.read()
	
	if ret:
		results[frame_nmr] = {}
  
		# Detect vehicles\
		detections = coco_model(frame)[0]
		registered_detections = []

		for detection in detections.boxes.data.tolist():
			x1, y1, x2, y2, score, class_id = detection

			if int(class_id) in vehicles:
				registered_detections.append([x1, y1, x2, y2, score])
	
	# Track vehicles
	track_ids = tracker.update(np.asarray(registered_detections))
	
	# Detect license plate
	license_plates = license_plate_detector(frame)[0]
	for license_plate in license_plates.boxes.data.tolist():
		x1, y1, x2, y2, score, class_id = license_plate

		# Assing license plate to car
		xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

		if car_id != -1:

			# Crop the license plate
			license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

			# Process license plate
			license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
			_, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
	
			#cv2.imshow("License plate", license_plate_crop)
			#cv2.imshow("License plate threshold", license_plate_crop_thresh)

			#cv2.waitKey(0)	

			# Read license plate number
			license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

			if license_plate_text is not None:
				results[frame_nmr][car_id] = {"car" : {"bbox" : [xcar1, ycar1, xcar2, ycar2]},
											"license_plate": {  "bbox": [x1, y1, x2, y2],
																"text": license_plate_text,
																"bbox_score": score,
																"text_score": license_plate_text_score}}

# Write results
write_csv(results, "./test.csv")