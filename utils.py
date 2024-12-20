import string
import easyocr
import psycopg2

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu = True)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
					'I': '1',
					'J': '3',
					'A': '4',
					'G': '6',
					'S': '5'}

dict_int_to_char = {'0': 'O',
					'1': 'I',
					'3': 'J',
					'4': 'A',
					'6': 'G',
					'5': 'S'}

def write_csv(results, output_path):
	"""
	Write the results to a CSV file.

	Args:
		results (dict): Dictionary containing the results.
		output_path (str): Path to the output CSV file.
	"""
	with open(output_path, 'w') as f:
		f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
												'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
												'license_number_score'))

		for frame_nmr in results.keys():
			for car_id in results[frame_nmr].keys():
				if 'car' in results[frame_nmr][car_id].keys() and \
				   'license_plate' in results[frame_nmr][car_id].keys() and \
				   'text' in results[frame_nmr][car_id]['license_plate'].keys():
					f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
															car_id,
															'[{} {} {} {}]'.format(
																results[frame_nmr][car_id]['car']['bbox'][0],
																results[frame_nmr][car_id]['car']['bbox'][1],
																results[frame_nmr][car_id]['car']['bbox'][2],
																results[frame_nmr][car_id]['car']['bbox'][3]),
															'[{} {} {} {}]'.format(
																results[frame_nmr][car_id]['license_plate']['bbox'][0],
																results[frame_nmr][car_id]['license_plate']['bbox'][1],
																results[frame_nmr][car_id]['license_plate']['bbox'][2],
																results[frame_nmr][car_id]['license_plate']['bbox'][3]),
															results[frame_nmr][car_id]['license_plate']['bbox_score'],
															results[frame_nmr][car_id]['license_plate']['text'],
															results[frame_nmr][car_id]['license_plate']['text_score'])
							)
		f.close()

def license_complies_format(text):
	"""
	Check if the license plate text complies with the required format.

	Args:
		text (str): License plate text.

	Returns:
		bool: True if the license plate complies with the format, False otherwise.
	"""
	if len(text) != 7:
		return False

	if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
	   (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
	   (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
	   (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
	   (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
	   (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
	   (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
		return True
	else:
		return False

def format_license(text):
	"""
	Format the license plate text by converting characters using the mapping dictionaries.

	Args:
		text (str): License plate text.

	Returns:
		str: Formatted license plate text.
	"""
	license_plate_ = ''
	mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
			   2: dict_char_to_int, 3: dict_char_to_int}
	for j in [0, 1, 2, 3, 4, 5, 6]:
		if text[j] in mapping[j].keys():
			license_plate_ += mapping[j][text[j]]
		else:
			license_plate_ += text[j]

	return license_plate_

def read_license_plate(license_plate_crop):
	"""
	Read the license plate text from the given cropped image
	
	Args:
		Licesne_plate_crop (PIL.Image.Image): Cropped iamge containing the license plate.
		
	Returns:
		tuple: Tuple containing the formatted license plate text and it's confidence score.
	
	
	"""
	detections = reader.readtext(license_plate_crop)

	for detection in detections:	
		bbox, text, score = detection
  
		text = text.upper().replace(' ', '')

		if license_complies_format(text):
			return text, score
			
   			#return format_license(text), score
	
	return None, None

def get_car(license_plate, vehicle_track_ids):
	"""
	Retrieve the vechicle coordinates and ID based on the license plate coordinates
	
	Args:
		license_plate(tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id)
		vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.
		
	Returns:
		tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
	"""
	x1, y1, x2, y2, score, class_id = license_plate 

	foundIt = False

	for i in range(len(vehicle_track_ids)):
		xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[i]
  
		if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
			car_indx = i
			foundIt = True
			break
	
	if foundIt:
		return vehicle_track_ids[car_indx]
	
	return -1, -1, -1, -1, -1 

def connectDB():
	""" Hace la conexion con la BD """
	conexion = psycopg2.connect(
	host="localhost",
	database="nombre_base_datos", #Cambiar dependiendo de la BD
	user="root",
	password="contraseña"
)

def sendQuery(valor_bool, conexion):
		""" Envia una consulta a la BD """
		cursor = conexion.cursor()
		
		# Crear la consulta
		if valor_bool:
			query = "UPDATE escaneo SET resultado = 1;" #Cambiar dependiendo de lo que se quiera hacer lo deje segun el diagrama
		else:
			query = "UPDATE escaneo SET resultado = 0;"
		
		# Ejecutar la consulta
		cursor.execute(query)
		
		# Cerrar el cursor y la conexión
		cursor.close()

def isInMap(license_plate, license_map, conexion=None):
    if license_plate in license_map:
        return True
    else:
        # ejecuta una consulta a la base de datos para registrar esa patente (opcional igual XD)
		# sino se deja el false nomas
        cursor = conexion.cursor()
        query = f"INSERT INTO escaneo (patente, resultado) VALUES ('{license_plate}', 0);" 
        cursor.execute(query)
        conexion.commit()
        cursor.close()
        return False

license_list = ["ABC1234", "XYZ5678", "LMN9102"]
print(isInMap("ABC1234", license_list)) #deberia dar true
