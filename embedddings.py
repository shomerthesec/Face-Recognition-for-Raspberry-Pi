 
import face_recognition
import os
from numpy import save
# Load the jpg files into numpy arrays

images=[ face_recognition.load_image_file(f'authorised/{img}') for img in os.listdir('authorised/') ]

# Get the face encodings for each face in each image file
# Since there could be more than one face in each image, it returns a list of encodings.
# But since I know each image only has one face, I only care about the first encoding in each image, so I grab index 0.
try:
    encodings = [ enc  for img in images for enc in face_recognition.face_encodings(img) ]

except IndexError:
    print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
    quit()
    
save('embeddings.npy',encodings)
