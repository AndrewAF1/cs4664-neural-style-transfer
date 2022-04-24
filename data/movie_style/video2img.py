import cv2
from random import random
  
# Function to extract frames
def video_to_img(path, domain):
      
    # Path to video file
    vidObj = cv2.VideoCapture(path)
  
    # Used as counter variable
    train_count = 0
    test_count = 0
  
    # checks whether frames were extracted
    success = 1
  
    while success:
  
        # vidObj object calls read
        # function extract frames
        success, image = vidObj.read()

        if random() < 1e-2:
  
            if random() < 0.2:
                folder = "test"
                test_count += 1
                count = test_count
            else:
                folder = "train"
                train_count += 1
                count = train_count
            
            path = "data/movie_style/{}{}/{}{}_frame{}.jpg".format(
                folder, domain, folder, domain, count
            )

            print(path)
            # Saves the frames with frame-count
            cv2.imwrite(path, image)
  

    print("frames:", train_count, test_count)

#VideoToImg("data/movie_style/videos/sw_ep4.mp4", "A")
video_to_img("data/movie_style/videos/sw_ep7.mp4", "B")