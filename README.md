# Pedestrians-Detecting-Tracking-and-Analysis-using-Yolov5-StrongSORT

## File explanation:  
**Group_Notebook.ipynb** : You will be running the program in this file followed by the inside instructions
                                         Citations are at the bottom of that file.  
**object_tracking.py** : Contains functions for task 1 and flags which allow you to observe specific outcome for different tasks
**jpg_to_mp4.py** : Contains functions to transfer the jpg images into mp4 video
**video_crop.py** : Contains functions for task 2.3 and 2.4
**analyses.py**: Contains functions for task 3

## Program guide:
The specific guidline is in the jupyter notbook, please refer that.  
In order to minimize the submission package size. We do not contain the output images,  
Please run specific cell inside the jupyter notebook to obtain the results.  
Input images are from https://motchallenge.net/data/STEP-ICCV21/  
please download images from it and add step_images folder and all the content inside the folder into the root directory as input test data sets.  
The output directory can be created automatically when running specific cell inside jupyter notebook.  
Task 2.3 and 2.4 can be accessed only when running view function inside object_tracking.py. The notebook have the option to obtain this outcome.  

## Citation of the open-source code we used  
**yolov5-strongsort-osnet-2022:**  
Mikel Broström, "Real-time multi-camera multi-object tracker using YOLOv5 and StrongSORT with OSNet", 2022.  
Available: https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet  

**Video cropping**  
Adrian Rosebrock 2015, pyimagesearch, accessed 21 July 2022.  
Available: <https://pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/>   
