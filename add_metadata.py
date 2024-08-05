import os
import pandas as pd
import numpy as np
import cv2
import torch
from tqdm import tqdm
from load_imgquality import load_imgquality
from load_imgeyeside import load_imgeyeside

# 1. Locate all ODinfo files in the folder progplots (relative dir)
def get_odinfo_files(directory='./progplots/'):
    odinfo_files = []
    for path, _, files in os.walk(directory):
        for name in files:
            if 'ODinfo' in name:
                odinfo_files.append(os.path.join(path, name).replace("\\", "/"))
    return odinfo_files

# 2. Define add_DiscHu_exclude
def add_DiscHu_exclude(odinfo):
    odinfo['DiscHu_exclude'] = odinfo['DiscHu'].apply(lambda hu: 1 if abs(hu - 0.159) > 0.01 else 0)
    return odinfo

# 3. Define add_DiscDetect_exclude
def add_DiscDetect_exclude(odinfo):
    odinfo['DiscDetect_exclude'] = odinfo['Disc_X'].apply(lambda discx: 1 if discx == 0 else 0)
    return odinfo

# 4. Define add_Shape_exclude
def add_Shape_exclude(odinfo):
    odinfo['Shape_exclude'] = odinfo['Shape'].apply(lambda shape: 1 if shape != '(1444, 1444)' else 0)
    return odinfo

# 5. Image preprocessing procedure
def img_preproc(impath):
    image = cv2.imread(impath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512)) / 255.
    return image

# 6. Define add_quality_eyeside
def add_quality_eyeside(odinfo, model_imgquality, model_imgeyeside):
    images = []
    
    for img_name in odinfo.ImgName:
        patient_id = img_name.split('/')[-1].split('_')[0]
        img_path = f'D:/Glaucoma/Data/GRAPE/progplots/{patient_id}/30crop/{img_name.split("/")[-1]}'
        images.append(img_preproc(img_path))
    
    images = np.array(images)
    if len(images.shape) == 3:
        images = np.expand_dims(images, axis=0)

    images = np.transpose(images, (0, 3, 1, 2))
    images = torch.from_numpy(images).to('cuda', non_blocking=True).float()

    imgquality = model_imgquality(images)
    imgeyeside = model_imgeyeside(images)

    odinfo['quality'] = np.squeeze(imgquality.cpu().detach().numpy())
    odinfo['eyeside'] = np.squeeze(imgeyeside.cpu().detach().numpy())
    return odinfo

# 7. Define add_eye
def get_eye(odinfo_filename):
    return odinfo_filename.split('_')[0][-2:]

def get_patient(odinfo_filename):
    return int(odinfo_filename.split('/')[2])

# 8. Define add_eyeside_exclude
def add_eyeside_exclude(odinfo, eye):
    odinfo['eyeside_exclude'] = odinfo['eyeside'].apply(lambda es: 0 if (es < 0 and eye == 'OD') or (es > 0 and eye == 'OS') else 1)
    return odinfo

# 9. Define add_age
def add_age(odinfo, patient_id, fundusinfo):
    age = fundusinfo[fundusinfo['Subject Number'] == patient_id]['Age'].values[0]
    odinfo['Age'] = age
    return odinfo

# 10. Define add_md
def add_md(odinfo, patient_id, eye, visitinfo):
    md_values = visitinfo[(visitinfo['Subject Number'] == patient_id) & (visitinfo['Laterality'] == eye) & (visitinfo['Corresponding CFP'] != '/')]['MD'].values
    odinfo['MD'] = pd.Series(md_values)
    return odinfo

# 11. Define add_timebetwvis
def add_timebetwvis(odinfo):
    odinfo['timebetwvis'] = odinfo['Examdate'].diff()
    return odinfo

# 12. Define add_sex
def add_sex(odinfo, patient_id, fundusinfo):
    sex = fundusinfo[fundusinfo['Subject Number'] == patient_id]['Gender'].values[0]
    odinfo['Sex'] = sex
    return odinfo

def main():
    # Load models in eval mode
    model_imgquality = load_imgquality()
    model_imgeyeside = load_imgeyeside()

    # Load fundus metadata
    fundusinfo = pd.read_excel(r"D:\Glaucoma\Data\GRAPE\VF and clinical information.xlsx", sheet_name=0)
    visitinfo = pd.read_excel(r"D:\Glaucoma\Data\GRAPE\VF and clinical information.xlsx", sheet_name=1)

    odinfo_files = get_odinfo_files()

    for odinfo_file in tqdm(odinfo_files):
        # Read the ODinfo file
        odinfo = pd.read_csv(odinfo_file)

        # Filter out rows with Disc_X == 0
        odinfo = odinfo[odinfo['Disc_X'] != 0]
        if len(odinfo) == 0:
            continue

        # Get the eye / laterality from the filename
        eye = get_eye(odinfo_file)
        patient_id = get_patient(odinfo_file)

        # Add the exclude columns
        odinfo = add_DiscHu_exclude(odinfo)
        odinfo = add_DiscDetect_exclude(odinfo)
        odinfo = add_Shape_exclude(odinfo)

        # Add quality and eyeside using the models
        odinfo = add_quality_eyeside(odinfo, model_imgquality, model_imgeyeside)

        # Add eyeside_exclude
        odinfo = add_eyeside_exclude(odinfo, eye)

        # Add age, MD, time between visits, and sex
        odinfo = add_age(odinfo, patient_id, fundusinfo)
        odinfo = add_md(odinfo, patient_id, eye, visitinfo)
        odinfo = add_timebetwvis(odinfo)
        odinfo = add_sex(odinfo, patient_id, fundusinfo)

        # Save the updated ODinfo file
        odinfo.to_csv(odinfo_file, index=False)

if __name__ == '__main__':
    main()
