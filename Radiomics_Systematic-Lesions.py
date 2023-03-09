## Radiomics_Systematic_Lesions.py
## Katie Merriman
## Written 2/27/23
## Takes in csv with patient MRNs and filepath of folder containing all NIfTIs and masks
## Can switch between filepaths for running locally or remotely by changing variable "local" at beginning of class definition
## Divides prostate into 12 sections following transrectal systematic biopsy regions
## Records number and names of all manually contoured lesions present in each section
## Records highest PI-RADS score present in each section (0 = none, 6 = whole prostate lesion)
## Collects ALL available PyRadiomics features for:
## 1. Manually contoured whole prostate
## 2. Manually contoured lesions
## 3. Systematically divided prostate sections
## Creates then appends .csv file with radiomics data after every patient

import pandas as pd
import os
import os.path
from os import path
import csv
import SimpleITK as sitk
import pandas as pd
from skimage import draw
import numpy as np
import pydicom
import math
import nibabel
import re
import dicom2nifti
import shutil
import glob
import radiomics
from radiomics import featureextractor, imageoperations, firstorder, glcm, glszm, ngtdm
import six
import logging
np.set_printoptions(threshold=np.inf)

class featureCalculator():
    def __init__(self):
        local = 0

        if local:
            self.csv_file = r'T:\MIP\Katie_Merriman\RadiomicsProject\Patients4Radiomics_test.csv'
            self.patientFolder = r'T:\MIP\Katie_Merriman\Project1Data\PatientNormalized_data'
            self.saveFolder = r'T:\MIP\Katie_Merriman\RadiomicsProject'
            self.fileName = r'T:\MIP\Katie_Merriman\RadiomicsProject\RadTest.csv'

        else:
        ### lambda desktop directory mapping
            self.csv_file = 'Mdrive_mount/MIP/Katie_Merriman/RadiomicsProject/Patients4Radiomics_test.csv'
            self.patientFolder = 'Mdrive_mount/MIP/Katie_Merriman/Project1Data/PatientNormalized_data'
            self.saveFolder = 'Mdrive_mount/MIP/Katie_Merriman/RadiomicsProject'
            self.fileName = 'Mdrive_mount/MIP/Katie_Merriman/RadiomicsProject/RadTest.csv'


        #self.patient_data = []
        #self.lesion_data = []

        self.PIRADSnames = ['PIRADS', 'pirads', 'PZ', 'TZ']
        self.PIRADS5names = ['PIRADS_5', 'pirads_5', 'PZ_5', 'TZ_5']
        self.PIRADS4names = ['PIRADS_4', 'pirads_4', 'PZ_4', 'TZ_4']
        self.PIRADS3names = ['PIRADS_3', 'pirads_3', 'PZ_3', 'TZ_3']
        self.wpLesion = '1_PIRADS_1_bt.nii'


    def calculate(self):
        #errors = []
        #voi_list = []

        """
        ## INITIALIZE NEW CSV FILE ##
        # Note: Single row of info needs to be in double brackets or each character will get its own cell
        headers = [['MRN', '']]
        # open csv file in 'a+' mode to append
        file = open(r'T:\MIP\Katie_Merriman\RadiomicsProject\Systematic_and_Lesion_Radiomics.csv', 'a+', newline='')
        # writing the data into the file
        with file:
            write = csv.writer(file)
            write.writerows(headers)
        """

        # make list of patients, path to files
        df_csv = pd.read_csv(self.csv_file, sep=',', header=0)
        patient = []
        for rows, file_i in df_csv.iterrows():
            p = (str(file_i['MRN_date']))
            p_path = os.path.join(self.patientFolder, p)
            patient.append([p, p_path])

        ### FOR EACH PATIENT: ###
        for i in range(0, len(patient)):
            print("MRN: ", patient[i][0])
            prost_head = []
            prost_data = []
            voi_list = []
            seg_head = []
            seg_data = []
            lesion_head = []
            lesion_data = []
            radiomics_data = []
            
            ### CALCULATE RADIOMICS FEATURES ###
            try:
            ## calculate whole prostate features
                [prost_head, prost_data] = self.calculateProst(patient[i])

                ## identify VOIs with PIRADS
                voi_list = self.getVOIlist(patient[i])

                ## calculate segment features
                [seg_head, seg_data] = self.calculateSegment(patient[i], voi_list, prost_data)

                ## calculate lesion features
                [lesion_head, lesion_data] = self.calculateLesion(patient[i], voi_list, prost_data[0])

                ### UPDATE CSV WITH PATIENT RADIOMICS INFO ###
                radiomics_data = [[patient[i][0]] + prost_data + seg_data + lesion_data]
                # open csv file in 'a+' mode to append

                ##########IF THE FILE DOESN'T EXIST, COMBINE 1st LINE OF EACH 
                #when running first patient, add header info to spreadsheet:

                file = open(self.fileName, 'a+', newline='')
                # writing the data into the file
                with file:
                    write = csv.writer(file)
                    if i==0: #for first patient only
                        headers = ['MRN']
                        headers = [headers + prost_head + seg_head + lesion_head]
                        #for head in range(len(prost_head)):
                        #    headers = headers + [[prost_head[head]]]
                        #for head in range(len(seg_head)):
                        #    headers = headers + [[seg_head[head]]]
                        #for head in range(len(lesion_head)):
                        #    headers = headers + [[lesion_head[head]]]
                        #headers = prost_head + seg_head + lesion_head
                        write.writerows(headers)
                    write.writerows(radiomics_data)
                file.close()
            except FileNotFoundError:
                file = open(self.fileName, 'a+', newline='')
                # writing the data into the file
                with file:
                    write = csv.writer(file)
                    write.writerows([[patient[i][0],'Error: File not found']])



    def calculateProst(self,patient):
        wpMask = os.path.join(patient[1], 'wp_bt.nii.gz')  # path join patient path + mrn_with_date + wp_bt.nii.gz
        [headers, prost_data] = self.calculateRadiomics(patient, "wp", wpMask)
        return [headers, prost_data]

    def getVOIlist(self, patient):
        voi_list = []
        for root, dirs, files in os.walk(patient[1]):
            for name in files:
                if name.endswith('bt.nii.gz'):
                    if not name.endswith('wp_bt.nii.gz'):
                        voiPath = os.path.join(root, name)
                        PIRADS = -1
                        if any([substring in name for substring in self.PIRADS5names]):
                            PIRADS = 5
                        elif any([substring in name for substring in self.PIRADS4names]):
                            PIRADS = 4
                        elif any([substring in name for substring in self.PIRADS3names]):
                            PIRADS = 3
                        elif any([substring in name for substring in self.wpLesion]):
                            PIRADS = 6
                        else:
                            PIRADS = 2
                        voi_list.append([voiPath, name, PIRADS])
        return voi_list

    def calculateSegment(self, patient, voi_list, prost_data):

        prost = prost_data[6:] #ignores shape data


        wpMask = os.path.join(patient[1], 'wp_bt.nii.gz') # path join patient path + mrn_with_date + wp_bt.nii.gz
        mask_img = sitk.ReadImage(wpMask)
        #mask_img = sitk.ReadImage(r'T:\MIP\Katie_Merriman\Project1Data\PatientNormalized_data\0497587_20131125\wp_bt.nii.gz')
        mask_arr = sitk.GetArrayFromImage(mask_img)
        mskNZ = mask_arr.nonzero()
        arr_size = mask_img.GetSize()
        sizeX = arr_size[0]
        sizeY = arr_size[1]
        sizeZ = arr_size[2]


        ### CREATE BASE/MID/APEX MASKS ###
        ## Find axial boundaries  of each section by dividing slices containing prostate by 3
        lowerZ = int(min(mskNZ[0]) + round((max(mskNZ[0]) - min(mskNZ[0])) / 3))
        UpperZ = int(max(mskNZ[0]) - round(max(mskNZ[0]) - min(mskNZ[0])) / 3)

        ## Create blank masks and arrays for base (upper), mid, and apex (lower)
        LowerMask = sitk.Image(sizeX, sizeY, sizeZ, sitk.sitkInt8)
        LowerArr = sitk.GetArrayFromImage(LowerMask)
        MidMask = sitk.Image(sizeX, sizeY, sizeZ, sitk.sitkInt8)
        MidArr = sitk.GetArrayFromImage(MidMask)
        UpperMask = sitk.Image(sizeX, sizeY, sizeZ, sitk.sitkInt8)
        UpperArr = sitk.GetArrayFromImage(UpperMask)

        # populate each blank array with 1s where original mask is 1 within each axial section boundary
        for index in range(len(mskNZ[0])):
            if mskNZ[0][index] < lowerZ:
                LowerArr[mskNZ[0][index], mskNZ[1][index], mskNZ[2][index]] = 1
            elif mskNZ[0][index] < UpperZ:
                MidArr[mskNZ[0][index], mskNZ[1][index], mskNZ[2][index]] = 1
            else:
                UpperArr[mskNZ[0][index], mskNZ[1][index], mskNZ[2][index]] = 1


        ## Write masks to file to check function
        LowerMask = sitk.GetImageFromArray(LowerArr)
        LowerMask.CopyInformation(mask_img)
        sitk.WriteImage(LowerMask, os.path.join(self.saveFolder,'lower.nii.gz'))

        MidMask = sitk.GetImageFromArray(MidArr)
        MidMask.CopyInformation(mask_img)
        sitk.WriteImage(MidMask, os.path.join(self.saveFolder,'mid.nii.gz'))

        UpperMask = sitk.GetImageFromArray(UpperArr)
        UpperMask.CopyInformation(mask_img)
        sitk.WriteImage(UpperMask, os.path.join(self.saveFolder,'upper.nii.gz'))






        ## Divide each section into anterior/posterior and left/right quadrants

        ## Apex
        # Find x and y axis to define quadrants
        # (note: allows for differences in coronal/saggital center for apex vs mid vs base due to angle of prostate)
        LowerArrNZ = LowerArr.nonzero()
        LowCenterX = int(min(LowerArrNZ[1] + round((max(LowerArrNZ[1]) - min(LowerArrNZ[1])) / 2)))
        LowCenterY = int(min(LowerArrNZ[2] + round((max(LowerArrNZ[2]) - min(LowerArrNZ[2])) / 2)))

        # Create blank masks/arrays
        LowAntRMask = sitk.Image(sizeX, sizeY, sizeZ, sitk.sitkInt8)
        LowAntRArr = sitk.GetArrayFromImage(LowAntRMask)
        LowAntLMask = sitk.Image(sizeX, sizeY, sizeZ, sitk.sitkInt8)
        LowAntLArr = sitk.GetArrayFromImage(LowAntLMask)
        LowPostRMask = sitk.Image(sizeX, sizeY, sizeZ, sitk.sitkInt8)
        LowPostRArr = sitk.GetArrayFromImage(LowPostRMask)
        LowPostLMask = sitk.Image(sizeX, sizeY, sizeZ, sitk.sitkInt8)
        LowPostLArr = sitk.GetArrayFromImage(LowPostLMask)

        # fill blank arrays with 1s corresponding to mask values in desired quadrant
        for index in range(len(LowerArrNZ[2])):
            if LowerArrNZ[1][index] < LowCenterX: # if anterior
                if LowerArrNZ[2][index] < LowCenterY: # if right anterior
                    LowAntRArr[LowerArrNZ[0][index], LowerArrNZ[1][index], LowerArrNZ[2][index]] = 1
                else: #else left anterior
                    LowAntLArr[LowerArrNZ[0][index], LowerArrNZ[1][index], LowerArrNZ[2][index]] = 1
            elif LowerArrNZ[2][index] < LowCenterY: # else posterior, if right posterior
                LowPostRArr[LowerArrNZ[0][index], LowerArrNZ[1][index], LowerArrNZ[2][index]] = 1
            else: # else left posterior
                LowPostLArr[LowerArrNZ[0][index], LowerArrNZ[1][index], LowerArrNZ[2][index]] = 1

        ## Write each quadrant to file to check function
        LowAntRMask = sitk.GetImageFromArray(LowAntRArr)
        LowAntRMask.CopyInformation(mask_img)
        LowAntRpath = os.path.join(self.saveFolder, 'LowAntR.nii.gz')
        sitk.WriteImage(LowAntRMask, LowAntRpath)

        LowAntLMask = sitk.GetImageFromArray(LowAntLArr)
        LowAntLMask.CopyInformation(mask_img)
        LowAntLpath = os.path.join(self.saveFolder, 'LowAntL.nii.gz')
        sitk.WriteImage(LowAntLMask, LowAntLpath)

        LowPostRMask = sitk.GetImageFromArray(LowPostRArr)
        LowPostRMask.CopyInformation(mask_img)
        LowPostRpath = os.path.join(self.saveFolder, 'LowPostR.nii.gz')
        sitk.WriteImage(LowPostRMask, LowPostRpath)

        LowPostLMask = sitk.GetImageFromArray(LowPostLArr)
        LowPostLMask.CopyInformation(mask_img)
        LowPostLpath = os.path.join(self.saveFolder, 'LowPostL.nii.gz')
        sitk.WriteImage(LowPostLMask, LowPostLpath)

        ## Mid
        # Find x and y axis to define quadrants
        # (note: allows for differences in coronal/saggital center for apex vs mid vs base due to angle of prostate)
        MidArrNZ = MidArr.nonzero()
        MidCenterX = int(min(MidArrNZ[1] + round((max(MidArrNZ[1]) - min(MidArrNZ[1])) / 2)))
        MidCenterY = int(min(MidArrNZ[2] + round((max(MidArrNZ[2]) - min(MidArrNZ[2])) / 2)))

        # Create blank masks/arrays
        MidAntRMask = sitk.Image(sizeX, sizeY, sizeZ, sitk.sitkInt8)
        MidAntRArr = sitk.GetArrayFromImage(MidAntRMask)
        MidAntLMask = sitk.Image(sizeX, sizeY, sizeZ, sitk.sitkInt8)
        MidAntLArr = sitk.GetArrayFromImage(MidAntLMask)
        MidPostRMask = sitk.Image(sizeX, sizeY, sizeZ, sitk.sitkInt8)
        MidPostRArr = sitk.GetArrayFromImage(MidPostRMask)
        MidPostLMask = sitk.Image(sizeX, sizeY, sizeZ, sitk.sitkInt8)
        MidPostLArr = sitk.GetArrayFromImage(MidPostLMask)

        # fill blank arrays with 1s corresponding to mask values in desired quadrant
        for index in range(len(MidArrNZ[2])):
            if MidArrNZ[1][index] < MidCenterX:  # if anterior
                if MidArrNZ[2][index] < MidCenterY:  # if right anterior
                    MidAntRArr[MidArrNZ[0][index], MidArrNZ[1][index], MidArrNZ[2][index]] = 1
                else:  # else left anterior
                    MidAntLArr[MidArrNZ[0][index], MidArrNZ[1][index], MidArrNZ[2][index]] = 1
            elif MidArrNZ[2][index] < MidCenterY:  # else posterior, if right posterior
                MidPostRArr[MidArrNZ[0][index], MidArrNZ[1][index], MidArrNZ[2][index]] = 1
            else:  # else left posterior
                MidPostLArr[MidArrNZ[0][index], MidArrNZ[1][index], MidArrNZ[2][index]] = 1

        ## Write each quadrant to file to check function
        MidAntRMask = sitk.GetImageFromArray(MidAntRArr)
        MidAntRMask.CopyInformation(mask_img)
        MidAntRpath = os.path.join(self.saveFolder, 'MidAntR.nii.gz')
        sitk.WriteImage(MidAntRMask, MidAntRpath)

        MidAntLMask = sitk.GetImageFromArray(MidAntLArr)
        MidAntLMask.CopyInformation(mask_img)
        MidAntLpath = os.path.join(self.saveFolder, 'MidAntL.nii.gz')
        sitk.WriteImage(MidAntLMask, MidAntLpath)

        MidPostRMask = sitk.GetImageFromArray(MidPostRArr)
        MidPostRMask.CopyInformation(mask_img)
        MidPostRpath = os.path.join(self.saveFolder, 'MidPostR.nii.gz')
        sitk.WriteImage(MidPostRMask, MidPostRpath)

        MidPostLMask = sitk.GetImageFromArray(MidPostLArr)
        MidPostLMask.CopyInformation(mask_img)
        MidPostLpath = os.path.join(self.saveFolder, 'MidPostL.nii.gz')
        sitk.WriteImage(MidPostLMask, MidPostLpath)

        ## Base
        # Find x and y axis to define quadrants
        # (note: allows for differences in coronal/saggital center for apex vs mid vs base due to angle of prostate)
        UpperArrNZ = UpperArr.nonzero()
        UpCenterX = int(min(UpperArrNZ[1] + round((max(UpperArrNZ[1]) - min(UpperArrNZ[1])) / 2)))
        UpCenterY = int(min(UpperArrNZ[2] + round((max(UpperArrNZ[2]) - min(UpperArrNZ[2])) / 2)))

        # Create blank masks/arrays
        UpAntRMask = sitk.Image(sizeX, sizeY, sizeZ, sitk.sitkInt8)
        UpAntRArr = sitk.GetArrayFromImage(UpAntRMask)
        UpAntLMask = sitk.Image(sizeX, sizeY, sizeZ, sitk.sitkInt8)
        UpAntLArr = sitk.GetArrayFromImage(UpAntLMask)
        UpPostRMask = sitk.Image(sizeX, sizeY, sizeZ, sitk.sitkInt8)
        UpPostRArr = sitk.GetArrayFromImage(UpPostRMask)
        UpPostLMask = sitk.Image(sizeX, sizeY, sizeZ, sitk.sitkInt8)
        UpPostLArr = sitk.GetArrayFromImage(UpPostLMask)

        # fill blank arrays with 1s corresponding to mask values in desired quadrant
        for index in range(len(UpperArrNZ[2])):
            if UpperArrNZ[1][index] < UpCenterX:  # if anterior
                if UpperArrNZ[2][index] < UpCenterY:  # if right anterior
                    UpAntRArr[UpperArrNZ[0][index], UpperArrNZ[1][index], UpperArrNZ[2][index]] = 1
                else:  # else left anterior
                    UpAntLArr[UpperArrNZ[0][index], UpperArrNZ[1][index], UpperArrNZ[2][index]] = 1
            elif UpperArrNZ[2][index] < UpCenterY:  # else posterior, if right posterior
                UpPostRArr[UpperArrNZ[0][index], UpperArrNZ[1][index], UpperArrNZ[2][index]] = 1
            else:  # else left posterior
                UpPostLArr[UpperArrNZ[0][index], UpperArrNZ[1][index], UpperArrNZ[2][index]] = 1

        ## Write each quadrant to file to check function
        UpAntRMask = sitk.GetImageFromArray(UpAntRArr)
        UpAntRMask.CopyInformation(mask_img)
        UpAntRpath = os.path.join(self.saveFolder, 'UpAntR.nii.gz')
        sitk.WriteImage(UpAntRMask, UpAntRpath)

        UpAntLMask = sitk.GetImageFromArray(UpAntLArr)
        UpAntLMask.CopyInformation(mask_img)
        UpAntLpath = os.path.join(self.saveFolder, 'UpAntL.nii.gz')
        sitk.WriteImage(UpAntLMask, UpAntLpath)

        UpPostRMask = sitk.GetImageFromArray(UpPostRArr)
        UpPostRMask.CopyInformation(mask_img)
        UpPostRpath = os.path.join(self.saveFolder, 'UpPostR.nii.gz')
        sitk.WriteImage(UpPostRMask, UpPostRpath)

        UpPostLMask = sitk.GetImageFromArray(UpPostLArr)
        UpPostLMask.CopyInformation(mask_img)
        UpPostLpath = os.path.join(self.saveFolder, 'UpPostL.nii.gz')
        sitk.WriteImage(UpPostLMask, UpPostLpath)


        headers = []
        seg_data = []



        # for each segment, get radiomics data, diff between segment data and whole prostate data,
        # names of lesions overlapping segment, and max PIRADS score in segment
        [head, data] = self.SegRadiomics(patient, prost, voi_list, "ApexAntR", LowAntRArr, LowAntRpath)
        headers = headers + head
        seg_data.append(data)

        [head, data] = self.SegRadiomics(patient, prost, voi_list, "ApexAntL", LowAntLArr, LowAntLpath)
        headers = headers + head
        seg_data.append(data)

        [head, data] = self.SegRadiomics(patient, prost, voi_list, "ApexPostR", LowPostRArr, LowPostRpath)
        headers = headers + head
        seg_data.append(data)

        [head, data] = self.SegRadiomics(patient, prost, voi_list, "ApexPostL", LowPostLArr, LowPostLpath)
        headers = headers + head
        seg_data.append(data)

        [head, data] = self.SegRadiomics(patient, prost, voi_list, "MidAntR", MidAntRArr, MidAntRpath)
        headers = headers + head
        seg_data.append(data)

        [head, data] = self.SegRadiomics(patient, prost, voi_list, "MidAntL", MidAntLArr, MidAntLpath)
        headers = headers + head
        seg_data.append(data)

        [head, data] = self.SegRadiomics(patient, prost, voi_list, "MidPostR", MidPostRArr, MidPostRpath)
        headers = headers + head
        seg_data.append(data)

        [head, data] = self.SegRadiomics(patient, prost, voi_list, "MidPostL", MidPostLArr, MidPostLpath)
        headers = headers + head
        seg_data.append(data)

        [head, data] = self.SegRadiomics(patient, prost, voi_list, "BaseAntR", UpAntRArr, UpAntRpath)
        headers = headers + head
        seg_data.append(data)

        [head, data] = self.SegRadiomics(patient, prost, voi_list, "BaseAntL", UpAntLArr, UpAntLpath)
        headers = headers + head
        seg_data.append(data)

        [head, data] = self.SegRadiomics(patient, prost, voi_list, "BasePostR", UpPostRArr, UpPostRpath)
        headers = headers + head
        seg_data.append(data)

        [head, data] = self.SegRadiomics(patient, prost, voi_list, "BasePostL", UpPostLArr, UpPostLpath)
        headers = headers + head
        seg_data.append(data)


        ## calculate single section minimums, maximums, and difference from whole prostate data

        # create headers corresponding to min/max/diff data
        head_segInfo = []
        feat = int((len(head)-3)/2)
            # feat is num of features excluding lesion overlap features (3)
            # and diff-from-prostate features (half of remaining)
        for h in range(feat):
            # uses BasePostL headers, ignores shape headers and lesion overlap data,
            headers.append('minSingleSeg' + head[h][9:]) # removes 'BasePostL' from start of header feature name
            head_segInfo.append('Seg_with_min' + head[h][9:])
        for h in range(feat):
            headers.append('maxSingleSeg' + head[h][9:])
            head_segInfo.append('Seg_with_max' + head[h][9:])
        for h in range(feat):
            headers.append('maxDiffSingleSeg' + head[feat + h][9:])
                # diff-from-prost features start where main features end
            head_segInfo.append('Seg_with_maxDiff' + head[feat + h][9:])
        headers = headers + head_segInfo # put info about which seg had min/max at end of headers for seg data


        minSegList = []
        maxSegList = []
        diffSegList = []
        minSegInfo = []
        maxSegInfo = []
        diffSegInfo = []# save info about which seg had min/max at end of seg section
        for seg in range(12):
            for features in range(len(prost)): # prost does not contain any of the 6 shape features
                if seg == 0:
                    minSegList.append(seg_data[0][features+1]) #ignores name at beginning
                    # save radiomics from element [2] on (ignoring elements containing shape data)
                    # into every other spot of min list
                    minSegInfo.append(seg_data[0][0])
                    # save name of first segment (in element [1]) to right of each value
                    # repeat for max, and for difference between segment radiomics value and whole prostate value
                    maxSegList.append(seg_data[0][features+1])
                    maxSegInfo.append(seg_data[0][0])
                    diffSegList.append(seg_data[0][features+1] - prost[features])
                    diffSegInfo.append(seg_data[0][0])
                else:
                    if seg_data[seg][features + 1]<minSegList[features]:
                        minSegList[features] = seg_data[seg][features + 1]  # update list with new minimum
                        minSegInfo[features] = seg_data[seg][0]  # update segment associated with new minimum
                    if seg_data[seg][features + 1] > minSegList[features]:
                        maxSegList[features] = seg_data[seg][features + 1]  # update list with new maximum
                        maxSegInfo[features] = seg_data[seg][0]  # update segment associated with new maximum
                    if abs(seg_data[seg][features + 1]- prost[features]) > abs(diffSegList[features]):
                        # update with new difference if single section absolute difference is larger
                        diffSegList[features] = seg_data[seg][features + 1] - prost[features]
                        diffSegInfo[features] = seg_data[seg][0]


        segment_data = seg_data[0][1:]+ seg_data[1][1:] + seg_data[2][1:] + seg_data[3][1:] + seg_data[4][1:] + seg_data[5][1:] + seg_data[6][1:] \
                       + seg_data[7][1:] + seg_data[8][1:] + seg_data[9][1:] + seg_data[10][1:] + seg_data[11][1:] + minSegList + maxSegList \
                       + diffSegList + minSegInfo + maxSegInfo + diffSegInfo
        #for l in range(len(headers)):
        #    print(headers[l], segment_data[l])
        return [headers, segment_data]



    def calculateLesion(self, patient, voi_list, prost_vol):
        lesion_data = []
        for lesion in voi_list:
            voiMask = lesion[0]  # path to lesion nifti
            mask_img = sitk.ReadImage(voiMask)
            name = lesion[1][:-7] # removes ".nii.gz" from lesion name
            [header,data] = self.calculateRadiomics(patient, 'index', mask_img)
            lesion_data.append([name]+data)

        minLesionList = []
        minLesionInfo = []
        maxLesionList = []
        maxLesionInfo = []
        maxPIRADS = -1
        indexVol = -1
        index = -1
        P5vol = 0
        P4vol = 0
        P3vol = 0
        P6vol = 0
        P2vol = 0
        # determine mins/maxes
        for lesion in range(len(lesion_data)):
            for features in range(len(lesion_data[0]) - 1):
                if lesion == 0:
                    minLesionList.append(lesion_data[0][features + 1])
                    # save radiomics from element [1] on (ignoring elements containing name)
                    # into every other spot of min list
                    minLesionInfo.append(lesion_data[0][0])
                    # save name of first segment to right of each value
                    # repeat for max, and for difference between segment radiomics value and whole prostate value
                    maxLesionList.append(lesion_data[0][features + 1])
                    maxLesionInfo.append(lesion_data[0][0])

                else:
                    if lesion_data[lesion][features + 1] < minLesionList[features]:
                        minLesionList[features] = lesion_data[lesion][features + 1]  # update list with new minimum
                        minLesionInfo[features] = lesion_data[lesion][0]
                        # update segment associated with new minimum
                    if lesion_data[lesion][features + 1] > minLesionList[features]:
                        maxLesionList[features] = lesion_data[lesion][features + 1]  # update list with new maximum
                        maxLesionInfo[features] = lesion_data[lesion][0]
                        # update segment associated with new maximum
            ## Add volume to correct PIRADS category and identify index lesion
            PIRADS = voi_list[lesion][2]
            vol = lesion_data[lesion][1]
            if PIRADS > maxPIRADS:
                maxPIRADS = PIRADS
                indexVol = vol
                index = lesion
            elif PIRADS == maxPIRADS:
                if vol > indexVol:
                    maxPIRADS = PIRADS
                    indexVol = vol
                    index = lesion
            if PIRADS == 6:
                P6vol = P6vol + vol
            elif PIRADS == 5:
                P5vol = P5vol + vol
            elif PIRADS == 4:
                P4vol = P4vol + vol
            elif PIRADS == 3:
                P3vol = P3vol + vol
            else:
                P2vol = P2vol + vol
            total = P6vol + P5vol + P4vol + P3vol + P2vol
            P6burden = P6vol/prost_vol
            P5burden = P5vol/prost_vol
            P4burden = P4vol/prost_vol
            P3burden = P3vol/prost_vol
            P2burden = P2vol/prost_vol
            totalBurden = total/prost_vol
            P6perc = P6vol/total
            P5perc = P5vol/total
            P4perc = P4vol/total
            P3perc = P3vol/total
            P2perc = P2vol/total
        voi_data = [maxPIRADS] + [P6vol] + [P5vol] + [P4vol] + [P3vol] + [P2vol] + [total] \
                   + [P6burden] + [P5burden] + [P4burden] + [P3burden] + [P2burden] + [totalBurden] \
                   + [P6perc] + [P5perc] + [P4perc] + [P3perc] + [P2perc] \
                   + lesion_data[index] + minLesionList + maxLesionList + minLesionInfo + maxLesionInfo

        headers = ['PI-RADS','P6vol','P5vol','P4vol','P3vol','P2vol','totalvol',
                   'P6burden','P5burden','P4burden', 'P3burden','P2burden','totalburden',
                   'P6perc','P5perc','P4perc','P3perc','P2perc','indexLesion'] + header # adds up through index headers

        lesionHeaders = []

        #add min/max headers
        for head in header:
            headers = headers + ['minSingleLesion' + head[5:]] + ['maxSingleLesion' + head[5:]]
                # replaces 'index' with 'min' or 'max' at start of header name
            lesionHeaders = lesionHeaders + ['Lesion_with_min' + head[5:]] + ['Lesion_with_max' + head[5:]]

        headers = headers + lesionHeaders

        return [headers, voi_data]

    def SegRadiomics(self, patient, prost, voi_list, name, segArr, maskpath):
        segHeaders = []
        segData = []

        # for each segment, get radiomics data, diff between segment data and whole prostate data,
        # names of lesions overlapping segment, and max PIRADS score in segment
        [head,data] = (self.calculateRadiomics(patient, name, maskpath))
        diffHeader = []
        diffData = data[6:]
        for feature in range(len(prost)):
            diffData[feature] = diffData[feature] - prost[feature]
            diffHeader.append("diff_" + head[feature+6])
        segPIRADS = 0
        seglesions = 'N/A'
        numSeglesions = 0
        for lesions in voi_list:
            voi_mask = sitk.ReadImage(lesions[0])
            voi_arr = sitk.GetArrayFromImage(voi_mask)
            voiNZ = voi_arr.nonzero()
            for vox in range(len(voiNZ[0])):
                if segArr[voiNZ[0][vox],voiNZ[1][vox],voiNZ[2][vox]] == 1:
                    if seglesions == 'N/A':
                        seglesions = lesions[1]
                    else:
                        seglesions = seglesions + "; " + lesions[1]
                    numSeglesions = numSeglesions + 1
                    if lesions[2] > segPIRADS:
                        segPIRADS = lesions[2]
                    break
        segHeaders = head[6:] + diffHeader + ["segPIRADS"] + ["numSegLesions"] + ["segLesions"]
        segData = [name] + data[6:] + diffData + [segPIRADS] + [numSeglesions] + [seglesions]  # ignores all 6 shape features,
        # adds name at front for min/max identification
        return [segHeaders, segData]



    def calculateRadiomics(self, patient, name, mask):



        #name = 'name'
        #header = ['VOIname']
        data = []
        header_shape = []
        data_shape = []
        header_all = []
        data_all = []

        # imageName, maskName = radiomics.getTestCase('brain1')
        #maskName = r'T:\MIP\Katie_Merriman\Project1Data\PatientNormalized_data\0497587_20131125\wp_bt.nii.gz'
        #T2Name = r'T:\MIP\Katie_Merriman\Project1Data\PatientNormalized_data\0497587_20131125\T2n.nii.gz'
        #ADCName = r'T:\MIP\Katie_Merriman\Project1Data\PatientNormalized_data\0497587_20131125\ADCn.nii.gz'
        #highBName = r'T:\MIP\Katie_Merriman\Project1Data\PatientNormalized_data\0497587_20131125\highBn.nii.gz'

        maskName = mask
        T2Name = os.path.join(self.patientFolder,patient[0],'T2n.nii.gz')
        ADCName = os.path.join(self.patientFolder,patient[0],'ADCn.nii.gz')
        highBName = os.path.join(self.patientFolder,patient[0],'highBn.nii.gz')

        #if T2Name is None or maskName is None:  # Something went wrong, in this case PyRadiomics will also log an error
        #    print('Error getting testcase!')
        #    exit()

        # Regulate verbosity with radiomics.verbosity (default verbosity level = WARNING)
        # radiomics.setVerbosity(logging.INFO)

        # Get the PyRadiomics logger (default log-level = INFO)
        #logger = radiomics.logger
        #logger.setLevel(logging.DEBUG)  # set level to DEBUG to include debug log messages in log file

        # Set up the handler to write out all log entries to a file
        #handler = logging.FileHandler(filename='testLog.txt', mode='w')
        #formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
        #handler.setFormatter(formatter)
        #logger.addHandler(handler)

        # Define settings for signature calculation
        # These are currently set equal to the respective default values
        settings = {}
        settings['geometryTolerance'] = 0.0001
        settings['binWidth'] = 25
        settings[
            'resampledPixelSpacing'] = None  # [3,3,3] is an example for defining resampling (voxels with size 3x3x3mm)
        settings['interpolator'] = sitk.sitkBSpline

        # Initialize feature extractor
        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

        # By default, only original is enabled. Optionally enable some image types:
        # extractor.enableImageTypes(Original={}, LoG={}, Wavelet={})

        ## Shape only needs to be calculated once per mask - doesn't change based on T2/ADC/highB
        # Disable all classes except shape
        extractor.disableAllFeatures()
        extractor.enableFeatureClassByName('shape')
        # Only enable mean and skewness in firstorder
        extractor.enableFeaturesByName(
            shape=['MeshVolume', 'Elongation', 'Flatness',  'Sphericity', 'SurfaceArea', 'SurfaceVolumeRatio'])

        print("Calculating shape features for " + name)
        featureVector = extractor.execute(T2Name, maskName)

        for featureName in featureVector.keys():
            if 'original' in featureName:
                if not 'Mask' in featureName:
                    if not 'Image' in featureName:
                        #print("Computed %s: %s" % (featureName, featureVector[featureName]))
                        header_shape.append(name + featureName[7:])
                        data_shape.append(featureVector[featureName])  # returns array of size one for each feature.
                        # making it feature.min gives single value
                        #print(name + featureName[7:], featureVector[featureName])

        ## Calculate remaining features for T2/ADC/highB
        extractor.disableAllFeatures()
        extractor.enableFeatureClassByName('firstorder')
        extractor.enableFeatureClassByName('glcm')
        extractor.enableFeatureClassByName('glrlm')
        extractor.enableFeatureClassByName('glszm')
        extractor.enableFeatureClassByName('gldm')
        extractor.enableFeatureClassByName('ngtdm')
        '''
        firstorder
        shape
        glcm
        glrlm
        glszm
        gldm
        ngtdm
        '''

        print("Calculating T2 features")
        featureVector = extractor.execute(T2Name, maskName)

        for featureName in featureVector.keys():
            if 'original' in featureName:
                if not 'Mask' in featureName:
                    if not 'Image' in featureName:
                        #print("Computed %s: %s" % (featureName, featureVector[featureName]))
                        header_all.append(name + '_T2' + featureName[8:])
                        data_all.append(featureVector[featureName])  # returns array of size one for each feature.
                        # making it feature.min gives single value
                        #print(name + '_T2' + featureName[8:],featureVector[featureName])

        print("Calculating ADC features")
        featureVector = extractor.execute(ADCName, maskName)

        for featureName in featureVector.keys():
            if 'original' in featureName:
                if not 'Mask' in featureName:
                    if not 'Image' in featureName:
                        #print("Computed %s: %s" % (featureName, featureVector[featureName]))
                        header_all.append(name + '_ADC' + featureName[8:])
                        data_all.append(featureVector[featureName])  # returns array of size one for each feature.
                        # making it feature.min gives single value

        print("Calculating highB features")
        featureVector = extractor.execute(highBName, maskName)

        for featureName in featureVector.keys():
            if 'original' in featureName:
                if not 'Mask' in featureName:
                    if not 'Image' in featureName:
                        #print("Computed %s: %s" % (featureName, featureVector[featureName]))
                        header_all.append(name + '_highB' + featureName[8:])
                        data_all.append(featureVector[featureName])  # returns array of size one for each feature.
                        # making it feature.min gives single value

        header = header_shape + header_all
        for i in range(len(data_shape)):
            data_shape[i] = data_shape[i].min()
        for i in range(len(data_all)):
            data_all[i] = data_all[i].min()
        data = data_shape + data_all # saves data with volume first

        return [header, data]


if __name__ == '__main__':
    c = featureCalculator()
    c.calculate()
    print('Conversions complete!')