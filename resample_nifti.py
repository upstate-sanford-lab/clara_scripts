import SimpleITK as sitk
import numpy as np
import nibabel as nib
import os

class ResampleNifti4Clara:
    '''this script is designed to resampled properly labeled .nifti files to 1x1x1 for clara'''

    def __init__(self):
        self.imgpath='/home/tom/clara_experiments/kidney_data/RightKidney'
        self.savepath='/home/tom/clara_experiments/kidney_data/RightKidney_resampled'

    def resample_all_pts(self,imgn='img',segn='seg'):
        '''use function below, iterate over patients'''

        for file in os.listdir(self.imgpath):
            print('processing file {}'.format(file))
            id=file.split('_')[0]
            if id==imgn:
                self.resample_img(Input_path=os.path.join(self.imgpath,file), savename=file)
            if id==segn:
                self.resample_mask(Input_path=os.path.join(self.imgpath,file))

    def resample_img(self,Input_path, savename):
        '''
        resample the image
        :param Input_path:
        :param savename:
        :return:
        '''

        #print("Reading Dicom directory:", Input_path)
        image = sitk.ReadImage(os.path.join(Input_path))
        new_spacing = [1, 1, 1]
        orig_size = np.array(image.GetSize(), dtype=np.int)
        orig_spacing = np.array(image.GetSpacing())
        new_size = orig_size * (orig_spacing / new_spacing)
        new_size = np.ceil(new_size).astype(np.int)  # Image dimensions are in integers
        new_size = [int(s) for s in new_size]

        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator = sitk.sitkLinear
        resample.SetOutputSpacing(new_spacing)
        resample.SetSize(new_size)
        resample.SetOutputDirection(image.GetDirection())
        resample.SetOutputOrigin(image.GetOrigin())
        new_image = resample.Execute(image)
        sitk.WriteImage(new_image, os.path.join(self.savepath,str.replace(savename,'.nii.gz','-resampled.nii.gz')))

    def resample_mask(self,Input_path):
        '''
        Reesample the mask with image affine matrix to match the image
        '''

        # read in first image to get shape
        img_out = sitk.ReadImage(os.path.join(Input_path))
        image= sitk.ReadImage(os.path.join(os.path.split(Input_path)[0],'img_'+'_'.join(os.path.split(Input_path)[1].split('_')[1:])))

        new_spacing = [1, 1, 1]
        orig_size = np.array(img_out.GetSize(), dtype=np.int)
        orig_spacing = np.array(img_out.GetSpacing())
        new_size = orig_size * (orig_spacing / new_spacing)
        new_size = np.ceil(new_size).astype(np.int)  # Image dimensions are in integers
        new_size = [int(s) for s in new_size]

        # t2 resample
        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator = sitk.sitkLinear
        resample.SetOutputSpacing(new_spacing)
        resample.SetSize(new_size)
        resample.SetOutputDirection(image.GetDirection())
        resample.SetOutputOrigin(image.GetOrigin())
        image_resamp = resample.Execute(image) # t2
        new_image = resample.Execute(img_out)  # mask
        new_arr = sitk.GetArrayFromImage(new_image)
        new_arr[new_arr > 0] = 1
        new_image = sitk.GetImageFromArray(new_arr)
        new_image.CopyInformation(image_resamp)
        sitk.WriteImage(new_image, os.path.join(self.savepath,os.path.split(Input_path)[1].split('.')[0] + '-resampled.nii'))

    def compress_nii(self):
        '''recursively converts .nii files to .nii.gz and removes original .nii file
        :param path - path to directory that contains all files

        '''
        path=self.savepath
        for file in os.listdir(path):
            if file.endswith('.nii'):
                n_f = nib.load(os.path.join(path, file))
                nib.save(n_f, os.path.join(path, file + '.gz'))
                os.remove(os.path.join(path, file))


if __name__=="__main__":
    c=ResampleNifti4Clara()
    c.resample_all_pts()
    c.compress_nii()

