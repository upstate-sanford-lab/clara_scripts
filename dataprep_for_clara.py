import os
import nibabel as nib
import random
import shutil
import json


class ToClaraFormat:
    '''this class seeks to simplify the process of preparing data for clara train
    -- the format expected by this is a nifti format (.nii or .nii.gz)
    -- for scripts to help convert to nifti, see github.com/sanford-upstate
    -- labeling is as follows [img/segmentation] + '_' +[image and seg ID]+'.nii.gz
        -- this labeling scheme is important for sorting into a training and validation dataset
        -- all the data needs to be placed within a single director file called 'all_files'
    --assuming the data is already resampled, first execute the sort_data method to get the data into one directory
    -- then execute the clara_filestructure method
    '''
    random.seed(0)

    def __init__(self):
        self.rootdir='/home/tom/Documents/clara/data'
        self.basepath='/home/tom/clara_experiments/data_prostate'            #path to directory that contains files 'all_files'
        self.clarapath='/workspace/data/data_prostate/'
        self.center='SUNY_wp_for_clara'

    def clara_filestructure(self, i_name='img', s_name='seg', val_p=0.2, train_val_n=['training', 'validation'],resample=False):
        '''
        takes images and labels and splits them into train, val_test and creates .json
        note - basepath is the path with the directory 'all_files' within it
        :param i_name: (str) name of image files
        :param s_name: (str) name of segmentation files
        :param val_p (float) percent in validation dataset
        :return:
        '''

        # set up file structure
        path = os.path.join(self.basepath, self.center)  # replace if you have a different file name

        # set up saving filestructure
        if not os.path.exists(os.path.join(self.basepath, self.center + '_split')):
            os.mkdir(os.path.join(self.basepath, self.center + '_split'))
        s_path=os.path.join(self.basepath, self.center + '_split')

        # find all image files and all segmentation files
        img_dir = []; seg_dir = []
        for file in os.listdir(path):
            if file.split("_")[0] == i_name: img_dir += [file]
            if file.split("_")[0] == s_name: seg_dir += [file]

        # check all images have a counterpart
        for img in sorted(img_dir):
            if not s_name+'_' + img.split('_')[1].split('.')[0] + '.nii.gz' in seg_dir:
                print('img {} does not have a segmentation!').format(img)
                raise ValueError

        # split data into training and validation datasets
        val_sample = random.sample(img_dir, int(len(img_dir) * val_p))
        train_sample = set(img_dir) - set(val_sample)
        sample_dict = {train_val_n[0]: train_sample, train_val_n[1]: val_sample}
        for key in sample_dict.keys():
            if not os.path.exists(os.path.join(s_path, key)):
                os.mkdir(os.path.join(s_path, key))
            [shutil.copy2(os.path.join(path, file), os.path.join(s_path, key, file)) for file in
             sample_dict[key]]
            [shutil.copy2(os.path.join(path, s_name + '_' + file.split('_')[1]),
                          os.path.join(s_path, key, s_name + '_' + file.split('_')[1])) for file in
             sample_dict[key]]

        # building data structure to save as json file
        json_d = {

            "description": "Awesome",
            "modality": {"0": "T2"},
            "labels": {"0": "background", "1": "WP"},
            "reference": "SUNY",
            "tensorImageSize": "3D",
            "name": "Prostate",

        }
        for db in train_val_n:
            db_l = []
            for file in os.listdir(os.path.join(s_path, db)):
                if file.split('_')[0] == i_name:
                    db_l += [{'image': os.path.join(self.clarapath,self.center+'_split', db, file),
                              'label': os.path.join(self.clarapath,self.center+'_split', db,
                                                    s_name + '_' + file.split('_')[1])}]
            json_d[db] = db_l

        # saving as json file
        with open(os.path.join(s_path, 'datalist.json'), 'w') as outfile:
            json.dump(json_d, outfile,indent=2)


    def sort_data(self,anon=True):
        '''make savedir and sort img and seg files into the savedirectory properly labeled'''

        basepath=self.rootdir
        prostateX_n='SUNY_prostates'
        savedir='SUNY_prostates_for_clara'

        print("copying files")
        for pt in sorted(os.listdir(os.path.join(basepath,prostateX_n))):
            mask_name=find_file_by_annotator(os.path.join(basepath, prostateX_n,pt,'nifti','mask'))
            if mask_name==None:
                print("mask not found for patient {}".format(pt))
                continue
            if anon==True:
                anon_pt=str(random.randint(1000000000,9999999999))
                mask_path=os.path.join(basepath,prostateX_n,pt,'nifti','mask',mask_name)
                shutil.copy2(mask_path,os.path.join(basepath,savedir,'seg_'+anon_pt+'.nii'))
                img_file_path=os.path.join(basepath, prostateX_n,pt,'nifti','t2','t2_resampled.nii.gz')
                shutil.copy2(img_file_path,os.path.join(basepath,savedir,'img_'+anon_pt+'.nii.gz'))

            else:
                mask_path=os.path.join(basepath,prostateX_n,pt,'nifti','mask',mask_name)
                shutil.copy2(mask_path,os.path.join(basepath,savedir,'seg_'+pt.split('_')[0]+'.nii'))
                img_file_path=os.path.join(basepath, prostateX_n,pt,'nifti','t2','t2_resampled.nii.gz')
                shutil.copy2(img_file_path,os.path.join(basepath,savedir,'img_'+pt.split('_')[0]+'.nii.gz'))

        #compress all the uncompressed nifti files
        print('compressing files')
        compress_nii(os.path.join(basepath,savedir))

    def random_split_by_center(self):
        '''randomply split data into three datasets'''

        basepath=os.path.join(self.rootdir,'prostateX_for_clara')

        #split files into three groups randomly
        filelist=[file for file in os.listdir(basepath) if file.split('_')[0]=='img']
        sample1=random.sample(filelist,100)
        remaining_ds_1=set(filelist)-set(sample1)
        sample2=random.sample(remaining_ds_1,100)
        remaining_ds_2=set(filelist)-(set(sample1+sample2))
        sample3=random.sample(remaining_ds_2,100)

        #sanity check
        print(len(sample1)); print(len(sample2)); print(len(sample3))
        print(set(sample1).intersection(set(sample2)))
        print(set(sample1).intersection(set(sample3)))
        print(set(sample2).intersection(set(sample3)))

        ctr_d={'UCLA':sample1,'NCI':sample2,'SUNY':sample3}
        for center in ctr_d.keys():
            os.mkdir(os.path.join(basepath,center))
            c_sample=ctr_d[center]
            for file in c_sample:
                segname='seg_'+file.split('_')[1]
                shutil.move(os.path.join(basepath,file),os.path.join(basepath,center,file))
                shutil.move(os.path.join(basepath,segname),os.path.join(basepath,center,segname))

        os.mkdir(os.path.join(basepath,'leftover'))
        for file in os.listdir(basepath):
            if file.endswith('.nii.gz'):
                shutil.move(os.path.join(basepath,file),os.path.join(basepath,'leftover',file))

##########helper functions##########3
def compress_nii(path):
    '''recursively converts .nii files to .nii.gz and removes original .nii file
    :param path - path to directory that contains all files

    '''
    for file in os.listdir(path):
        if file.endswith('.nii'):
            n_f=nib.load(os.path.join(path,file))
            nib.save(n_f,os.path.join(path,file+'.gz'))
            os.remove(os.path.join(path, file))
        if os.path.isdir(os.path.join(path,file)):
            compress_nii(os.path.join(path,file))

def find_file_by_annotator(path='/home/tom/Desktop/prostateX/PEx0000_00000000/nifti/mask',type='wp'):
    for file in sorted(os.listdir(path)):
        #print(file)
        if len(file.split('_')) < 5:
            if file== type+'_bt_resampled.nii': return file
            elif file == type+'_mm_resampled.nii': return file
            elif file == type+'_ts_resampled.nii': return file
            elif file == type+'_pseg_resampled.nii':return file
            elif file == type+'_dk_resampled.nii': return file

def build_json_FL(val_p=0.2,center='SUNY',i_name='img',s_name='seg'):

    path='/home/tom/clara_experiments/FL/dataset/ProstateX_FL_Data/SUNY'
    json_d = {
        "description": "Whole Prostate Segmentation",
        "modality": {"0": "T2"},
        "labels": {"0": "background", "1": "prostate"},
        "name":"Prostate",
        "reference": center,
        "tensorImageSize": "3D",
}

    #split data into training and validation datasets
    files=os.listdir(os.path.join(path,'Image'))
    val_ids = [file.split('_')[1] for file in random.sample(files, int(len(files) * val_p))]
    train_ids = [file.split('_')[1] for file in list(set(files) - set(val_ids))]
    ID_dict={'validation':val_ids,"training":train_ids}

    for sample in ID_dict.keys():
        db_l = []
        for id in ID_dict[sample]:
            db_l += [{'image': os.path.join(center, 'Image', i_name+'_'+id),
                      'label': os.path.join(center, 'Mask', s_name+'_'+id)}]
        json_d[sample] = db_l

    #saving as json file
    with open(os.path.join(os.path.dirname(path),'Json', 'datalist.json'), 'w') as outfile:
        json.dump(json_d, outfile, indent=2)


if __name__=='__main__':
    c=ToClaraFormat()
    build_json_FL()
