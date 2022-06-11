from cProfile import label
import math
import os
import numpy as np
import pandas as pd
from PIL import Image


class DataLoader():
    """
    The DataLoader class handles loading image data from respective folders regardless of platform while performing act-
    ions such as bootstrapping and sampling.  

    Attributes:
        platform (str): Name of the platform, options are ["Local", "Kaggle", "Colab"]
        n_classes (int): Number of classfication classes, options are [2, 3]
        data_dir (str): Folder directory containing the images, must contain the ./train/ and ./test/ folder
        txt_dir (str): Folder directory containing the label files such as "train_COVIDx9A.txt", by default = data_dir
        img_size (int): The final output image size after cropping and resizing
        combined (bool): Determines whether to combine original data's train and test sets for custom test split
        channels (str): Determines whether to output RGB or Greyscale (L) images
    """
    class_label = {
        3: ('train_COVIDx9A.txt', 'test_COVIDx9A.txt'),
        2: ('train.txt', 'test.txt')
    }
    
    platform_path = {
        'Local': './data/',
        'Kaggle': '/kaggle/input/covidx-cxr2/',
        'Colab': './content/'
    }

    label_colnames = ['patient_id', 'filename', 'class', 'data_source']

    label_encode = {
        'positive': 1,
        'negative': 0,
        'COVID-19': 2,
        'pneumonia': 1,
        'normal': 0
    }


    def __init__(self, platform: str = 'Local', n_classes: int = 2, data_dir: str = None, txt_dir: str = None,
                 img_size: int = 224, combined: bool = True, channels: str = 'RGB'):

        # data validation checks
        assert platform in self.platform_path, 'Platforms must be in ["Local", "Kaggle", "Colab"]'
        self.platform = platform

        assert n_classes in [2, 3], 'n_classes must be in [2, 3]'
        self.n_classes = n_classes

        # if directories not specified, take default
        if data_dir is None:
            self.data_dir = self.platform_path.get(platform)
        else:
            self.data_dir = data_dir

        if txt_dir is None:
            self.txt_dir = self.platform_path.get(platform)
        else:
            self.txt_dir = txt_dir

        self.img_size = img_size
        self.combined = combined

        assert channels in ['L', 'RGB'], "channels must be in ['L', 'RGB']"
        self.channels = channels

        # check if either label file is in txt_dir folder    
        for i in range(2):
            if not os.path.exists(os.path.join(self.txt_dir, self.class_label.get(self.n_classes)[i])):
                raise FileNotFoundError(self.class_label.get(self.n_classes)[i] + ' not found in ' + self.txt_dir)

        # print a summary message
        print(f'Platform: {self.platform}\nNum Classes: {self.n_classes}')
        print(f'Data Folder: {self.data_dir}\nLabel Folder: {self.txt_dir}')
        print(f'Image size: {self.img_size}\nCombined: {self.combined}\nImage Channels: {self.channels}')


    def __crop_image(self, img: Image) -> Image:
        """
        Crop and resize image to a square
        The image is cropped at the centre using the shorter length.
        It's then resized to the proposed dimensions.

        Args:
            img (Image): a PIL.Image object to be resized

        Returns:
            image_final (Image): a PIL.Image object that's resized
        """
        width, height = img.size
        diff = abs(width-height)

        # initialise final image parameters
        left, right, top, bottom = 0, width, 0, height

        # crop based on whether the difference in dimensions is odd or even
        if diff % 2 == 0:
            if width >= height:
                bottom = height
                left = diff / 2
                right = width - left
            elif height > width:
                top = diff / 2
                bottom = height - top
                right = width
        else:
            if width > height:
                bottom = height
                left = diff / 2 + 0.5
                right = width - left + 1
            elif height > width:
                top = diff / 2 + 0.5
                bottom = height - top + 1
                right = width

        # crop image into a square
        img_cropped = img.crop((left, top, right, bottom))
        # resize to desired shape
        img_final = img_cropped.resize((self.img_size, self.img_size))
        
        return img_final


    def __load_metadata(self) -> pd.DataFrame:
        """
        Loads the label txt files, return them as separate or combined pd.DataFrame

        Returns:
            train_df (pd.DataFrame): DF of all training images file names and labels
            test_df (pd.DataFrame): DF of all test images file names and labels
            combined_df (pd.DataFrame): DF of combined images file names and labels
        """
        # name of the label files
        train_filename, test_filename = self.class_label.get(self.n_classes)
        # read in label files
        train_df = pd.read_csv(os.path.join(self.txt_dir, train_filename), header=None, sep=' ')
        test_df = pd.read_csv(os.path.join(self.txt_dir, test_filename), header=None, sep=' ')

        train_df.columns = self.label_colnames
        test_df.columns = self.label_colnames

        # assign file path for each image
        train_df['filepath'] = train_df.apply(lambda row: os.path.join(self.data_dir, 'train/', row['filename']), axis=1)
        test_df['filepath'] = test_df.apply(lambda row: os.path.join(self.data_dir, 'test/', row['filename']), axis=1)

        # combine all images metadata
        combined_df = pd.concat([train_df, test_df], ignore_index=True)

        # return combined or separate based on argument
        if self.combined:
            return combined_df
        else:
            return train_df, test_df


    def __bootstrap_sample(self, label_df: pd.DataFrame, rand_state: int,
                           sample_size: int, sample_percent: float) -> pd.DataFrame:
        """
        Creates a bootstrapped training sample from a given population.
        Patients with multiple images have one image randomly selected with the remainder dropped from the data.
        The data is samples with replacement with an equal number of positive/negative images.
        The validation set is from remaining COVID-19 images and a sample of non-COVID.
        
        Arguments:
            label_df (pd.DataFrame): DF of image metadata
            rand_state (int): Random state for Pandas sampling
            sample_size (int): Total number of images in the training set
            sample_percent (float): Percentage of training images vs total images, if specified, overrides sample_size
            
        Returns:
            df_train (pd.DataFrame): DF of images metadata for training after bootstrapping
            df_val (pd.DataFrame): DF of images metadata for validation after bootstrapping
        """
        # defaults to sample_size, unless sample_percent is specified
        if sample_percent is not None:
            sample_size = math.floor(label_df.shape[0] * sample_percent)
        else:
            assert sample_size <= label_df.shape[0], f'sample_size must be no larger than {label_df.shape[0]}'

        sample_half = math.floor(sample_size/2)

        # randomly pick one image from a patient and drop the rest from the sample population
        patient_img_count = label_df[['patient_id', 'filename']].groupby(['patient_id']).count().reset_index()
        duplicate_patients = patient_img_count['patient_id'][patient_img_count['filename'] > 1]
        
        for patient in duplicate_patients.unique():
            sample_row = label_df.loc[label_df['patient_id'] == patient, :].sample(n=1, random_state=rand_state)
            label_df = label_df.loc[label_df['patient_id'] != patient, :]
            label_df = pd.concat([label_df, sample_row], axis=0)

        # sample with replacement for positive/negative classes to create training data
        df_train_positive = label_df[label_df['class'] == 'positive'].sample(n=sample_half,
                                                                             replace=True, random_state=rand_state)
        df_train_negative = label_df[label_df['class'] == 'negative'].sample(n=sample_half,
                                                                             replace=True, random_state=rand_state)
        
        df_train = pd.concat([df_train_positive, df_train_negative], axis=0)
        df_remaining = label_df.loc[label_df['filename'].isin(df_train['filename']) == False, :]
        
        # remaining has lots of negative samples so we sample without replacement to match positive samples
        df_val_size = min(df_train_positive.shape[0], df_remaining[df_remaining['class']=='positive'].shape[0])

        df_val_positive = df_remaining.loc[df_remaining['class'] == 'positive', :].sample(n=df_val_size,
                                                                                          replace=False,
                                                                                          random_state=rand_state)
        df_val_negative = df_remaining.loc[df_remaining['class'] == 'negative', :].sample(n=df_val_positive.shape[0],
                                                                                          replace=False,
                                                                                          random_state=rand_state)
        
        df_val = pd.concat([df_val_positive, df_val_negative], axis=0)
        
        # shuffle datasets before returning
        df_train = df_train.sample(frac=1, random_state=rand_state).reset_index(drop=True)
        df_val = df_val.sample(frac=1, random_state=rand_state).reset_index(drop=True)

        return df_train, df_val

    
    def __normal_sample(self, label_df: pd.DataFrame, rand_state: int,
                        sample_size: int, sample_percent: float) -> pd.DataFrame:
        """
        Creates a normal set of training sample from a given population.
        Patients with multiple images are not split between datasets to prevent leakage.
        
        Arguments:
            label_df (pd.DataFrame): DF of image metadata
            rand_state (int): Random state for Pandas sampling
            sample_size (int): Total number of images in the training set
            sample_percent (float): Percentage of training images vs total images, if specified, overrides sample_size
            
        Returns:
            df_train (pd.DataFrame): DF of images metadata for training (or testing)
            df_left (pd.DataFrame): DF of images metadata for images not selected in df_train
        """
        if sample_percent is not None:
            assert sample_percent > 0 and sample_percent <= 1.00, 'sample_percent must be in range (0, 1.00]'
            sample_size = math.floor(label_df.shape[0] * sample_percent)
        else:
            assert sample_size <= label_df.shape[0], f'sample_size must be no larger than {label_df.shape[0]}'

        sample_quarter = math.floor(sample_size/4)

        positive_patients = label_df.loc[label_df['class'] == 'positive', :]

        # DF of positive patients and their positive image count, then random shuffle to be sampled
        positive_img_count = positive_patients[['patient_id', 'filename']].groupby(['patient_id']).count().reset_index()
        positive_img_count = positive_img_count.sample(frac=1, random_state=rand_state)

        # sample positive patients until total number of images equals required number
        positive_count = 0
        positive_patients = []

        for row in positive_img_count.iterrows():
            positive_count += row[1]['filename']
            positive_patients.append(row[1]['patient_id'])
            if positive_count > sample_quarter:
                break
        
        # repeat the process above for patients with negative images, however, remove any patients already picked
        # this is to avoid cases where someone has images in both class, so they don't get picked twice
        negative_patients = label_df.loc[(label_df['class'] == 'negative') &
                                         (~label_df['patient_id'].isin(positive_patients)), :]

        negative_img_count = negative_patients[['patient_id', 'filename']].groupby(['patient_id']).count().reset_index()
        negative_img_count = negative_img_count.sample(frac=1, random_state=rand_state)

        negative_count = 0
        negative_patients = []

        for row in negative_img_count.iterrows():
            negative_count += row[1]['filename']
            negative_patients.append(row[1]['patient_id'])
            if negative_count > sample_quarter:
                break

        # combined all patients picked and filter out their iamges
        train_patients = positive_patients + negative_patients

        df_train = label_df[label_df['patient_id'].isin(train_patients)]
        df_left = pd.concat([label_df, df_train, df_train]).drop_duplicates(keep=False)

        return df_train, df_left


    def __test_split(self, test_percent: float) -> pd.DataFrame:
        """
        Split out the holdout test set from the whole data
        This is only for when the original data is combined, otherwise the original data defined the holdout test set
        
        Arguments:
            test_percent (float): Percentage of holdout images vs total images

        Returns:
            df_train_val (pd.DataFrame): DF of images metadata for training and validation
            df_test (pd.DataFrame): DF of images metadata for holdout testing
        """
        assert test_percent > 0 and test_percent <= 1.00, 'test_percent must be in range (0, 1.00]'

        # load in combined data
        label_df = self.__load_metadata()
        test_size = math.floor(label_df.shape[0] * test_percent)

        # obtain the proportion of positive class in the data
        positive_percent = label_df['class'].value_counts(normalize=True)['positive']
        # determine the number of positive images in the holdout set for balanced test data
        positive_size = math.floor(test_size * positive_percent)

        positive_patients = label_df.loc[label_df['class'] == 'positive', :]

        # DF of positive patients and their positive image count, then random shuffle to be sampled
        positive_img_count = positive_patients[['patient_id', 'filename']].groupby(['patient_id']).count().reset_index()
        positive_img_count = positive_img_count.sample(frac=1, random_state=50)


        # sample positive patients until total number of images equals required number
        positive_count = 0
        positive_patients = []

        for row in positive_img_count.iterrows():
            positive_count += row[1]['filename']
            positive_patients.append(row[1]['patient_id'])
            if positive_count > positive_size:
                break
        
        # repeat the process above for patients with negative images, however, remove any patients already picked
        # this is to avoid cases where someone has images in both class, so they don't get picked twice
        negative_patients = label_df.loc[(label_df['class'] == 'negative') &
                                         (~label_df['patient_id'].isin(positive_patients)), :]

        negative_img_count = negative_patients[['patient_id', 'filename']].groupby(['patient_id']).count().reset_index()
        negative_img_count = negative_img_count.sample(frac=1, random_state=50)

        negative_count = 0
        negative_patients = []

        for row in negative_img_count.iterrows():
            negative_count += row[1]['filename']
            negative_patients.append(row[1]['patient_id'])
            if negative_count + positive_count > test_size:
                break

        # combined all patients picked and filter out their iamges
        test_patients = positive_patients + negative_patients

        df_test = label_df[label_df['patient_id'].isin(test_patients)]
        df_train_val = pd.concat([label_df, df_test, df_test]).drop_duplicates(keep=False)

        return df_train_val, df_test
    
    
    def __train_test_split(self, rand_state: int, bootstrap: bool, sample_size: int, sample_percent: float,
                           test_percent: float, load_full_train: bool = False) -> pd.DataFrame:
        """
        Split to train, val, test based on user choices
        
        Arguments:
            rand_state (int): Random state that will determine the images selected
            bootstrap (bool): Determines whether to bootstrap the data or not
            sample_size (int): Total number of images in the training set
            sample_percent (float): Percentage of training images vs total images, if specified, overrides sample_size
            test_percent (float): Percentage of holdout images vs total images
            load_full_train (bool): Determines whether to return all training data
        Returns:
            df_train (pd.DataFrame): DF of images metadata for training 
            df_val (pd.DataFrame): DF of images metadata for validation
            df_test (pd.DataFrame): DF of images metadata for holdout testing
        """
        if self.combined:
            df_train_val, df_test = self.__test_split(test_percent=test_percent)
        else:
            df_train_val, df_test = self.__load_metadata(combined=self.combined)

        if load_full_train:
            df_train = df_train_val
            df_val = None
        elif bootstrap:
            df_train, df_val = self.__bootstrap_sample(label_df=df_train_val, rand_state=rand_state,
                                                       sample_size=sample_size, sample_percent=sample_percent)
        else:
            df_train, df_left = self.__normal_sample(label_df=df_train_val, rand_state=rand_state,
                                                        sample_size=sample_size, sample_percent=sample_percent)
            df_val, _ = self.__normal_sample(label_df=df_left, rand_state=rand_state,
                                                sample_size=sample_size, sample_percent=sample_percent)

        return df_train, df_val, df_test


    def __load_images(self, image_df: pd.DataFrame) -> np.array:
        """
        Loads in the images based on the input image metadata

        Arguments:
            image_df (pd.DataFrame): DF of images metadata, which includes their file paths

        Returns:
            images (np.array): A (no of images) x (img_size^2 x channels) array containing images specified in image_df
        """
        images = []

        for idx, file_path in enumerate(image_df['filepath']):
            img = Image.open(file_path)
            img = img.convert(self.channels)
            img_final = self.__crop_image(img)
            img_array = np.asarray(img_final).flatten()

            images.append(img_array)

        images = np.array(images)

        return images


    def __load_labels(self, image_df: pd.DataFrame) -> np.array:
        """
        Loads in the image labels based on the input image metadata

        Arguments:
            image_df (pd.DataFrame): DF of images metadata, which includes their file paths

        Returns:
            labels (np.array): A (no of images x 1) array containing image labels
        """
        labels = []

        for label in image_df['class']:
            labels.append(self.label_encode.get(label))

        labels = np.array(labels)

        return labels


    def load_train_val(self, rand_state: int = 1, bootstrap: bool = True, load_full_train: bool = False,
                       load_val: bool = True, sample_size: int = 3000, sample_percent: float = None) -> np.array:
        """
        Loads in the training and validation images based on the input image metadata

        Arguments:
            rand_state (int): Random state that will determine the images selected
            bootstrap (bool): Determines whether to bootstrap the data or not
            load_full_train (bool): Determines whether to return all training data (also overwrites load_val to False)
            load_val (bool): Determines whether to load the validation data
            sample_size (int): Total number of images in the training set
            sample_percent (float): Percentage of training images vs total images, if specified, overrides sample_size

        Returns:
            X_train (np.array): A (no of images) x (img_size^2 x channels) array containing training images
            X_val (np.array): A (no of images) x (img_size^2 x channels) array containing validation images
            Y_train (np.array): A (no of images x 1) array containing training image labels
            Y_val (np.array): A (no of images x 1) array containing validation image labels
        """
        if sample_percent is not None:
            assert sample_percent > 0 and sample_percent <= 1.00, 'sample_percent must be in range (0, 1.00]'

        df_train, df_val, _ = self.__train_test_split(rand_state=rand_state, bootstrap=bootstrap,
                                                      sample_size=sample_size, sample_percent=sample_percent,
                                                      test_percent=0.1, load_full_train=load_full_train)

        X_train = self.__load_images(df_train)
        Y_train = self.__load_labels(df_train)

        if load_val and not load_full_train:
            X_val = self.__load_images(df_val)
            Y_val = self.__load_labels(df_val)
        else:
            X_val = None
            Y_val = None

        return X_train, X_val, Y_train, Y_val


    def load_test(self, test_percent: float = 0.1) -> np.array:
        """
        Loads in the holdout test images.
        The images loaded are always fixed given the same test_percent

        Arguments:
            test_percent (float): Percentage of holdout images vs total images

        Returns:
            X_test (np.array): A (no of images) x (img_size^2 x channels) array containing holdout test images
            Y_test (np.array): A (no of images x 1) array containing holdout test image labels   
        """
        assert test_percent > 0 and test_percent <= 1.00, 'test_percent must be in range (0, 1.00]'

        _, _, df_test = self.__train_test_split(rand_state=1, bootstrap=True, sample_size=3000, 
                                                sample_percent=None, test_percent=test_percent, load_full_train=False)

        X_test = self.__load_images(df_test)
        Y_test = self.__load_labels(df_test)

        return X_test, Y_test