### Download UCR dataset

import os
import requests
import pyzipper
import numpy as np
from tqdm import tqdm


datasetURL = "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip"
datasetPath = "./UCRArchive_2018.zip"
datasetTitle = "UCRArchive_2018.zip"

datalistURL = "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/DataSummary.csv"
datalistPath = "./DataSummary.csv"
datalistTitle = "DataSummary.csv"

dataname_list = []

# Inner Access Number of Constant Variable
_ID = 0
_Type = 1
_Train = 2
_Test = 3
_Class = 4
_Length = 5
_ED = 6
_DTW_learned_w = 7
_DTW_w_100 = 8
_Default_rate = 9
_Data_donor_editor = 10

# Define Data Information Class
class DataInfo() :
    def __init__(self):
        self.ID = 0
        self.type = ""
        self.train_num = 0
        self.test_num = 0
        self.class_num = 0
        self.length = 0
        self.editor = ""
    
    def get_info( self, dataset_name ):
        dataset_dict = get_datalist()
        
        self.train_num = int(dataset_dict[dataset_name][_Train])
        self.test_num = int(dataset_dict[dataset_name][_Test])
        self.class_num = int(dataset_dict[dataset_name][_Class])

        if(dataset_dict[dataset_name][_Length] != "Vary"):
            self.length = int(dataset_dict[dataset_name][_Length]) - 1


# Make Data Information Instance
datainfo = DataInfo()

def download():
  """
  Download the datasets ZIP file and the spreadsheet file of the datasets.
  """

  download_dataseet()
  download_datasetlist()
  
  return
  
def download_dataseet():
  """
  Download the datasets ZIP file.
  """
  
  if( os.path.isfile(datasetPath) ):
    print("UCR Archive is already downloaded.")
  else:
    print("Datasets Downloading...")

    file_size = int(requests.head(datasetURL).headers["content-length"])
    res = requests.get(datasetURL, stream=True)
    pbar = tqdm(total=file_size, unit="B", unit_scale=True)

    with open(datasetPath, 'wb') as file:
        for chunk in res.iter_content(chunk_size=1024):
            file.write(chunk)
            pbar.update(len(chunk))
        pbar.close()
          
  return

def download_datasetlist():
  """
  Download the spreadsheet file of the datasets.
  """
  if( os.path.isfile(datalistTitle) == False):
      print("Datalist Downloading...")

      file_size = int(requests.head(datalistURL).headers["content-length"])
      res = requests.get(datalistURL, stream=True)
      pbar = tqdm(total=file_size, unit="B", unit_scale=True)

      with open(datalistPath, 'wb') as file:
          for chunk in res.iter_content(chunk_size=1024):
              file.write(chunk)
              pbar.update(len(chunk))
          pbar.close()
          
  return

def extract( password ):
  """
  Extract the dataset files from ZIP file.
  """
  bytePwd = password.encode('utf-8')
  print("Extracting...")

  with pyzipper.AESZipFile(datasetPath) as f:
      f.extractall(pwd=bytePwd)

  print("Complete!")

  return

def download_and_extract( password ):
  """
  Download and extract the datasets and download the spreadsheet file of the datasets.
  """
  download()
  extract(password)

  return


def get_datalist():
  """
  Make dictionary of the names and themselves of datasets.
  """
  dataname = np.loadtxt(datalistPath, dtype="unicode", delimiter=",", skiprows=1, usecols=(2))
  datalist = np.loadtxt(datalistPath, dtype="unicode", delimiter=",", skiprows=1, usecols=(0,1,3,4,5,6,7,8,9,10,11))

  return( dict(zip(dataname, datalist)) )


def get_dataset_name_list():
  """
  Get the name list of the datasets.
  If there is not the spreadsheet file of the datasets, it will be downloaded.
  
  
  Returns
  ----------
  dataset_name : array of string
    List of the dataset names.
  """
  
  download_datasetlist()
  
  dataname = np.loadtxt(datalistPath, dtype="unicode", delimiter=",", skiprows=1, usecols=(2))
  
  return( dataname )


def get_data( datasetName ):
  """
  Get the train data and test data with labels.
  This function does not support "Vary" length dataset.
  If "Vary" length dataset is specified, return values are 0.
  
  Parameter
  ----------
  datasetName : string
    Target dataset name.

  Returns
  ----------
  train_data : array of float
    Train data.
  train_label : array of int
    Train data labels.
  test_data : array of float
    Test data.
  test_label : array of int
    Test data labels.
      
  """
  
  traindataPath = "./UCRArchive_2018/" + datasetName + "/" + datasetName + "_TRAIN.tsv"
  testdataPath = "./UCRArchive_2018/" + datasetName + "/" + datasetName + "_TEST.tsv"

  datalist_dict = get_datalist()

  datainfo.get_info( datasetName )

  if(datalist_dict[datasetName][_Length] != "Vary"):
      # Get data length from DataSummary file.
      dataLength = int(datalist_dict[datasetName][_Length])

      # Get train data
      train_data = np.loadtxt(traindataPath, dtype="float", delimiter= "\t", skiprows=0, usecols=(tuple(np.arange(1,dataLength))))

      # Get train label
      train_label = np.loadtxt(traindataPath, dtype="int", delimiter= "\t", skiprows=0, usecols=(0))

      # Get test data
      test_data = np.loadtxt(testdataPath, dtype="float", delimiter= "\t", skiprows=0, usecols=(tuple(np.arange(1,dataLength))))

      # Get train label
      test_label = np.loadtxt(testdataPath, dtype="int", delimiter= "\t", skiprows=0, usecols=(0))

      # print(train_data)
      # print(train_label)
      # print(test_data)
      # print(test_label)


  else:
      print("'Vary' length dataset is not supported.")
      train_data = np.arange(0)
      train_label = np.arange(0)
      test_data = np.arange(0)
      test_label = np.arange(0)

  return (train_data, train_label, test_data, test_label)
  
def get_dataset_information( datasetName ):
    """
    Get information of the dataset.
    
    Parameter
    ----------
    datasetName : string
      Target dataset name.
    
    Returns
    ----------
    traindata_num : int
      The number of train data.
    testdata_num : int
      The number of test data.
    class_num : int
      The number of class.
    data_length : int
      The length of data.
      If the length is "Vary", return value is 0.
      (data_length is the number subtracted 1 from Length number in the dataset spreadsheet.
       Because the Length number in the dataset spreadsheet includes label data.)
    """
    
    datalist_dict = get_datalist()

    datainfo.get_info( datasetName )

    traindata_num = int(datalist_dict[datasetName][_Train])
    testdata_num = int(datalist_dict[datasetName][_Test])
    class_num = int(datalist_dict[datasetName][_Class])
    
    if(str.isdecimal( datalist_dict[datasetName][_Length] )):
      data_length = int(datalist_dict[datasetName][_Length]) - 1
    elif( datalist_dict[datasetName][_Length] == "Vary" ):
      data_length = 0
    else:
      data_length = 0
      
    return(traindata_num, testdata_num, class_num, data_length)
