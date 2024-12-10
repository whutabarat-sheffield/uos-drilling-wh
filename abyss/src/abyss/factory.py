import abyss.dataparser as dp
import os
from nptdms import TdmsFile
import warnings

class FileLoader:
    '''
        FileLoader is a class-factory for simplifying how to load the various data files.
        It is designed to provide a simpler interface for loading the files
        It uses the functions defined in dataparser to load the files into memory and handle
        converting to the target data type

        Use load_file to load the data stored in the file
    '''

    def load_file(fname,**kwargs):
        '''
            Load the target file into memory and return the object.

            The kwargs control the format the data is returned as, file specific options and any other
            required data

            Inputs:
                fname : Path to source file
                kwargs : Options + additional data
        '''
        return _load_file(fname,**kwargs)

    def _load_file(fname,**kwargs):
        # get extension
        ext = os.path.splitext(os.path.basename(fname))[1]
        # if it's a NPZ file
        if ext == '.npz':
            # if the user hasn't specified the columns text file
            if (not ('columns' in kwargs)) or (not ('cols' in kwargs)):
                # check if an accompanying file exists in the same location as source
                if os.path.isfile(os.path.splitext(fname)[1] + '.txt'):
                    columns = os.path.splitext(fname)[1] + '.txt'
                else:
                    raise ValueError(f"Missing accompanying columns text file for {fname}")
            else:
                columns = kwargs['columns' if 'columns' in kwargs else 'cols']
            # if the user wants a dataframe
            if kwargs.get('as_df',False):
                return dp.loadSetitecNPZ(fname,columns)
            elif kwargs.get('as_np',False):
                return dp.loadSetitecNPZ(fname,columns,as_np=True)
            elif kwargs.get('as_rec',False):
                return dp.loadSetitecNPZ(fname,columns,as_rec=True)
            else:
                return dp.loadSetitecNPZ(fname,columns)
        # if it's a MAT file
        elif ext == '.mat':
            data_sz,units = dp.estimateMatDataSize(fname,True)
            # if estimated data shape is greater than 1GB, raise a warning
            if data_sz > 1e+9:
                warnings.warn(f"Estimated Data Size is greather than 1 GB {str(data_sz)+units}!\nExtracting the data will take some time and a lot of memory.")
            # estimate the total mat size
            # check if it's a MAT of change points
            is_cp = self._isChangePoints(fname)
            # if it's a pandas Dataframe
            if kwargs.get('as_df',False):
                if is_cp:
                    return dp.getMatChangePoints(fname,as_df=True)
                else:
                    return dp.getMatData(fname,**kwargs)
            # if it's a Numpy array
            elif kwargs.get('as_np',False):
                if is_cp:
                    dp.getMatChangePoints(fname,as_df=True).to_records()
                else:
                    dp.getMatData(fname,**kwargs).to_records()
            # default array
            else:
                return dp.getMatData(fname,**kwargs)
        # if it's a TDMS file
        elif ext == '.tdms':
            if kwargs.get('as_df',False):
                return self._loadTDMSAsDF(fname)
            elif kwargs.get('as_np',False):
                return self._loadTDMSAsRec(fname)
            else:
                return self._loadTDMSAsDF(fname)
        # if it's a spreadsheet
        elif ext == '.xls':
            # check if the file is a measurement file
            is_meas = self._isMeasurement(fname)
            # if the user wants the data as a numpy array
            if kwargs.get('as_np',False):
                # if it's a measurement array
                if is_meas:
                    return dp.loadMeasurementsAsNP(fname)
                else:
                    return dp.loadSetitecXlsAsNP(fname)
            # if the user wants the data as a pandas array
            elif kwargs.get('as_df',False):
                if is_meas:
                    return dp.loadMeasurementsAsPD(fname)
                else:
                    return dp.loadSetitecXls(fname)
            # if the user just wants it raw
            else:
                if is_meas:
                    return dp.loadMeasurementsAsPD(fname)
                else:
                    return dp.loadSetitecXls(fname)

    def _loadTDMSAsDF(fname):
        return TdmsFile.read(fname).as_dataframe()

    def _loadTDMSAsRec(fname):
        return TdmsFile.read(fname).as_dataframe().to_records()
 
    def _isMeasurement(fname):
        # read first line, strip characters and check for the presence of a star in it
        # if there's a star then it's one of the exported file
        # if there is NOT a start then it's a manually constructed measurement file
        return not ('*' in open(fname,'r').readline().strip())

    def _isChangePoints(fname):
        with h5py.File(fname,'r') as source:
            # get key of source array
            keys = list(source.keys())
            data_key = list(filter(lambda kk : not ('#' in kk),keys))[0]
            # if it's a cell then it's change points
            # if it's a cell_array then it's a cell array
            return source[data_key].attrs['MATLAB_class'] == b'cell'
            
