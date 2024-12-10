import sys
from os.path import abspath, join
import bz2
import pandas as pd
from pathlib import Path

THIS_DIR = abspath('.')
CODE_DIR = abspath(join(THIS_DIR, '..', '..', 'sandbox'))
SRC_DIR = abspath(join(THIS_DIR, '..', '..', 'abyss','src'))
sys.path.append(THIS_DIR)
sys.path.append(abspath('..'))
sys.path.append(CODE_DIR)
sys.path.append(SRC_DIR)

# from dataparser import loadSetitecXls
from ingest_xls import create_dataframe_from_xls_files__firmware_v3


if __name__ == "__main__":
    #\\fr0-vsiaas-5706\EDU_Pipe\EDU_Incoming_Backup\Hamburg\SETITEC
    #\\fr0-vsiaas-5706\EDU_Pipe\EDU_Incoming_Backup\StNazaire\SETITEC
    HAM_FOLDER_ROOT = r"\\fr0-vsiaas-5706\EDU_Pipe\EDU_Incoming_Backup\Hamburg\SETITEC"
    # SNZ_FOLDER_ROOT = r"\\fr0-vsiaas-5706\EDU_Pipe\EDU_Incoming_Backup\StNazaire\SETITEC"

    # HAM_folder_content_filename = join(abspath('.'), 'scripts', 'windo', 'hamburg', r"FOLDER-CONTENT_HAM-SETITEC_UTC20230222-003346.csv.bz2")
    HAM_folder_content_filename = join(abspath('.'), r"FOLDER-CONTENT_HAM-SETITEC_UTC20230222-003346.csv.bz2")

    with bz2.open(HAM_folder_content_filename, mode='rt') as content_file:
        # df_setitec_drive = vaex.from_csv(content_file)
        df_setitec_drive = pd.read_csv(content_file, usecols=list(range(10))) # added usecols because some columns are not in the first 10 rows

    setitec_folder = Path(HAM_FOLDER_ROOT)
    # mask = df_setitec_drive['machine_id'] == 20110019
    # bare_filelist = df_setitec_drive[mask].file.tolist()
    file_dir = df_setitec_drive.file_dir.tolist()
    file = df_setitec_drive.file.tolist()
    pathlist = list(zip(file_dir, file))
    o = map(lambda x: setitec_folder / x[0] / x[1], pathlist)
    filelist = list(o)
    # may need to drop the first file (488_1) because it's an air drilling
    # filelist = filelist[1:]

    df = create_dataframe_from_xls_files__firmware_v3(filelist)
    df.to_parquet('df.parquet.gzip', compression='gzip')