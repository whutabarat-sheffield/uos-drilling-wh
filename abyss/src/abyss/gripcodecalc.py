import pandas as pd
import numpy as np

class GripCodeCalc():
    def __init__(self):
        self.gripref_df = pd.DataFrame()
        self.gripref_df['code'] = [ i for i in range(1,26)] 
        self.gripref_df['G'] = [ 1.5875 * i for i in self.gripref_df['code'] ]
        self.gripref_df['GUL'] = [ x + 0.127 for x in self.gripref_df['G'] ] # added the tolerance of 0.127 mm
        self.gripref_df['GLL'] = np.concatenate([[0], self.gripref_df['GUL'][:-1]])# - 0.127]) # added the tolerance of 0.127 mm
        
    def length_code(self, depth):
        code = self.length_array([depth])
        return code[0]
 
    def length_array(self, depths):
        try:
            iterator = iter(depths)
        except TypeError:
            depths = [depths]
        else:
            pass
        code = np.dot(
            (np.array(depths)[:, None] > self.gripref_df['GLL'].values) &
            (np.array(depths)[:, None] <= self.gripref_df['GUL'].values),
            self.gripref_df['code'].values
        )
        return code