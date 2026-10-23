import unittest
import numpy as np
import pandas as pd
from model import FEATURES, candidates, load_data, validate_record

class ModelTests(unittest.TestCase):
    def test_clean_data_and_no_duplicate_leakage(self):
        df, audit = load_data()
        self.assertEqual(audit['invalid_target_rows_removed'], 1)
        self.assertEqual(audit['exact_duplicates_removed'], 1)
        self.assertEqual(len(df), 1337)
        self.assertFalse(df.duplicated().any())
    def test_invalid_inputs_rejected(self):
        base = dict(age=30, bmi=25, children=0, gender='female', smoker='no', region='northwest')
        for change in [dict(age=-1), dict(bmi=float('nan')), dict(children=1.5), dict(smoker='maybe')]:
            with self.assertRaises(ValueError): validate_record(base | change)
    def test_pipeline_handles_missing_and_unseen_values(self):
        df, _ = load_data()
        model = candidates()['Ridge regression'].fit(df[FEATURES],df.charges)
        sample=df[FEATURES].iloc[:1].copy()
        sample.loc[sample.index[0], 'bmi']=np.nan
        sample.loc[sample.index[0], 'region']='unseen'
        self.assertTrue(np.isfinite(model.predict(sample)).all())

if __name__=='__main__': unittest.main()
