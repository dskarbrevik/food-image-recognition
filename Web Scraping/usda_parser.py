import pandas as pd


food_desc = pd.read_csv(
        'sr28/FOOD_DES.txt',
        sep='^',
        quotechar='~',
        encoding='latin_1',
        header=None,
        names=[
            'food_id',
            'food_group_id',
            'long_desc',
            'short_desc',
            'common_name',
            'brand_name',
            'data_quality_flag',
            'refuse_desc',
            'refuse_frac',
            'sci_name',
            'nitro_factor',
            'protein_factor',
            'fat_factor',
            'carb_factor'])
nutri_data = pd.read_csv(
        'sr28/NUT_DATA.txt',
        sep='^',
        quotechar='~',
        encoding='latin_1',
        header=None,
        names=[
            'food_id',
            'nutrient_id',
            'nutrient_value_100g',
            'data_count',
            'std_err',
            'source_code',
            'derivation_code',
            'surrogate_food_id',
            'added_nutrient_flag',
            'n_studies',
            'min',
            'max',
            'df',
            'l_err_bound',
            'h_err_bound',
            'stat_comment',
            'date_modified',
            'confidence_code'])
nutri_def = pd.read_csv(
        'sr28/NUTR_DEF.txt',
        sep='^',
        quotechar='~',
        encoding='latin_1',
        header=None,
        names=[
            'nutrient_id',
            'units',
            'tag_name',
            'nutri_desc',
            'precision',
            'sort_order'])
