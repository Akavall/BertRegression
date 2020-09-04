
import pandas as pd 
import random as rn

import sys 
import os 

sys.path.append(os.getcwd())

from src import data_sources as ds

def make_prev_sale_date(row):

    num_to_month = {
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December"
    }

    template = "Sold in {} {}.".format(num_to_month[row["MoSold"]], row["YrSold"])

    return template.format(num_to_month[row["MoSold"]], row["YrSold"])

def make_square_footage(row):

    templates_one_floor = [
        "First floor square footage is {} square feet.",
        "First floor square footage is {} sq. ft.",
        "First floor area is {} square feet.",
        "First floor area is {} sq. ft.",
        "First floor area is {} square meters.",
        "First floor area is {} sq. m.",
    ]

    templates_two_floors = [
        "First floor square footage is {} square feet and second floor square footage is {} square feet.",
        "First floor square footage is {} sq. ft. and second floor square footage is {} sq. ft.",
        "First floor area is {} square feet and second floor area is {} square feet.",
        "First floor area is {} sq. ft. and second floor area is {} sq. ft.",
        "First floor area is {} square meters and second area is {} square meters.",
        "First floor area is {} sq. m. and second area is {} sq. m.",
    ]

    if row["2ndFlrSF"] > 0:
        template = rn.choice(templates_two_floors)

        if "m" in template:
            return template.format(int(row["1stFlrSF"] / 10.764), int(row["2ndFlrSF"] / 10.764))

        return template.format(row["1stFlrSF"], row["2ndFlrSF"])
    
    else:
        template = rn.choice(templates_one_floor)

        if "m" in template:
            return template.format(int(row["1stFlrSF"] / 10.764))

        return template.format(row["1stFlrSF"])
        
def make_baths_counts(row):

    if row["FullBath"] > 1 and row["HalfBath"] > 1:
        return "{} full bathrooms and {} half bathrooms.".format(row["FullBath"], row["HalfBath"])
    elif row["FullBath"] == 1 and row["HalfBath"] > 1:
        return "{} full bathroom and {} half bathrooms.".format(row["FullBath"], row["HalfBath"])
    elif row["FullBath"] > 1 and row["HalfBath"] == 1:
        return "{} full bathrooms and {} half bathroom.".format(row["FullBath"], row["HalfBath"])
    elif row["FullBath"] == 1 and row["HalfBath"] == 1:
        return "{} full bathroom and {} half bathroom.".format(row["FullBath"], row["HalfBath"])

    elif row["FullBath"] > 1 and row["HalfBath"] == 0:
        return "{} full bathrooms.".format(row["FullBath"])
    elif row["FullBath"] == 1 and row["HalfBath"] == 0:
        return "{} full bathroom.".format(row["FullBath"])


    elif row["FullBath"] == 0 and row["HalfBath"] > 1:
        return "{} half bathrooms.".format(row["HalfBath"])
    elif row["FullBath"] == 0 and row["HalfBath"] == 1:
        return "{} half bathroom.".format(row["HalfBath"])

    else:
        return ""

def make_lot_area(row):

    templates = [
        "Lot area is {} square feet.",
        "Lot area is {} sq ft.",
        "Lot area is {} square meters.",
        "Lot area is {} sq m.",
    ]

    template = rn.choice(templates)

    if "m" in templates:
        return template.format(row["LotArea"] /  10.764)

    return template.format(row["LotArea"])

def make_overall_quality(row):

    num_to_quality = {

       10: "Very Excellent",
       9: "Excellent",
       8: "Very Good",
       7: "Good",
       6: "Above Average",
       5: "Average",
       4: "Below Average",
       3: "Fair",
       2: "Poor",
       1: "Very Poor",

    }

    num_to_quality = {k: v.lower() for k, v in num_to_quality.items()}

    template = "The overall material and finish of the house is {}."

    return template.format(num_to_quality[row["OverallQual"]])

def make_overall_condition(row):

    num_to_quality = {

       10: "Very Excellent",
       9: "Excellent",
       8: "Very Good",
       7: "Good",
       6: "Above Average",
       5: "Average",
       4: "Below Average",
       3: "Fair",
       2: "Poor",
       1: "Very Poor",

    }

    num_to_quality = {k: v.lower() for k, v in num_to_quality.items()}

    templates = ["The overall condition of the house is {}.",
                 "The house is in {} condition.",
    ]

    template = rn.choice(templates)

    return template.format(num_to_quality[row["OverallCond"]])

def join_sentences(row):

    rn.shuffle(row)

    return " ".join(row)



if __name__ == "__main__":

    df_train = pd.read_csv(ds.ORIGINAL_TRAIN_FILE_PATH)
    df_train_temp = pd.DataFrame()

    df_train_temp["date_sold"] = df_train.apply(make_prev_sale_date, axis=1)
    df_train_temp["sq_ft"] = df_train.apply(make_square_footage, axis=1)
    df_train_temp["bath_counts"] = df_train.apply(make_baths_counts, axis=1)
    df_train_temp["lot_area"] = df_train.apply(make_lot_area, axis=1)
    df_train_temp["overall_quality"] = df_train.apply(make_overall_quality, axis=1)
    df_train_temp["overall_condition"] = df_train.apply(make_overall_condition, axis=1)

    df_train_to_write = pd.DataFrame()

    df_train_to_write["Id"] = df_train["Id"]
    df_train_to_write["text"] = df_train_temp.apply(join_sentences, axis=1)
    df_train_to_write["target"] = df_train["SalePrice"]

    df_train_to_write.to_csv(ds.TRAIN_FILE_PATH, index=False)

    df_test = pd.read_csv(ds.ORIGINAL_TEST_FILE_PATH)
    df_test_temp = pd.DataFrame()

    df_test_temp["date_sold"] = df_test.apply(make_prev_sale_date, axis=1)
    df_test_temp["sq_ft"] = df_test.apply(make_square_footage, axis=1)
    df_test_temp["bath_counts"] = df_test.apply(make_baths_counts, axis=1)
    df_test_temp["lot_area"] = df_test.apply(make_lot_area, axis=1)
    df_test_temp["overall_quality"] = df_test.apply(make_overall_quality, axis=1)
    df_test_temp["overall_condition"] = df_test.apply(make_overall_condition, axis=1)

    df_test_to_write = pd.DataFrame()

    df_test_to_write["Id"] = df_test["Id"]
    df_test_to_write["text"] = df_test_temp.apply(join_sentences, axis=1)

    df_test_to_write.to_csv(ds.TEST_FILE_PATH, index=False)