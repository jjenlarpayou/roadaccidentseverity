import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, make_scorer, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold, train_test_split, cross_validate, RepeatedStratifiedKFold
import numpy as np
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTEN
from imblearn.under_sampling import RandomUnderSampler
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

def preparation_check():
    df_location = pd.read_csv('../../Dataset/Location.csv')
    Q1 = df_location['Crash_Longitude'].quantile(0.10)
    Q3 = df_location['Crash_Longitude'].quantile(0.90)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_location[(df_location['Crash_Longitude'] < lower_bound) | (df_location['Crash_Longitude'] > upper_bound)]
    df_no_outliers = df_location[(df_location['Crash_Longitude'] >= lower_bound) & (df_location['Crash_Longitude'] <= upper_bound)]
    Q1 = df_no_outliers['Crash_Latitude'].quantile(0.10)
    Q3 = df_no_outliers['Crash_Latitude'].quantile(0.90)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 *IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_no_outliers[(df_no_outliers['Crash_Latitude'] < lower_bound) | (df_no_outliers['Crash_Latitude'] > upper_bound)]
    df_no_outliers = df_no_outliers[(df_no_outliers['Crash_Latitude'] >= lower_bound) & (df_no_outliers['Crash_Latitude'] <= upper_bound)]
    df_filtered_2023 = df_no_outliers.copy()
    # df_filtered_2023['Crash_Severity'] = df_filtered_2023['Crash_Severity'].replace({'Hospitalisation': 'Serious injury'})
    # df_filtered_2023['Crash_Severity'] = df_filtered_2023['Crash_Severity'].replace({'Medical treatment': 'Serious injury'})
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Road_Surface_Condition']!='Unknown']
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Atmospheric_Condition']!='Unknown']
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Lighting_Condition']!='Unknown']
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Type']!='Other']
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Speed_Limit']!= None]
    # Removing < 0.01 theshold
    threshold = 0.01
    remove_frequency = []
    frequency = df_filtered_2023['Crash_Nature'].value_counts(normalize=True)
    remove_frequency.append(frequency[frequency < threshold])
    df_filtered_2023 = df_filtered_2023[~df_filtered_2023['Crash_Nature'].isin(frequency[frequency < threshold].index)]
    
    frequency = df_filtered_2023['Crash_Roadway_Feature'].value_counts(normalize=True)
    remove_frequency.append(frequency[frequency < threshold])
    df_filtered_2023 = df_filtered_2023[~df_filtered_2023['Crash_Roadway_Feature'].isin(frequency[frequency < threshold].index)]
    
    frequency = df_filtered_2023['Crash_Controlling_Authority'].value_counts(normalize=True)
    remove_frequency.append(frequency[frequency < threshold])
    df_filtered_2023 = df_filtered_2023[~df_filtered_2023['Crash_Controlling_Authority'].isin(frequency[frequency < threshold].index)]
    
    frequency = df_filtered_2023['Crash_Traffic_Control'].value_counts(normalize=True)
    remove_frequency.append(frequency[frequency < threshold])
    df_filtered_2023 = df_filtered_2023[~df_filtered_2023['Crash_Traffic_Control'].isin(frequency[frequency < threshold].index)]
    
    conditions = [
    df_filtered_2023['Crash_Month'].isin(['January', 'February', 'March']),
    df_filtered_2023['Crash_Month'].isin(['April', 'May', 'June']),
    df_filtered_2023['Crash_Month'].isin(['July', 'August', 'September']),
    df_filtered_2023['Crash_Month'].isin(['October', 'November', 'December'])]
    choices = ['Q1', 'Q2', 'Q3', 'Q4']
    df_filtered_2023['Crash_Month'] = np.select(conditions, choices, default='Unknown')
    df_filtered_2023 = df_filtered_2023.reset_index(drop=True)
            
    columns = df_filtered_2023[['Crash_Month','Crash_Nature','Crash_Type','Crash_Controlling_Authority','Crash_Roadway_Feature',
                          'Crash_Traffic_Control','Crash_Speed_Limit','Crash_Road_Surface_Condition','Crash_Atmospheric_Condition','Crash_Lighting_Condition',
                          'Crash_Road_Horiz_Align','Crash_Road_Vert_Align']]

    column_name = ['Crash_Month', 'Crash_Nature','Crash_Type','Crash_Controlling_Authority','Crash_Roadway_Feature',
                          'Crash_Traffic_Control','Crash_Speed_Limit','Crash_Road_Surface_Condition','Crash_Atmospheric_Condition','Crash_Lighting_Condition',
                          'Crash_Road_Horiz_Align','Crash_Road_Vert_Align']
    #Encoder
    X_encoder = OneHotEncoder(sparse_output=False)
    X_encoded = X_encoder.fit_transform(columns)

    X_column_names = X_encoder.get_feature_names_out(column_name)

    X = pd.DataFrame(X_encoded, columns=X_column_names)
    X = X.drop(columns='Crash_Speed_Limit_nan')
    X = X.drop(columns='Crash_Road_Vert_Align_nan')
    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(df_filtered_2023['Crash_Severity'])
    # y = df_filtered_2023['Crash_Severity']
    return X, y, df_filtered_2023, remove_frequency

def preparation():
    df_location = pd.read_csv('../../Dataset/Location.csv')
    Q1 = df_location['Crash_Longitude'].quantile(0.10)
    Q3 = df_location['Crash_Longitude'].quantile(0.90)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_location[(df_location['Crash_Longitude'] < lower_bound) | (df_location['Crash_Longitude'] > upper_bound)]
    df_no_outliers = df_location[(df_location['Crash_Longitude'] >= lower_bound) & (df_location['Crash_Longitude'] <= upper_bound)]
    Q1 = df_no_outliers['Crash_Latitude'].quantile(0.10)
    Q3 = df_no_outliers['Crash_Latitude'].quantile(0.90)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 *IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_no_outliers[(df_no_outliers['Crash_Latitude'] < lower_bound) | (df_no_outliers['Crash_Latitude'] > upper_bound)]
    df_no_outliers = df_no_outliers[(df_no_outliers['Crash_Latitude'] >= lower_bound) & (df_no_outliers['Crash_Latitude'] <= upper_bound)]
    df_filtered_2023 = df_no_outliers.copy()
    df_filtered_2023 = df_filtered_2023[(df_filtered_2023['Crash_Roadway_Feature']!='Other') & (df_filtered_2023['Crash_Roadway_Feature']!='Miscellaneous')]
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Road_Surface_Condition']!='Unknown']
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Atmospheric_Condition']!='Unknown']
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Lighting_Condition']!='Unknown']
    conditions = [
    df_filtered_2023['Crash_Month'].isin(['January', 'February', 'March']),
    df_filtered_2023['Crash_Month'].isin(['April', 'May', 'June']),
    df_filtered_2023['Crash_Month'].isin(['July', 'August', 'September']),
    df_filtered_2023['Crash_Month'].isin(['October', 'November', 'December'])]
    choices = ['Q1', 'Q2', 'Q3', 'Q4']
    df_filtered_2023['Crash_Month'] = np.select(conditions, choices, default='Unknown')
            
    columns = df_filtered_2023[['Crash_Month','Crash_Nature','Crash_Type','Crash_Controlling_Authority','Crash_Roadway_Feature',
                          'Crash_Traffic_Control','Crash_Speed_Limit','Crash_Road_Surface_Condition','Crash_Atmospheric_Condition','Crash_Lighting_Condition',
                          'Crash_Road_Horiz_Align','Crash_Road_Vert_Align']]

    column_name = ['Crash_Month', 'Crash_Nature','Crash_Type','Crash_Controlling_Authority','Crash_Roadway_Feature',
                          'Crash_Traffic_Control','Crash_Speed_Limit','Crash_Road_Surface_Condition','Crash_Atmospheric_Condition','Crash_Lighting_Condition',
                          'Crash_Road_Horiz_Align','Crash_Road_Vert_Align']
    #Encoder
    X_encoder = OneHotEncoder(sparse=False)
    X_encoded = X_encoder.fit_transform(columns)

    X_column_names = X_encoder.get_feature_names_out(column_name)

    X = pd.DataFrame(X_encoded, columns=X_column_names)

    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(df_filtered_2023[['Crash_Severity']])
    y = pd.DataFrame(y, columns=['Crash_Severity'])
    return X, y, df_filtered_2023[['Crash_Severity']], df_filtered_2023

def preparation_baseline():
    df_location = pd.read_csv('../../Dataset/Location.csv')
    Q1 = df_location['Crash_Longitude'].quantile(0.10)
    Q3 = df_location['Crash_Longitude'].quantile(0.90)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_location[(df_location['Crash_Longitude'] < lower_bound) | (df_location['Crash_Longitude'] > upper_bound)]
    df_no_outliers = df_location[(df_location['Crash_Longitude'] >= lower_bound) & (df_location['Crash_Longitude'] <= upper_bound)]
    Q1 = df_no_outliers['Crash_Latitude'].quantile(0.10)
    Q3 = df_no_outliers['Crash_Latitude'].quantile(0.90)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 *IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_no_outliers[(df_no_outliers['Crash_Latitude'] < lower_bound) | (df_no_outliers['Crash_Latitude'] > upper_bound)]
    df_no_outliers = df_no_outliers[(df_no_outliers['Crash_Latitude'] >= lower_bound) & (df_no_outliers['Crash_Latitude'] <= upper_bound)]
    df_filtered_2023 = df_no_outliers.copy()
    
    conditions = [
    df_filtered_2023['Crash_Month'].isin(['January', 'February', 'March']),
    df_filtered_2023['Crash_Month'].isin(['April', 'May', 'June']),
    df_filtered_2023['Crash_Month'].isin(['July', 'August', 'September']),
    df_filtered_2023['Crash_Month'].isin(['October', 'November', 'December'])]
    choices = ['Q1', 'Q2', 'Q3', 'Q4']
    df_filtered_2023['Crash_Month'] = np.select(conditions, choices, default='Unknown')
    df_filtered_2023 = df_filtered_2023.reset_index(drop=True)
            
    columns = df_filtered_2023[['Crash_Month','Crash_Nature','Crash_Type','Crash_Controlling_Authority','Crash_Roadway_Feature',
                          'Crash_Traffic_Control','Crash_Speed_Limit','Crash_Road_Surface_Condition','Crash_Atmospheric_Condition','Crash_Lighting_Condition',
                          'Crash_Road_Horiz_Align','Crash_Road_Vert_Align']]

    column_name = ['Crash_Month', 'Crash_Nature','Crash_Type','Crash_Controlling_Authority','Crash_Roadway_Feature',
                          'Crash_Traffic_Control','Crash_Speed_Limit','Crash_Road_Surface_Condition','Crash_Atmospheric_Condition','Crash_Lighting_Condition',
                          'Crash_Road_Horiz_Align','Crash_Road_Vert_Align']
    #Encoder
    X_encoder = OneHotEncoder(sparse_output=False)
    X_encoded = X_encoder.fit_transform(columns)

    X_column_names = X_encoder.get_feature_names_out(column_name)

    X = pd.DataFrame(X_encoded, columns=X_column_names)
    X = X.drop(columns=['Crash_Speed_Limit_nan', 'Crash_Road_Vert_Align_nan', 'Crash_Road_Surface_Condition_Unknown', 'Crash_Atmospheric_Condition_Unknown','Crash_Lighting_Condition_Unknown'])    
    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(df_filtered_2023['Crash_Severity'])
    return X, y, df_filtered_2023

def preparation_class_1():
    df_location = pd.read_csv('../../Dataset/Location.csv')
    Q1 = df_location['Crash_Longitude'].quantile(0.10)
    Q3 = df_location['Crash_Longitude'].quantile(0.90)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_location[(df_location['Crash_Longitude'] < lower_bound) | (df_location['Crash_Longitude'] > upper_bound)]
    df_no_outliers = df_location[(df_location['Crash_Longitude'] >= lower_bound) & (df_location['Crash_Longitude'] <= upper_bound)]
    Q1 = df_no_outliers['Crash_Latitude'].quantile(0.10)
    Q3 = df_no_outliers['Crash_Latitude'].quantile(0.90)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 *IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_no_outliers[(df_no_outliers['Crash_Latitude'] < lower_bound) | (df_no_outliers['Crash_Latitude'] > upper_bound)]
    df_no_outliers = df_no_outliers[(df_no_outliers['Crash_Latitude'] >= lower_bound) & (df_no_outliers['Crash_Latitude'] <= upper_bound)]
    df_filtered_2023 = df_no_outliers.copy()
    df_filtered_2023 = df_filtered_2023[~df_filtered_2023['Crash_Roadway_Feature'].isin(['Railway Crossing','Intersection - Multiple Road','Forestry/National Park Road','Intersection - Y-Junction','Bikeway','Intersection - 5+ way','Other'])]
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Road_Vert_Align']!='nan']
    df_filtered_2023 = df_filtered_2023[~df_filtered_2023['Crash_Traffic_Control'].isin(['Pedestrian operated lights', 'Road/Rail worker', 'Railway - lights only', 'Police', 'Railway - lights and boom gate', 'Flashing amber lights', 'LATM device', 'Railway crossing sign', 'Supervised school crossing', 'School crossing - flags','Miscellaneous'])]
    df_filtered_2023 = df_filtered_2023[~df_filtered_2023['Crash_Nature'].isin(['Non-collision - miscellaneous', 'Struck by external load', 'Other', 'Struck by internal load'])]
    df_filtered_2023 = df_filtered_2023[~df_filtered_2023['Crash_Controlling_Authority'].isin(['Other','Not coded'])]
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Road_Surface_Condition']!='Unknown']
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Atmospheric_Condition']!='Unknown']
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Lighting_Condition']!='Unknown']
    df_filtered_2023['Crash_Severity'] = df_filtered_2023['Crash_Severity'].replace({'Medical treatment': 'Minor injury'})
    df_filtered_2023['Crash_Severity'] = df_filtered_2023['Crash_Severity'].replace({'Hospitalisation': 'Serious injury','Fatal': 'Serious injury'})
    conditions = [
    df_filtered_2023['Crash_Month'].isin(['January', 'February', 'March']),
    df_filtered_2023['Crash_Month'].isin(['April', 'May', 'June']),
    df_filtered_2023['Crash_Month'].isin(['July', 'August', 'September']),
    df_filtered_2023['Crash_Month'].isin(['October', 'November', 'December'])]
    choices = ['Q1', 'Q2', 'Q3', 'Q4']
    df_filtered_2023['Crash_Month'] = np.select(conditions, choices, default='Unknown')
            
    columns = df_filtered_2023[['Crash_Month','Crash_Nature','Crash_Type','Crash_Controlling_Authority','Crash_Roadway_Feature',
                          'Crash_Traffic_Control','Crash_Speed_Limit','Crash_Road_Surface_Condition','Crash_Atmospheric_Condition','Crash_Lighting_Condition',
                          'Crash_Road_Horiz_Align','Crash_Road_Vert_Align']]

    column_name = ['Crash_Month', 'Crash_Nature','Crash_Type','Crash_Controlling_Authority','Crash_Roadway_Feature',
                          'Crash_Traffic_Control','Crash_Speed_Limit','Crash_Road_Surface_Condition','Crash_Atmospheric_Condition','Crash_Lighting_Condition',
                          'Crash_Road_Horiz_Align','Crash_Road_Vert_Align']
    #Encoder
    X_encoder = OneHotEncoder(sparse=False)
    X_encoded = X_encoder.fit_transform(columns)

    X_column_names = X_encoder.get_feature_names_out(column_name)

    X = pd.DataFrame(X_encoded, columns=X_column_names)

    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(df_filtered_2023[['Crash_Severity']])
    y = pd.DataFrame(y, columns=['Crash_Severity'])
    return X, y, df_filtered_2023[['Crash_Severity']], df_filtered_2023

def preparation_class_2():
    df_location = pd.read_csv('../../Dataset/Location.csv')
    Q1 = df_location['Crash_Longitude'].quantile(0.10)
    Q3 = df_location['Crash_Longitude'].quantile(0.90)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_location[(df_location['Crash_Longitude'] < lower_bound) | (df_location['Crash_Longitude'] > upper_bound)]
    df_no_outliers = df_location[(df_location['Crash_Longitude'] >= lower_bound) & (df_location['Crash_Longitude'] <= upper_bound)]
    Q1 = df_no_outliers['Crash_Latitude'].quantile(0.10)
    Q3 = df_no_outliers['Crash_Latitude'].quantile(0.90)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 *IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_no_outliers[(df_no_outliers['Crash_Latitude'] < lower_bound) | (df_no_outliers['Crash_Latitude'] > upper_bound)]
    df_no_outliers = df_no_outliers[(df_no_outliers['Crash_Latitude'] >= lower_bound) & (df_no_outliers['Crash_Latitude'] <= upper_bound)]
    df_filtered_2023 = df_no_outliers.copy()
    df_filtered_2023 = df_filtered_2023[(df_filtered_2023['Crash_Roadway_Feature']!='Other') & (df_filtered_2023['Crash_Roadway_Feature']!='Miscellaneous')]
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Road_Surface_Condition']!='Unknown']
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Atmospheric_Condition']!='Unknown']
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Lighting_Condition']!='Unknown']
    df_filtered_2023['Crash_Severity'] = df_filtered_2023['Crash_Severity'].replace({'Medical treatment': 'Minor injury'})
    conditions = [
    df_filtered_2023['Crash_Month'].isin(['January', 'February', 'March']),
    df_filtered_2023['Crash_Month'].isin(['April', 'May', 'June']),
    df_filtered_2023['Crash_Month'].isin(['July', 'August', 'September']),
    df_filtered_2023['Crash_Month'].isin(['October', 'November', 'December'])]
    choices = ['Q1', 'Q2', 'Q3', 'Q4']
    df_filtered_2023['Crash_Month'] = np.select(conditions, choices, default='Unknown')
            
    columns = df_filtered_2023[['Crash_Month','Crash_Nature','Crash_Type','Crash_Controlling_Authority','Crash_Roadway_Feature',
                          'Crash_Traffic_Control','Crash_Speed_Limit','Crash_Road_Surface_Condition','Crash_Atmospheric_Condition','Crash_Lighting_Condition',
                          'Crash_Road_Horiz_Align','Crash_Road_Vert_Align']]

    column_name = ['Crash_Month', 'Crash_Nature','Crash_Type','Crash_Controlling_Authority','Crash_Roadway_Feature',
                          'Crash_Traffic_Control','Crash_Speed_Limit','Crash_Road_Surface_Condition','Crash_Atmospheric_Condition','Crash_Lighting_Condition',
                          'Crash_Road_Horiz_Align','Crash_Road_Vert_Align']
    #Encoder
    X_encoder = OneHotEncoder(sparse=False)
    X_encoded = X_encoder.fit_transform(columns)

    X_column_names = X_encoder.get_feature_names_out(column_name)

    X = pd.DataFrame(X_encoded, columns=X_column_names)

    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(df_filtered_2023[['Crash_Severity']])
    y = pd.DataFrame(y, columns=['Crash_Severity'])
    return X, y, df_filtered_2023[['Crash_Severity']], df_filtered_2023

def preparation_class_3():
    df_location = pd.read_csv('../../Dataset/Location.csv')
    Q1 = df_location['Crash_Longitude'].quantile(0.10)
    Q3 = df_location['Crash_Longitude'].quantile(0.90)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_location[(df_location['Crash_Longitude'] < lower_bound) | (df_location['Crash_Longitude'] > upper_bound)]
    df_no_outliers = df_location[(df_location['Crash_Longitude'] >= lower_bound) & (df_location['Crash_Longitude'] <= upper_bound)]
    Q1 = df_no_outliers['Crash_Latitude'].quantile(0.10)
    Q3 = df_no_outliers['Crash_Latitude'].quantile(0.90)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 *IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_no_outliers[(df_no_outliers['Crash_Latitude'] < lower_bound) | (df_no_outliers['Crash_Latitude'] > upper_bound)]
    df_no_outliers = df_no_outliers[(df_no_outliers['Crash_Latitude'] >= lower_bound) & (df_no_outliers['Crash_Latitude'] <= upper_bound)]
    df_filtered_2023 = df_no_outliers.copy()
    df_filtered_2023 = df_filtered_2023[(df_filtered_2023['Crash_Roadway_Feature']!='Other') & (df_filtered_2023['Crash_Roadway_Feature']!='Miscellaneous')]
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Road_Surface_Condition']!='Unknown']
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Atmospheric_Condition']!='Unknown']
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Lighting_Condition']!='Unknown']
    df_filtered_2023['Crash_Severity'] = df_filtered_2023['Crash_Severity'].replace({'Hospitalisation': 'Serious injury','Fatal': 'Serious injury'})
    df_filtered_2023['Crash_Severity'] = df_filtered_2023['Crash_Severity'].replace({'Medical treatment': 'Minor injury'})
    conditions = [
    df_filtered_2023['Crash_Month'].isin(['January', 'February', 'March']),
    df_filtered_2023['Crash_Month'].isin(['April', 'May', 'June']),
    df_filtered_2023['Crash_Month'].isin(['July', 'August', 'September']),
    df_filtered_2023['Crash_Month'].isin(['October', 'November', 'December'])]
    choices = ['Q1', 'Q2', 'Q3', 'Q4']
    df_filtered_2023['Crash_Month'] = np.select(conditions, choices, default='Unknown')
            
    columns = df_filtered_2023[['Crash_Month','Crash_Nature','Crash_Type','Crash_Controlling_Authority','Crash_Roadway_Feature',
                          'Crash_Traffic_Control','Crash_Speed_Limit','Crash_Road_Surface_Condition','Crash_Atmospheric_Condition','Crash_Lighting_Condition',
                          'Crash_Road_Horiz_Align','Crash_Road_Vert_Align']]

    column_name = ['Crash_Month', 'Crash_Nature','Crash_Type','Crash_Controlling_Authority','Crash_Roadway_Feature',
                          'Crash_Traffic_Control','Crash_Speed_Limit','Crash_Road_Surface_Condition','Crash_Atmospheric_Condition','Crash_Lighting_Condition',
                          'Crash_Road_Horiz_Align','Crash_Road_Vert_Align']
    #Encoder
    X_encoder = OneHotEncoder(sparse=False)
    X_encoded = X_encoder.fit_transform(columns)

    X_column_names = X_encoder.get_feature_names_out(column_name)

    X = pd.DataFrame(X_encoded, columns=X_column_names)

    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(df_filtered_2023[['Crash_Severity']])
    y = pd.DataFrame(y, columns=['Crash_Severity'])
    return X, y, df_filtered_2023[['Crash_Severity']], df_filtered_2023

def preparation_class_4():
    df_location = pd.read_csv('../../Dataset/Location.csv')
    Q1 = df_location['Crash_Longitude'].quantile(0.10)
    Q3 = df_location['Crash_Longitude'].quantile(0.90)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_location[(df_location['Crash_Longitude'] < lower_bound) | (df_location['Crash_Longitude'] > upper_bound)]
    df_no_outliers = df_location[(df_location['Crash_Longitude'] >= lower_bound) & (df_location['Crash_Longitude'] <= upper_bound)]
    Q1 = df_no_outliers['Crash_Latitude'].quantile(0.10)
    Q3 = df_no_outliers['Crash_Latitude'].quantile(0.90)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 *IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_no_outliers[(df_no_outliers['Crash_Latitude'] < lower_bound) | (df_no_outliers['Crash_Latitude'] > upper_bound)]
    df_no_outliers = df_no_outliers[(df_no_outliers['Crash_Latitude'] >= lower_bound) & (df_no_outliers['Crash_Latitude'] <= upper_bound)]
    df_filtered_2023 = df_no_outliers.copy()
    df_filtered_2023 = df_filtered_2023[(df_filtered_2023['Crash_Roadway_Feature']!='Other') & (df_filtered_2023['Crash_Roadway_Feature']!='Miscellaneous')]
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Road_Surface_Condition']!='Unknown']
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Atmospheric_Condition']!='Unknown']
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Lighting_Condition']!='Unknown']
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Severity']!='Property damage only']
    conditions = [
    df_filtered_2023['Crash_Month'].isin(['January', 'February', 'March']),
    df_filtered_2023['Crash_Month'].isin(['April', 'May', 'June']),
    df_filtered_2023['Crash_Month'].isin(['July', 'August', 'September']),
    df_filtered_2023['Crash_Month'].isin(['October', 'November', 'December'])]
    choices = ['Q1', 'Q2', 'Q3', 'Q4']
    df_filtered_2023['Crash_Month'] = np.select(conditions, choices, default='Unknown')
            
    columns = df_filtered_2023[['Crash_Month','Crash_Nature','Crash_Type','Crash_Controlling_Authority','Crash_Roadway_Feature',
                          'Crash_Traffic_Control','Crash_Speed_Limit','Crash_Road_Surface_Condition','Crash_Atmospheric_Condition','Crash_Lighting_Condition',
                          'Crash_Road_Horiz_Align','Crash_Road_Vert_Align']]

    column_name = ['Crash_Month', 'Crash_Nature','Crash_Type','Crash_Controlling_Authority','Crash_Roadway_Feature',
                          'Crash_Traffic_Control','Crash_Speed_Limit','Crash_Road_Surface_Condition','Crash_Atmospheric_Condition','Crash_Lighting_Condition',
                          'Crash_Road_Horiz_Align','Crash_Road_Vert_Align']
    #Encoder
    X_encoder = OneHotEncoder(sparse=False)
    X_encoded = X_encoder.fit_transform(columns)

    X_column_names = X_encoder.get_feature_names_out(column_name)

    X = pd.DataFrame(X_encoded, columns=X_column_names)

    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(df_filtered_2023[['Crash_Severity']])
    y = pd.DataFrame(y, columns=['Crash_Severity'])
    return X, y, df_filtered_2023[['Crash_Severity']], df_filtered_2023

def preparation_class_5():
    df_location = pd.read_csv('../../Dataset/Location.csv')
    Q1 = df_location['Crash_Longitude'].quantile(0.10)
    Q3 = df_location['Crash_Longitude'].quantile(0.90)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_location[(df_location['Crash_Longitude'] < lower_bound) | (df_location['Crash_Longitude'] > upper_bound)]
    df_no_outliers = df_location[(df_location['Crash_Longitude'] >= lower_bound) & (df_location['Crash_Longitude'] <= upper_bound)]
    Q1 = df_no_outliers['Crash_Latitude'].quantile(0.10)
    Q3 = df_no_outliers['Crash_Latitude'].quantile(0.90)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 *IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_no_outliers[(df_no_outliers['Crash_Latitude'] < lower_bound) | (df_no_outliers['Crash_Latitude'] > upper_bound)]
    df_no_outliers = df_no_outliers[(df_no_outliers['Crash_Latitude'] >= lower_bound) & (df_no_outliers['Crash_Latitude'] <= upper_bound)]
    df_filtered_2023 = df_no_outliers.copy()
    df_filtered_2023 = df_filtered_2023[(df_filtered_2023['Crash_Roadway_Feature']!='Other') & (df_filtered_2023['Crash_Roadway_Feature']!='Miscellaneous')]
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Road_Surface_Condition']!='Unknown']
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Atmospheric_Condition']!='Unknown']
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Lighting_Condition']!='Unknown']
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Severity']!='Property damage only']
    df_filtered_2023['Crash_Severity'] = df_filtered_2023['Crash_Severity'].replace({'Hospitalisation': 'Serious injury','Fatal': 'Serious injury'})
    conditions = [
    df_filtered_2023['Crash_Month'].isin(['January', 'February', 'March']),
    df_filtered_2023['Crash_Month'].isin(['April', 'May', 'June']),
    df_filtered_2023['Crash_Month'].isin(['July', 'August', 'September']),
    df_filtered_2023['Crash_Month'].isin(['October', 'November', 'December'])]
    choices = ['Q1', 'Q2', 'Q3', 'Q4']
    df_filtered_2023['Crash_Month'] = np.select(conditions, choices, default='Unknown')
            
    columns = df_filtered_2023[['Crash_Month','Crash_Nature','Crash_Type','Crash_Controlling_Authority','Crash_Roadway_Feature',
                          'Crash_Traffic_Control','Crash_Speed_Limit','Crash_Road_Surface_Condition','Crash_Atmospheric_Condition','Crash_Lighting_Condition',
                          'Crash_Road_Horiz_Align','Crash_Road_Vert_Align']]

    column_name = ['Crash_Month', 'Crash_Nature','Crash_Type','Crash_Controlling_Authority','Crash_Roadway_Feature',
                          'Crash_Traffic_Control','Crash_Speed_Limit','Crash_Road_Surface_Condition','Crash_Atmospheric_Condition','Crash_Lighting_Condition',
                          'Crash_Road_Horiz_Align','Crash_Road_Vert_Align']
    #Encoder
    X_encoder = OneHotEncoder(sparse_output=False)
    X_encoded = X_encoder.fit_transform(columns)

    X_column_names = X_encoder.get_feature_names_out(column_name)

    X = pd.DataFrame(X_encoded, columns=X_column_names)

    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(df_filtered_2023[['Crash_Severity']])
    y = pd.DataFrame(y, columns=['Crash_Severity'])
    return X, y, df_filtered_2023[['Crash_Severity']], df_filtered_2023

def preparation_class_6():
    df_location = pd.read_csv('../../Dataset/Location.csv')
    Q1 = df_location['Crash_Longitude'].quantile(0.10)
    Q3 = df_location['Crash_Longitude'].quantile(0.90)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_location[(df_location['Crash_Longitude'] < lower_bound) | (df_location['Crash_Longitude'] > upper_bound)]
    df_no_outliers = df_location[(df_location['Crash_Longitude'] >= lower_bound) & (df_location['Crash_Longitude'] <= upper_bound)]
    Q1 = df_no_outliers['Crash_Latitude'].quantile(0.10)
    Q3 = df_no_outliers['Crash_Latitude'].quantile(0.90)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 *IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_no_outliers[(df_no_outliers['Crash_Latitude'] < lower_bound) | (df_no_outliers['Crash_Latitude'] > upper_bound)]
    df_no_outliers = df_no_outliers[(df_no_outliers['Crash_Latitude'] >= lower_bound) & (df_no_outliers['Crash_Latitude'] <= upper_bound)]
    df_filtered_2023 = df_no_outliers.copy()
    df_filtered_2023 = df_filtered_2023[(df_filtered_2023['Crash_Roadway_Feature']!='Other') & (df_filtered_2023['Crash_Roadway_Feature']!='Miscellaneous')]
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Road_Surface_Condition']!='Unknown']
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Atmospheric_Condition']!='Unknown']
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Lighting_Condition']!='Unknown']
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Severity']!='Property damage only']
    df_filtered_2023['Crash_Severity'] = df_filtered_2023['Crash_Severity'].replace({'Medical treatment': 'Minor injury'})
    conditions = [
    df_filtered_2023['Crash_Month'].isin(['January', 'February', 'March']),
    df_filtered_2023['Crash_Month'].isin(['April', 'May', 'June']),
    df_filtered_2023['Crash_Month'].isin(['July', 'August', 'September']),
    df_filtered_2023['Crash_Month'].isin(['October', 'November', 'December'])]
    choices = ['Q1', 'Q2', 'Q3', 'Q4']
    df_filtered_2023['Crash_Month'] = np.select(conditions, choices, default='Unknown')
            
    columns = df_filtered_2023[['Crash_Month','Crash_Nature','Crash_Type','Crash_Controlling_Authority','Crash_Roadway_Feature',
                          'Crash_Traffic_Control','Crash_Speed_Limit','Crash_Road_Surface_Condition','Crash_Atmospheric_Condition','Crash_Lighting_Condition',
                          'Crash_Road_Horiz_Align','Crash_Road_Vert_Align']]

    column_name = ['Crash_Month', 'Crash_Nature','Crash_Type','Crash_Controlling_Authority','Crash_Roadway_Feature',
                          'Crash_Traffic_Control','Crash_Speed_Limit','Crash_Road_Surface_Condition','Crash_Atmospheric_Condition','Crash_Lighting_Condition',
                          'Crash_Road_Horiz_Align','Crash_Road_Vert_Align']
    #Encoder
    X_encoder = OneHotEncoder(sparse=False)
    X_encoded = X_encoder.fit_transform(columns)

    X_column_names = X_encoder.get_feature_names_out(column_name)

    X = pd.DataFrame(X_encoded, columns=X_column_names)

    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(df_filtered_2023[['Crash_Severity']])
    y = pd.DataFrame(y, columns=['Crash_Severity'])
    return X, y, df_filtered_2023[['Crash_Severity']], df_filtered_2023

def preparation_class_7():
    df_location = pd.read_csv('../../Dataset/Location.csv')
    Q1 = df_location['Crash_Longitude'].quantile(0.10)
    Q3 = df_location['Crash_Longitude'].quantile(0.90)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_location[(df_location['Crash_Longitude'] < lower_bound) | (df_location['Crash_Longitude'] > upper_bound)]
    df_no_outliers = df_location[(df_location['Crash_Longitude'] >= lower_bound) & (df_location['Crash_Longitude'] <= upper_bound)]
    Q1 = df_no_outliers['Crash_Latitude'].quantile(0.10)
    Q3 = df_no_outliers['Crash_Latitude'].quantile(0.90)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 *IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_no_outliers[(df_no_outliers['Crash_Latitude'] < lower_bound) | (df_no_outliers['Crash_Latitude'] > upper_bound)]
    df_no_outliers = df_no_outliers[(df_no_outliers['Crash_Latitude'] >= lower_bound) & (df_no_outliers['Crash_Latitude'] <= upper_bound)]
    df_filtered_2023 = df_no_outliers.copy()
    df_filtered_2023 = df_filtered_2023[(df_filtered_2023['Crash_Roadway_Feature']!='Other') & (df_filtered_2023['Crash_Roadway_Feature']!='Miscellaneous')]
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Road_Surface_Condition']!='Unknown']
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Atmospheric_Condition']!='Unknown']
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Lighting_Condition']!='Unknown']
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Severity']!='Property damage only']
    df_filtered_2023['Crash_Severity'] = df_filtered_2023['Crash_Severity'].replace({'Hospitalisation': 'Serious injury'})
    df_filtered_2023['Crash_Severity'] = df_filtered_2023['Crash_Severity'].replace({'Medical treatment': 'Serious injury'})
    conditions = [
    df_filtered_2023['Crash_Month'].isin(['January', 'February', 'March']),
    df_filtered_2023['Crash_Month'].isin(['April', 'May', 'June']),
    df_filtered_2023['Crash_Month'].isin(['July', 'August', 'September']),
    df_filtered_2023['Crash_Month'].isin(['October', 'November', 'December'])]
    choices = ['Q1', 'Q2', 'Q3', 'Q4']
    df_filtered_2023['Crash_Month'] = np.select(conditions, choices, default='Unknown')
            
    columns = df_filtered_2023[['Crash_Month','Crash_Nature','Crash_Type','Crash_Controlling_Authority','Crash_Roadway_Feature',
                          'Crash_Traffic_Control','Crash_Speed_Limit','Crash_Road_Surface_Condition','Crash_Atmospheric_Condition','Crash_Lighting_Condition',
                          'Crash_Road_Horiz_Align','Crash_Road_Vert_Align']]

    column_name = ['Crash_Month', 'Crash_Nature','Crash_Type','Crash_Controlling_Authority','Crash_Roadway_Feature',
                          'Crash_Traffic_Control','Crash_Speed_Limit','Crash_Road_Surface_Condition','Crash_Atmospheric_Condition','Crash_Lighting_Condition',
                          'Crash_Road_Horiz_Align','Crash_Road_Vert_Align']
    #Encoder
    X_encoder = OneHotEncoder(sparse_output=False)
    X_encoded = X_encoder.fit_transform(columns)

    X_column_names = X_encoder.get_feature_names_out(column_name)

    X = pd.DataFrame(X_encoded, columns=X_column_names)

    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(df_filtered_2023[['Crash_Severity']])
    y = pd.DataFrame(y, columns=['Crash_Severity'])
    return X, y, df_filtered_2023[['Crash_Severity']], df_filtered_2023

def preparation_class_8():
    df_location = pd.read_csv('../../Dataset/Location.csv')
    Q1 = df_location['Crash_Longitude'].quantile(0.10)
    Q3 = df_location['Crash_Longitude'].quantile(0.90)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_location[(df_location['Crash_Longitude'] < lower_bound) | (df_location['Crash_Longitude'] > upper_bound)]
    df_no_outliers = df_location[(df_location['Crash_Longitude'] >= lower_bound) & (df_location['Crash_Longitude'] <= upper_bound)]
    Q1 = df_no_outliers['Crash_Latitude'].quantile(0.10)
    Q3 = df_no_outliers['Crash_Latitude'].quantile(0.90)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 *IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_no_outliers[(df_no_outliers['Crash_Latitude'] < lower_bound) | (df_no_outliers['Crash_Latitude'] > upper_bound)]
    df_no_outliers = df_no_outliers[(df_no_outliers['Crash_Latitude'] >= lower_bound) & (df_no_outliers['Crash_Latitude'] <= upper_bound)]
    df_filtered_2023 = df_no_outliers.copy()
    df_filtered_2023 = df_filtered_2023[~df_filtered_2023['Crash_Roadway_Feature'].isin(['Railway Crossing','Intersection - Multiple Road','Forestry/National Park Road','Intersection - Y-Junction','Bikeway','Intersection - 5+ way','Other'])]
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Road_Vert_Align']!='nan']
    df_filtered_2023 = df_filtered_2023[~df_filtered_2023['Crash_Traffic_Control'].isin(['Pedestrian operated lights', 'Road/Rail worker', 'Railway - lights only', 'Police', 'Railway - lights and boom gate', 'Flashing amber lights', 'LATM device', 'Railway crossing sign', 'Supervised school crossing', 'School crossing - flags','Miscellaneous'])]
    df_filtered_2023 = df_filtered_2023[~df_filtered_2023['Crash_Nature'].isin(['Non-collision - miscellaneous', 'Struck by external load', 'Other', 'Struck by internal load'])]
    df_filtered_2023 = df_filtered_2023[~df_filtered_2023['Crash_Controlling_Authority'].isin(['Other','Not coded'])]
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Road_Surface_Condition']!='Unknown']
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Atmospheric_Condition']!='Unknown']
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Lighting_Condition']!='Unknown']
    df_filtered_2023['Crash_Severity'] = df_filtered_2023['Crash_Severity'].replace({'Hospitalisation': 'Serious injury'})
    df_filtered_2023['Crash_Severity'] = df_filtered_2023['Crash_Severity'].replace({'Medical treatment': 'Serious injury'})
    conditions = [
    df_filtered_2023['Crash_Month'].isin(['January', 'February', 'March']),
    df_filtered_2023['Crash_Month'].isin(['April', 'May', 'June']),
    df_filtered_2023['Crash_Month'].isin(['July', 'August', 'September']),
    df_filtered_2023['Crash_Month'].isin(['October', 'November', 'December'])]
    choices = ['Q1', 'Q2', 'Q3', 'Q4']
    df_filtered_2023['Crash_Month'] = np.select(conditions, choices, default='Unknown')
            
    columns = df_filtered_2023[['Crash_Month','Crash_Nature','Crash_Type','Crash_Controlling_Authority','Crash_Roadway_Feature',
                          'Crash_Traffic_Control','Crash_Speed_Limit','Crash_Road_Surface_Condition','Crash_Atmospheric_Condition','Crash_Lighting_Condition',
                          'Crash_Road_Horiz_Align','Crash_Road_Vert_Align']]

    column_name = ['Crash_Month', 'Crash_Nature','Crash_Type','Crash_Controlling_Authority','Crash_Roadway_Feature',
                          'Crash_Traffic_Control','Crash_Speed_Limit','Crash_Road_Surface_Condition','Crash_Atmospheric_Condition','Crash_Lighting_Condition',
                          'Crash_Road_Horiz_Align','Crash_Road_Vert_Align']
    #Encoder
    X_encoder = OneHotEncoder(sparse_output=False)
    X_encoded = X_encoder.fit_transform(columns)

    X_column_names = X_encoder.get_feature_names_out(column_name)

    X = pd.DataFrame(X_encoded, columns=X_column_names)

    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(df_filtered_2023['Crash_Severity'])
    return X, y, df_filtered_2023[['Crash_Severity']], df_filtered_2023

def preparation_class_outlier():
    df_location = pd.read_csv('../../Dataset/Location.csv')
    Q1 = df_location['Crash_Longitude'].quantile(0.10)
    Q3 = df_location['Crash_Longitude'].quantile(0.90)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_location[(df_location['Crash_Longitude'] < lower_bound) | (df_location['Crash_Longitude'] > upper_bound)]
    df_no_outliers = df_location[(df_location['Crash_Longitude'] >= lower_bound) & (df_location['Crash_Longitude'] <= upper_bound)]
    Q1 = df_no_outliers['Crash_Latitude'].quantile(0.10)
    Q3 = df_no_outliers['Crash_Latitude'].quantile(0.90)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 *IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_no_outliers[(df_no_outliers['Crash_Latitude'] < lower_bound) | (df_no_outliers['Crash_Latitude'] > upper_bound)]
    df_no_outliers = df_no_outliers[(df_no_outliers['Crash_Latitude'] >= lower_bound) & (df_no_outliers['Crash_Latitude'] <= upper_bound)]
    df_filtered_2023 = df_no_outliers.copy()
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Road_Surface_Condition']!='Unknown']
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Atmospheric_Condition']!='Unknown']
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Lighting_Condition']!='Unknown']
    df_filtered_2023['Crash_Severity'] = df_filtered_2023['Crash_Severity'].replace({'Hospitalisation': 'Serious injury'})
    df_filtered_2023['Crash_Severity'] = df_filtered_2023['Crash_Severity'].replace({'Medical treatment': 'Serious injury'})
    conditions = [
    df_filtered_2023['Crash_Month'].isin(['January', 'February', 'March']),
    df_filtered_2023['Crash_Month'].isin(['April', 'May', 'June']),
    df_filtered_2023['Crash_Month'].isin(['July', 'August', 'September']),
    df_filtered_2023['Crash_Month'].isin(['October', 'November', 'December'])]
    choices = ['Q1', 'Q2', 'Q3', 'Q4']
    df_filtered_2023['Crash_Month'] = np.select(conditions, choices, default='Unknown')
            
    columns = df_filtered_2023[['Crash_Month','Crash_Nature','Crash_Type','Crash_Controlling_Authority','Crash_Roadway_Feature',
                          'Crash_Traffic_Control','Crash_Speed_Limit','Crash_Road_Surface_Condition','Crash_Atmospheric_Condition','Crash_Lighting_Condition',
                          'Crash_Road_Horiz_Align','Crash_Road_Vert_Align']]

    column_name = ['Crash_Month', 'Crash_Nature','Crash_Type','Crash_Controlling_Authority','Crash_Roadway_Feature',
                          'Crash_Traffic_Control','Crash_Speed_Limit','Crash_Road_Surface_Condition','Crash_Atmospheric_Condition','Crash_Lighting_Condition',
                          'Crash_Road_Horiz_Align','Crash_Road_Vert_Align']
    #Encoder
    X_encoder = OneHotEncoder(sparse_output=False)
    X_encoded = X_encoder.fit_transform(columns)

    X_column_names = X_encoder.get_feature_names_out(column_name)

    X = pd.DataFrame(X_encoded, columns=X_column_names)

    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(df_filtered_2023['Crash_Severity'])
    ones_count = (X == 1).sum()
    rare_columns = [column for column, count in ones_count.items() if count < 300]
    X = X.drop(columns=rare_columns)
    return X, y, df_filtered_2023[['Crash_Severity']], df_filtered_2023, rare_columns

def preparation_class_duplicates():
    df_location = pd.read_csv('../../Dataset/Location.csv')
    Q1 = df_location['Crash_Longitude'].quantile(0.10)
    Q3 = df_location['Crash_Longitude'].quantile(0.90)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_location[(df_location['Crash_Longitude'] < lower_bound) | (df_location['Crash_Longitude'] > upper_bound)]
    df_no_outliers = df_location[(df_location['Crash_Longitude'] >= lower_bound) & (df_location['Crash_Longitude'] <= upper_bound)]
    Q1 = df_no_outliers['Crash_Latitude'].quantile(0.10)
    Q3 = df_no_outliers['Crash_Latitude'].quantile(0.90)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 *IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_no_outliers[(df_no_outliers['Crash_Latitude'] < lower_bound) | (df_no_outliers['Crash_Latitude'] > upper_bound)]
    df_no_outliers = df_no_outliers[(df_no_outliers['Crash_Latitude'] >= lower_bound) & (df_no_outliers['Crash_Latitude'] <= upper_bound)]
    df_filtered_2023 = df_no_outliers.copy()
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Road_Surface_Condition']!='Unknown']
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Atmospheric_Condition']!='Unknown']
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Lighting_Condition']!='Unknown']
    df_filtered_2023['Crash_Severity'] = df_filtered_2023['Crash_Severity'].replace({'Hospitalisation': 'Serious injury'})
    df_filtered_2023['Crash_Severity'] = df_filtered_2023['Crash_Severity'].replace({'Medical treatment': 'Serious injury'})
    conditions = [
    df_filtered_2023['Crash_Month'].isin(['January', 'February', 'March']),
    df_filtered_2023['Crash_Month'].isin(['April', 'May', 'June']),
    df_filtered_2023['Crash_Month'].isin(['July', 'August', 'September']),
    df_filtered_2023['Crash_Month'].isin(['October', 'November', 'December'])]
    choices = ['Q1', 'Q2', 'Q3', 'Q4']
    df_filtered_2023['Crash_Month'] = np.select(conditions, choices, default='Unknown')
            
    columns = df_filtered_2023[['Crash_Month','Crash_Nature','Crash_Type','Crash_Controlling_Authority','Crash_Roadway_Feature',
                          'Crash_Traffic_Control','Crash_Speed_Limit','Crash_Road_Surface_Condition','Crash_Atmospheric_Condition','Crash_Lighting_Condition',
                          'Crash_Road_Horiz_Align','Crash_Road_Vert_Align']]

    column_name = ['Crash_Month', 'Crash_Nature','Crash_Type','Crash_Controlling_Authority','Crash_Roadway_Feature',
                          'Crash_Traffic_Control','Crash_Speed_Limit','Crash_Road_Surface_Condition','Crash_Atmospheric_Condition','Crash_Lighting_Condition',
                          'Crash_Road_Horiz_Align','Crash_Road_Vert_Align']
    #Encoder
    X_encoder = OneHotEncoder(sparse_output=False)
    X_encoded = X_encoder.fit_transform(columns)

    X_column_names = X_encoder.get_feature_names_out(column_name)

    X = pd.DataFrame(X_encoded, columns=X_column_names)

    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(df_filtered_2023['Crash_Severity'])
    # Duplicates
    X['Crash_Severity'] = y
    df = X.reset_index(drop=True)
    duplicates = df.duplicated()
    print(f"Number of duplicate rows: {duplicates.sum()}")
    df_cleaned = df.drop_duplicates()
    print(f"Data shape after removing duplicates: {df_cleaned.shape}")
    X, y = df_cleaned.reset_index(drop=True).drop("Crash_Severity", axis=1), df_cleaned['Crash_Severity'].values
    
    ones_count = (X == 1).sum()
    rare_columns = [column for column, count in ones_count.items() if count < 300]
    X = X.drop(columns=rare_columns)
    return X, y, df_filtered_2023[['Crash_Severity']], df_filtered_2023

def preparation_class_threshold():
    df_location = pd.read_csv('../../Dataset/Location.csv')
    Q1 = df_location['Crash_Longitude'].quantile(0.10)
    Q3 = df_location['Crash_Longitude'].quantile(0.90)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_location[(df_location['Crash_Longitude'] < lower_bound) | (df_location['Crash_Longitude'] > upper_bound)]
    df_no_outliers = df_location[(df_location['Crash_Longitude'] >= lower_bound) & (df_location['Crash_Longitude'] <= upper_bound)]
    Q1 = df_no_outliers['Crash_Latitude'].quantile(0.10)
    Q3 = df_no_outliers['Crash_Latitude'].quantile(0.90)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 *IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_no_outliers[(df_no_outliers['Crash_Latitude'] < lower_bound) | (df_no_outliers['Crash_Latitude'] > upper_bound)]
    df_no_outliers = df_no_outliers[(df_no_outliers['Crash_Latitude'] >= lower_bound) & (df_no_outliers['Crash_Latitude'] <= upper_bound)]
    df_filtered_2023 = df_no_outliers.copy()
    df_filtered_2023['Crash_Severity'] = df_filtered_2023['Crash_Severity'].replace({'Hospitalisation': 'Serious injury'})
    df_filtered_2023['Crash_Severity'] = df_filtered_2023['Crash_Severity'].replace({'Medical treatment': 'Serious injury'})
    
    conditions = [
    df_filtered_2023['Crash_Month'].isin(['January', 'February', 'March']),
    df_filtered_2023['Crash_Month'].isin(['April', 'May', 'June']),
    df_filtered_2023['Crash_Month'].isin(['July', 'August', 'September']),
    df_filtered_2023['Crash_Month'].isin(['October', 'November', 'December'])]
    choices = ['Q1', 'Q2', 'Q3', 'Q4']
    df_filtered_2023['Crash_Month'] = np.select(conditions, choices, default='Unknown')
    df_filtered_2023 = df_filtered_2023.reset_index(drop=True)
            
    columns = df_filtered_2023[['Crash_Month','Crash_Nature','Crash_Type','Crash_Controlling_Authority','Crash_Roadway_Feature',
                          'Crash_Traffic_Control','Crash_Speed_Limit','Crash_Road_Surface_Condition','Crash_Atmospheric_Condition','Crash_Lighting_Condition',
                          'Crash_Road_Horiz_Align','Crash_Road_Vert_Align']]

    column_name = ['Crash_Month', 'Crash_Nature','Crash_Type','Crash_Controlling_Authority','Crash_Roadway_Feature',
                          'Crash_Traffic_Control','Crash_Speed_Limit','Crash_Road_Surface_Condition','Crash_Atmospheric_Condition','Crash_Lighting_Condition',
                          'Crash_Road_Horiz_Align','Crash_Road_Vert_Align']
    #Encoder
    X_encoder = OneHotEncoder(sparse_output=False)
    X_encoded = X_encoder.fit_transform(columns)

    X_column_names = X_encoder.get_feature_names_out(column_name)

    X = pd.DataFrame(X_encoded, columns=X_column_names)
    X = X.drop(columns=['Crash_Speed_Limit_nan', 'Crash_Road_Vert_Align_nan', 'Crash_Road_Surface_Condition_Unknown', 'Crash_Atmospheric_Condition_Unknown','Crash_Lighting_Condition_Unknown'])
    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(df_filtered_2023['Crash_Severity'])
    return X, y, df_filtered_2023

def preparation_class_threshold_check():
    df_location = pd.read_csv('../../Dataset/Location.csv')
    Q1 = df_location['Crash_Longitude'].quantile(0.10)
    Q3 = df_location['Crash_Longitude'].quantile(0.90)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_location[(df_location['Crash_Longitude'] < lower_bound) | (df_location['Crash_Longitude'] > upper_bound)]
    df_no_outliers = df_location[(df_location['Crash_Longitude'] >= lower_bound) & (df_location['Crash_Longitude'] <= upper_bound)]
    Q1 = df_no_outliers['Crash_Latitude'].quantile(0.10)
    Q3 = df_no_outliers['Crash_Latitude'].quantile(0.90)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 *IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_no_outliers[(df_no_outliers['Crash_Latitude'] < lower_bound) | (df_no_outliers['Crash_Latitude'] > upper_bound)]
    df_no_outliers = df_no_outliers[(df_no_outliers['Crash_Latitude'] >= lower_bound) & (df_no_outliers['Crash_Latitude'] <= upper_bound)]
    df_filtered_2023 = df_no_outliers.copy()
    df_filtered_2023['Crash_Severity'] = df_filtered_2023['Crash_Severity'].replace({'Hospitalisation': 'Serious injury'})
    df_filtered_2023['Crash_Severity'] = df_filtered_2023['Crash_Severity'].replace({'Medical treatment': 'Serious injury'})
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Road_Surface_Condition']!='Unknown']
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Atmospheric_Condition']!='Unknown']
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Lighting_Condition']!='Unknown']
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Type']!='Other']
    df_filtered_2023 = df_filtered_2023[df_filtered_2023['Crash_Speed_Limit']!= None]
    # Removing < 0.01 theshold
    threshold = 0.01
    remove_frequency = []
    frequency = df_filtered_2023['Crash_Nature'].value_counts(normalize=True)
    remove_frequency.append(frequency[frequency < threshold])
    df_filtered_2023 = df_filtered_2023[~df_filtered_2023['Crash_Nature'].isin(frequency[frequency < threshold].index)]
    
    frequency = df_filtered_2023['Crash_Roadway_Feature'].value_counts(normalize=True)
    remove_frequency.append(frequency[frequency < threshold])
    df_filtered_2023 = df_filtered_2023[~df_filtered_2023['Crash_Roadway_Feature'].isin(frequency[frequency < threshold].index)]
    
    frequency = df_filtered_2023['Crash_Controlling_Authority'].value_counts(normalize=True)
    remove_frequency.append(frequency[frequency < threshold])
    df_filtered_2023 = df_filtered_2023[~df_filtered_2023['Crash_Controlling_Authority'].isin(frequency[frequency < threshold].index)]
    
    frequency = df_filtered_2023['Crash_Traffic_Control'].value_counts(normalize=True)
    remove_frequency.append(frequency[frequency < threshold])
    df_filtered_2023 = df_filtered_2023[~df_filtered_2023['Crash_Traffic_Control'].isin(frequency[frequency < threshold].index)]
    
    conditions = [
    df_filtered_2023['Crash_Month'].isin(['January', 'February', 'March']),
    df_filtered_2023['Crash_Month'].isin(['April', 'May', 'June']),
    df_filtered_2023['Crash_Month'].isin(['July', 'August', 'September']),
    df_filtered_2023['Crash_Month'].isin(['October', 'November', 'December'])]
    choices = ['Q1', 'Q2', 'Q3', 'Q4']
    df_filtered_2023['Crash_Month'] = np.select(conditions, choices, default='Unknown')
    df_filtered_2023 = df_filtered_2023.reset_index(drop=True)
            
    columns = df_filtered_2023[['Crash_Month','Crash_Nature','Crash_Type','Crash_Controlling_Authority','Crash_Roadway_Feature',
                          'Crash_Traffic_Control','Crash_Speed_Limit','Crash_Road_Surface_Condition','Crash_Atmospheric_Condition','Crash_Lighting_Condition',
                          'Crash_Road_Horiz_Align','Crash_Road_Vert_Align']]

    column_name = ['Crash_Month', 'Crash_Nature','Crash_Type','Crash_Controlling_Authority','Crash_Roadway_Feature',
                          'Crash_Traffic_Control','Crash_Speed_Limit','Crash_Road_Surface_Condition','Crash_Atmospheric_Condition','Crash_Lighting_Condition',
                          'Crash_Road_Horiz_Align','Crash_Road_Vert_Align']
    #Encoder
    X_encoder = OneHotEncoder(sparse_output=False)
    X_encoded = X_encoder.fit_transform(columns)

    X_column_names = X_encoder.get_feature_names_out(column_name)

    X = pd.DataFrame(X_encoded, columns=X_column_names)
    X = X.drop(columns='Crash_Speed_Limit_nan')
    X = X.drop(columns='Crash_Road_Vert_Align_nan')
    # y_encoder = LabelEncoder()
    # y = y_encoder.fit_transform(df_filtered_2023['Crash_Severity'])
    y = df_filtered_2023[['Crash_Severity']]
    return X, y, df_filtered_2023, remove_frequency

def featureSelection(X, corr):
    correlation = X.corr()
    high_corr_pairs = []

    for i in range(correlation.shape[0]):
        for j in range(i+1, correlation.shape[0]):
            if correlation.iloc[i, j] > corr: # Set correlation
                high_corr_pairs.append((correlation.index[i], correlation.columns[j], correlation.iloc[i, j]))

    high_corr_df = pd.DataFrame(high_corr_pairs, columns=['Feature1', 'Feature2', 'Correlation'])
    return high_corr_df

def roc_curve(y_pred, y_test):
    y = label_binarize(y, classes=[0,1,2,3,4])
    n_classes = 5
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot of a ROC curve for a specific class
    for i in range(n_classes):
        plt.figure()
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
    
