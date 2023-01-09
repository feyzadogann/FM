import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.expand_frame_repr', False) # ciktinin tek bir satirda olmasini saglar
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.options.display.float_format = '{:.3f}'.format

excel_path=r"D:\FM2022_data\excel\fm2022_players_dataset.xlsx"
def read_data(path):
    df = pd.read_excel(path, parse_dates = True)
    return df

def check_df(dataframe, head = 5):
    """
    Prints details of given dataset
    Parameters
    ----------
    dataframe: pandas DataFrame
    head: int
        Number to restrict the output size

    Returns
    -------
        Nothing, but prints
    """
    print("##################### Info #####################")
    print(dataframe.info(), end = "\n\n")
    print("##################### Shape #####################")
    print(dataframe.shape, end = "\n\n")
    print("##################### NA #####################")
    print(dataframe.isnull().sum(), end = "\n\n")
    print("##################### Head #####################")
    print(dataframe.head(head), end = "\n\n")
    print("##################### Tail #####################")
    print(dataframe.tail(head), end = "\n\n")
    print("##################### Descriptive Statistics #####################")
    print(dataframe.describe([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).T)
    print("##################### # of Uniques #####################")
    for col in dataframe:
        print(col, dataframe[col].nunique())
    print("##################### Top " + str(head) + " Value Counts #####################")
    for col in dataframe.columns:
        print(col)
        print(dataframe[col].value_counts(dropna = False).head(head), end = "\n\n")
    print("##################### Bottom " + str(head) + " Value Counts #####################")
    for col in dataframe.columns:
        print(col)
        print(dataframe[col].value_counts(dropna = False).tail(head), end = "\n\n")

def prepare_data(dataframe):
    """
    Cleans the raw dataset and perform some rename and type conversion operations
    Parameters
    ----------
    dataframe: pandas DataFrame

    Returns
    dataframe: pandas DataFrame
    -------

    """
    dataframe.rename(columns = {"1'e 1": "Birebir",
                                "Bit": "Bitiricilik",
                                "Ces": "Cesaret",
                                "Cls": "Caliskanlik",
                                "Cev": "Ceviklik",
                                "Day": "Dayaniklilik",
                                "Den": "Denge",
                                "DoU": "Dogdugu Ulke",
                                "Elk": "Elle kontrol",
                                "Tac": "Elle oyun baslatma",
                                "GKO": "Genel fiziksel kondisyonu",
                                "Hva": "Hava toplari",
                                "Hzl": "Hizlanma",
                                "Ile": "Iletisim",
                                "KEK": "K-Elle kontrol ve top dagitimi",
                                "Kaf": "Kafa vurusu",
                                "Kar": "Karar alma",
                                "Krr": "Kararlilik",
                                "TZ Gol": "Kariyerinde attigi toplam gol",
                                "TZ Lig Gol": "Kariyerinde attigi toplam lig golu",
                                "TZ Lig Mac": "Kariyerinde oynadigi toplam lig maci",
                                "TZ Mac": "Kariyerinde oynadigi toplam mac",
                                "Kons": "Konsantrasyon",
                                "Kor": "Korner",
                                "Lid": "Liderlik",
                                "Mar": "Markaj",
                                "Ort": "Orta yapma",
                                "Ons": "Onsezi",
                                "Yet": "Ozel yetenek",
                                "Pen": "Penalti kullanma",
                                "Poz": "Pozisyon alma",
                                "Ref": "Refleksler",
                                "Sgk": "Sogukkanlilik",
                                "Baslangic": "Sozlesme Baslangic tarihi",
                                "Bitis": "Sozlesme bitis tarihi",
                                "Toy": "Takim oyunu",
                                "Tek": "Teknik",
                                "Uzs": "Uzaktan sut",
                                "U Tc": "Uzun tac",
                                "Viz": "Vizyon",
                                "Vuc": "Vucut zindeligi",
                                "Yum": "Yumrukla uzaklastirma egilimi",
                                "Zip": "Ziplama",
                                "Tsu": "Dripling",
                                "IKn": "Ilk kontrol",
                                "Srb": "Serbest vuruslar",
                                "Tkp": "Top kapma",
                                "Agre": "Agresiflik",
                                "TzA": "Topsuz alan",
                                "Ort P": "Ortalama puan",
                                "Frk": "Eksantriklik",
                                "Hkm": "Bolge hakimiyeti",
                                "DT": "Dogum tarihi",
                                "ANI": "Ani cikis egilimi",
                                "Ayk": "Degaj"
                                }, inplace = True)
    dataframe["Boy(cm)"] = dataframe["Boy"].astype(str).map(lambda x: x.replace(" cm", "")).astype(float)
    dataframe["Agirlik(kg)"] = dataframe["Agirlik"].astype(str).map(lambda x: x.replace(" kg", "")).astype(float)
    dataframe["Maas Aylik(euro)"] = dataframe[
        "Maas Aylik(euro)"].astype(str).map(lambda x: x.replace("€", "").replace(" ", "").replace(".", "")).astype(float)
    # dataframe["Min Srb Kal Bdl (euro)"] = dataframe["Min Srb Kal Bdl (euro)"].astype(str).map(lambda x: x.replace("€", "")).astype(float)

    dataframe.drop(columns = ["Istenen Bonservis", "Min Srb Kal Bdl (euro)", "Dogdugu Il", "Dogum tarihi", "Gol",
                              "Genel Mutluluk", "Boy",
                              "Agirlik", "Ortalama puan", "HuTa", "DI", "SvTe", "Or G", "Ofs", "Kon", "I.Sut", "Is P",
                              "Pena",
                              "Kurt. Pen.", "KLS", "Maas", "Gelisim Onerisi", "Oyuncu Durumu", "Potansiyel", "Tur",
                              "Yetenek",
                              "Yeni Oyun Karakteristigi", "Bulundugu Yer", "Yer", "Birikim Sev.", "Seviye", "Sut",
                              "Min Srb Kal Bdl",
                              "Eksiler", "Artilar", "Moral", "K-Elle kontrol ve top dagitimi", "Uzaklastirma",
                              "Piyasa Degeri"], inplace = True)

    # df["Sozlesme Baslangic tarihi"] = pd.to_datetime(df["Sozlesme Baslangic tarihi"], errors='coerce')
    # Kiralık olmasından kaynaklı çoklanan oyuncular
    dataframe = dataframe.sort_values(by = ["Isim", "Sozlesme bitis tarihi"]).groupby("Isim", as_index = False). \
        apply(lambda x: x.iloc[-1])
    dataframe.reset_index(inplace = True, drop = True)
    # Tüm değişkenleri NULL olan gözlem birimlerinin veri setinden çıkarılması
    dataframe = dataframe[~(dataframe["OID"].isnull())]
    dataframe["OID"] = dataframe["OID"].astype("object")
    return dataframe

def grab_col_names(dataframe, cat_th = 17, car_th = 300, show_info = True):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: pandas dataframe
                Değişken isimleri alınmak istenilen dataframe
        target_col: pandas Series
                Analiz edilmek istenen hedef değişken.
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri
        show_info: bool
                Toplam gözlem, değişken, kategorik, nümerik, nümerik görünümlü kategorik ve kardinal değişken sayısını
                ekrana yazdırma kararı

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   ((dataframe[col].dtypes == "int64") or (dataframe[col].dtypes == "float64"))]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if
                (dataframe[col].dtypes != "O") & ("id" not in col.lower()) & ("index" not in col)]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    if show_info:
        print(f"Observations: {dataframe.shape[0]}")
        print(f"Variables: {dataframe.shape[1]}")
        print(f'# of cat_cols: {len(cat_cols)}')
        print(cat_cols, end = "\n\n")
        print(f'# of num_cols: {len(num_cols)}')
        print(num_cols, end = "\n\n")
        print(f'# of cat_but_car: {len(cat_but_car)}')
        print(cat_but_car, end = "\n\n")
        print(f'# of num_but_cat: {len(num_but_cat)}')
        print(num_but_cat, end = "\n\n")

    return list(cat_cols), list(num_cols), list(cat_but_car), list(num_but_cat)

def show_counts(dataframe, column_type, head = 20):
    """
    Prints the value_count of each column in column_type

    Parameters
    ----------
    dataframe: pandas DataFrame
    column_type: list
        list object containing columns of same data type which is returned from grab_col_names() function
    head: int
        Number to restrict the output size
    Returns
    -------
        Nothing, but prints

    """
    for col in column_type:
        print(col, end = "\n-------------------\n")
        print(dataframe[col].value_counts(dropna = False).head(head), end = "\n\n")

def show_percentiles(dataframe, all_ = True, *args):
    """
    Prints the descriptive statistics of given dataset

    Parameters
    ----------
    dataframe: pandas DataFrame
        Dataset
    all_: bool
        Decision to show whether all dataframe columns or spesific one(s)
    args:
        If _all=False, selected column(s)

    Returns
    -------
        Nothing, but prints
    """
    if all_:
        print(dataframe.describe([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).T)
    else:
        for var in args:
            print(dataframe[var].describe([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).T, end = "\n\n")

def num_summary(dataframe, numerical_col, plot = False):
    """
    Gives a statistical summary of numerical columns with plotting option
    Parameters
    ----------
    dataframe: pandas DataFrame
        dataset
    numerical_col: str
        column to be summarized
    plot: bool
        Option to plot histogram of numerical_col

    Returns
    -------
        Nothing, but prints

    """
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins = 50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block = True)

    print("#####################################")

def cat_summary(dataframe, col_name, plot = False):
    """
       Gives a statistical summary of categorical columns with plotting option
       Parameters
       ----------
       dataframe: pandas DataFrame
           dataset
       col_name: str
           column to be summarized
       plot: bool
           Option to show countplot of categorical column col_name

       Returns
       -------
           Nothing, but prints


       """
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[
                            col_name].value_counts(dropna = False) / len(dataframe)}), end = '\n\n')

    if plot:
        sns.countplot(x = dataframe[col_name], data = dataframe)
        plt.show(block = True)

def show_nullity_percentages(dataframe):
    return (dataframe.isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending = False)[:15]

def visualize_nulls(dataframe, null_proportion):
    """
    Plots percentage of null columns by using pie graph from plotly.express library
    Parameters
    ----------
    dataframe: pandas DataFrame
        dataset
    null_proportion: float, int
        Specifies the lower limit proportion for a column to be included in the graph

    Returns
    -------
        Nothing, but plots
    """
    import plotly.express as px
    import plotly.io as pio
    pio.renderers.default = "browser"

    missing_data = dataframe.isnull().sum().sort_values(ascending = False)
    missing_data = missing_data.reset_index(drop = False)
    missing_data = missing_data.rename(columns = {"index": "Columns", 0: "Value"})
    missing_data['Proportion'] = (missing_data['Value'] / len(dataframe)) * 100
    sample = missing_data[missing_data['Proportion'] > null_proportion]
    fig = px.pie(sample, names = 'Columns', values = 'Proportion',
        color_discrete_sequence = px.colors.sequential.Viridis_r,
        title = 'Percentage of Missing values in Columns')
    fig.update_traces(textposition = 'inside', textinfo = 'label')
    fig.update_layout(paper_bgcolor = 'rgba(0,0,0,0)',
        plot_bgcolor = 'rgba(0,0,0,0)',
        font = dict(family = 'Cambria, monospace', size = 16, color = '#000000'))
    fig.show(block = True)

def outlier_thresholds(dataframe, col_name, q1 = 0.01, q3 = 0.99):
    """
    Returns upper and lower limit of given col_name according to given quartiles to decide thresholds
    Parameters
    ----------
    dataframe: pandas DataFrame
        dataset
    col_name: str
        column name to be examined for upper and lower thresholds
    q1: float
        1.quartile percentage
    q3: float
        3.quartile percentage

    Returns
    -------
    low_limit: float
        Lower limit of threshold specified by quartiles
    up_limit: float
        Upper limit of threshold specified by quartiles

    """
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name, q1 = 0.01, q3 = 0.99):
    """
    Returns a boolean to show if given dataset column contains outliers according to given quartiles
    Parameters
    ----------
    dataframe: pandas DataFrame
        dataset
    col_name: str
        column to be detected for outliers
    q1: float
        1.quartile percentage
    q3: float
        3.quartile percentage

    Returns
    -------
        bool
    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis = None):
        return True
    else:
        return False

def show_outlier_count_ratio(dataframe, col_name, q1 = 0.01, q3 = 0.99):
    """
    Shows count and ratio of outlier observations, if any (using check_outlier), among non-null observations in dataset
    Parameters
    ----------
    dataframe: pandas DataFrame
        dataset
    col_name: str
        column name
    q1: float
        1.quartile percentage
    q3: float
        3.quartile percentage

    Returns
    -------
        Nothing, but prints
    """
    flag_result = check_outlier(dataframe, col_name, q1, q3)
    if flag_result:
        low, up = outlier_thresholds(dataframe, col_name, q1, q3)
        outlier_count = dataframe[(dataframe[col_name] > up) | (dataframe[col_name] < low)].shape[0]
        non_null_cell_count = dataframe[col_name].notnull().sum()
        outlier_ratio = (outlier_count / non_null_cell_count) * 100
        print(f"{col_name} --> Outlier count: {outlier_count}, Outlier ratio among non-nulls: %{outlier_ratio:.4f}")
    else:
        print("No outlier found.")

def show_outliers(dataframe, col_name, q1 = 0.01, q3 = 0.99):
    """
    Shows count and ratio of outlier observations, if any (using check_outlier), with the information of col_name thresholds
    Parameters
    ----------
    dataframe: pandas DataFrame
        dataset
    col_name: str
        column name
    q1: float
        1.quartile percentage
    q3: float
        3.quartile percentage

    Returns
    -------
        pandas DataFrame or None
    """
    flag = check_outlier(dataframe, col_name, q1, q3)
    if flag:
        show_outlier_count_ratio(dataframe, col_name, q1, q3)
        low_l, up_l = outlier_thresholds(dataframe, col_name, q1, q3)
        print(f"Outlier thresholds for {col_name}: {low_l, up_l}", end = "\n\n")
        if dataframe.shape[0] < 1000:
            return dataframe[(dataframe[col_name] > up_l) | (
                        dataframe[col_name] < low_l)].sort_values(by = col_name, ascending = False)
        else:
            return None

def list_outliers(dataframe, q1 = 0.01, q3 = 0.99):
    """
    Lists outlier column names in numerical columns using grab_col_names() function
    Parameters
    ----------
    dataframe: pandas DataFrame
    q1: float
        1.quartile percentage
    q3: float
        3.quartile percentage

    Returns
    -------
    outliers: list
        columns containing outliers in list format
    """
    outliers = []
    cat_cols, num_cols, cat_but_card_cols, num_but_cat_cols = grab_col_names(dataframe, show_info = False)
    for col_name in num_cols:
        flag = check_outlier(dataframe, col_name, q1, q3)
        if flag:
            outliers.append(col_name)
    return outliers

def high_correlated_cols(dataframe, plot = False, corr_th = 0.90):
    """
    Returns correlated columns by given absolute correlation threshold and plots heatmap
    Parameters
    ----------
    dataframe: pandas DataFrame
        dataset
    plot: bool
        decision to plot a heatmap
    corr_th: float, int
        numerical to specify lower correlation limit

    Returns
    -------
    drop_list: list
         list containing columns that have correlation coefficient greater than corr_th
    """
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k = 1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        sns.set(rc = {'figure.figsize': (20, 20)})
        sns.heatmap(corr, cmap = "RdBu", annot = True, annot_kws = {"size": 6}, fmt = ".1%")
        plt.show(block = True)
    return drop_list

def catch_corr_level(dataframe, col_name, corr_levels):
    """
    Catches correlation between columns according to corr_level
    Parameters
    ----------
        dataframe: pandas DataFrame
            dataset
        col_name: pandas Series
            dataset column to be used for correlation
        corr_levels: list
            list containing correlation level indicator(s), "very low", "low", "moderate", "high", "very high"
            (0 <= corr <= 0.19): very low
            (0.2 <= corr <= 0.39): low
            (0.4 <= corr <= 0.59): moderate
            (0.6 <= corr <= 0.79): high
            (corr >= 0.8): very high

    Returns
    -------


    """
    corr_df_list = []
    for corr_level in corr_levels:
        corr_df = dataframe.corr().abs()[col_name]
        if corr_level == "very high":
            corr_df = corr_df[corr_df.values >= 0.8]
        elif corr_level == "high":
            corr_df = corr_df[(corr_df.values <= 0.79) & (corr_df.values >= 0.6)]
        elif corr_level == "moderate":
            corr_df = corr_df[(corr_df.values <= 0.59) & (corr_df.values >= 0.4)]
        elif corr_level == "low":
            corr_df = corr_df[(corr_df.values <= 0.39) & (corr_df.values >= 0.2)]
        elif corr_level == "very low":
            corr_df = corr_df[(corr_df.values <= 0.19) & (corr_df.values >= 0)]
        else:
            print("No such correlation level")
            break
        corr_df_list.append(corr_df)
    return corr_df_list

def plot_nullity_corr(dataframe):
    """
    Plots nullity correlation matrix using missingno library
    Parameters
    ----------
    dataframe: pandas DataFrame

    Returns
    -------
        Nothing, but plots
    """
    import missingno as msno
    msno.matrix(dataframe, figsize = (25, 25), fontsize = 8, labels = 10)
    plt.show(block = True)

def handle_missings(dataframe):
    """
    Applies specific steps to dataset to handle missing values
    Parameters
    ----------
    dataframe: pandas DataFrame
        dataset

    Returns
    -------
        Nothing

    """
    dataframe["Kiralama Bitisi"].fillna("Kiralik_Sozlesmesi_Yoktur", inplace = True)  # her oyuncu kiralık olmak zorunda değildir.
    dataframe["Medya Gozunde"].fillna("Bilgi_Saglanmamis", inplace = True)  # lisanssız oyuncular
    dataframe["Bilgi"].fillna("-", inplace = True)  # her oyuncu için detaylı bilgi olmayabilir
    dataframe["Kisilik"].fillna("Bilgi_Saglanmamis", inplace = True)  # lisanssız ve genç oyuncular
    dataframe["Iki. Pozisyon"].fillna("Ikinci_Pozisyon_Yok", inplace = True)  # her oyuncu ikinci pozisyona sahip olmayabilir
    dataframe["Kariyerinde attigi toplam gol"] = dataframe[
        "Kariyerinde attigi toplam gol"].fillna(0).astype(int)  # kariyeri boyunca maça çıkmamış olanlar
    dataframe["Kariyerinde attigi toplam lig golu"] = dataframe[
        "Kariyerinde attigi toplam lig golu"].fillna(0).astype(int)  # kariyeri boyunca maça çıkmamış olanlar
    dataframe["Kariyerinde oynadigi toplam lig maci"] = dataframe[
        "Kariyerinde oynadigi toplam lig maci"].fillna(0).astype(int)  # kariyeri boyunca maça çıkmamış olanlar
    dataframe["Kariyerinde oynadigi toplam mac"] = dataframe[
        "Kariyerinde oynadigi toplam mac"].fillna(0).astype(int)  # kariyeri boyunca maça çıkmamış olanlar
    dataframe.drop(dataframe.loc[(dataframe["Lig"].isnull()), :].index, inplace = True)  # tek satır

def replace_with_thresholds(dataframe):
    """
    Replaces outlier observations with its corresponding threshold values to suppress outliers
    Parameters
    ----------
    dataframe: pandas DataFrame
        dataset

    Returns
    -------
        Nothing
    """
    outlier_cols = list_outliers(dataframe)
    for variable in outlier_cols:
        low_limit, up_limit = outlier_thresholds(dataframe, variable)
        dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
        dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def feature_eng(datafrm):
    """
    Performs feature engineering operations spesific to football manager dataset
    Parameters
    ----------
    datafrm: pandas DataFrame
        dataset

    Returns
    -------
    df: pandas DataFrame
        Resulting dataset

    """
    # ÖZELLİKLER: BMI, VÜCUT TİPİ, YAŞ GRUBU
    # --------------------------------------------
    datafrm['BMI'] = datafrm['Agirlik(kg)'].astype(int) / ((datafrm['Boy(cm)'] / 100) ** 2).astype(int)
    datafrm["VücutTipi"] = pd.cut(
        datafrm["BMI"], [0, 20, 25, 30, 35, 45], right = False, labels = ['Zayıf', 'İdeal', 'HafifŞişman',
                                                                          'DikkatEdilmeli',
                                                                          'FazlaŞişman'])
    datafrm["Yas_Grubu"] = pd.cut(
        datafrm["Yas"], [0, 17, 23, 32, 37, 45], right = False, labels = ['AltYapı', 'Genç', 'Yetişkin', 'Tecrübeli',
                                                                          'Emektar'])

    # ÖZELLİKLER: GENEL MEVKİ, TEKNİK SKOR, FİZİKSEL SKOR, MENTAL SKOR, OVERALL SKOR
    # --------------------------------------------------------------------------------------
    datafrm["Birincil Mevki"] = datafrm["Mevki"].map(lambda x: x.split('(')[0].replace(" ", ""))
    # check: df["Birincil Mevki"].unique()


    technical_features = ["Bitiricilik", "Dripling", "Ilk kontrol", "Kafa vurusu", "Korner", "Markaj", "Orta yapma",
                          "Pas",
                          "Penalti kullanma", "Serbest vuruslar", "Top kapma", "Uzaktan sut", "Uzun tac"]

    mental_features = ["Agresiflik", "Cesaret", "Caliskanlik", "Karar alma", "Kararlilik", "Konsantrasyon", "Liderlik",
                       "Onsezi", "Ozel yetenek", "Pozisyon alma", "Sogukkanlilik", "Takim oyunu", "Topsuz alan",
                       "Vizyon", "Iletisim"]

    physical_features = ["Ceviklik", "Dayaniklilik", "Denge", "Guc", "Hiz", "Hizlanma", "Vucut zindeligi", "Ziplama",
                         "Boy(cm)", "Agirlik(kg)"]

    gk_technical_features = ["Ani cikis egilimi", "Birebir", "Bolge hakimiyeti", "Degaj", "Eksantriklik",
                             "Elle kontrol",
                             "Elle oyun baslatma", "Hava toplari", "Ilk kontrol", "Pas", "Refleksler",
                             "Yumrukla uzaklastirma egilimi"]

    technical_features_percentage = len(technical_features) / 100
    mental_features_percentage = len(mental_features) / 100
    physical_features_percentage = len(physical_features) / 100
    gk_technical_features_percentage = len(gk_technical_features) / 100

    hücum = ["ST", "OOS"]
    defans = ["D", "D/KB", "D", "D/OS", "D/OS/OOS", "D/OOS"]
    ofansif_bek = ["D/KB/OS", "KB/OS", "KB/OS/OOS", "D/KB/OS/OOS", "KB", "D/KB/OOS", "KB/OOS"]
    orta_saha = ["DOS", "OS", "OS/OOS", "DOS,OS", "DOS,OS/OOS", "DOS,OOS"]
    kale = ["K"]


    def genel_mevkileri_yarat(dataframe):
        for mevki in ofansif_bek:
            dataframe.loc[(dataframe["Birincil Mevki"] == mevki), "Genel Mevki"] = "Ofansif Bek"

        for mevki in hücum:
            dataframe.loc[(dataframe["Birincil Mevki"] == mevki), "Genel Mevki"] = "Hücum"

        for mevki in defans:
            dataframe.loc[(dataframe["Birincil Mevki"] == mevki), "Genel Mevki"] = "Defans"

        for mevki in orta_saha:
            dataframe.loc[(dataframe["Birincil Mevki"] == mevki), "Genel Mevki"] = "Orta Saha"

        for mevki in kale:
            dataframe.loc[(dataframe["Birincil Mevki"] == mevki), "Genel Mevki"] = "Kale"

    genel_mevkileri_yarat(datafrm)

    datafrm["Teknik_Skor"] = 0
    datafrm["Mental_Skor"] = 0
    datafrm["Fiziksel_Skor"] = 0

    def calculate_overall_score(dataframe_):
        df_kaleciler = dataframe_[dataframe_["Birincil Mevki"] == "K"]
        df_oyuncular = dataframe_[dataframe_["Birincil Mevki"] != "K"]
        # dataframe_.shape[0]

        for gk_col in gk_technical_features:
            df_kaleciler.loc[:, "Teknik_Skor"] += df_kaleciler[gk_col] * gk_technical_features_percentage

        for tech_var in technical_features:
            df_oyuncular.loc[:, "Teknik_Skor"] += df_oyuncular[tech_var] * technical_features_percentage

        df_concat = pd.concat([df_kaleciler, df_oyuncular], ignore_index = True)
        # df_concat.reset_index(inplace = True)
        # df_concat.shape[0]

        for ment_var in mental_features:
            df_concat.loc[:, "Mental_Skor"] += df_concat[ment_var] * mental_features_percentage
        for phys_var in physical_features:
            df_concat.loc[:, "Fiziksel_Skor"] = df_concat[phys_var] * physical_features_percentage


        def calculations(datafr):
            if datafr["Genel Mevki"] == "Ofansif Bek":
                return (datafr['Teknik_Skor'] * 0.35) + (datafr['Mental_Skor'] * 0.20) + (
                        datafr['Fiziksel_Skor'] * 0.45)

            elif datafr["Genel Mevki"] == "Defans":
                return (datafr['Teknik_Skor'] * 0.45) + (datafr['Mental_Skor'] * 0.20) + (
                        datafr['Fiziksel_Skor'] * 0.35)

            elif datafr["Genel Mevki"] == "Orta Saha":
                return (datafr['Teknik_Skor'] * 0.35) + (datafr['Mental_Skor'] * 0.45) + \
                       (datafr['Fiziksel_Skor'] * 0.20)

            elif datafr["Genel Mevki"] == "Hücum":
                return (datafr['Teknik_Skor'] * 0.35) + (datafr['Mental_Skor'] * 0.35) + \
                       (datafr['Fiziksel_Skor'] * 0.30)

            else:
                return (datafr['Teknik_Skor'] * 0.40) + (datafr['Mental_Skor'] * 0.20) + \
                       (datafr['Fiziksel_Skor'] * 0.40)

        df_concat["Overall"] = df_concat.apply(calculations, axis = 1)
        # scale_feature(df_concat, "Overall", (40, 99), "int64")

        return df_concat

    df_result = calculate_overall_score(datafrm)

    # ÖZELLİK: OYUNCU SEGMENTİ
    # -----------------------------
    df_result.loc[:, "Oyuncu_Segmenti"] = pd.qcut(
        df_result["Overall"], 10, labels = ["J", "I", "H", "G", "F", "E", "D", "C", "B", "A"])
    # check: df[["Teknik_Skor", "Mental_Skor", "Fiziksel_Skor", "Overall","Oyuncu_Segmenti"]].groupby("Oyuncu_Segmenti").agg({"count", "min","std", "min", "mean", "max"})

    # Bazı değişkenlerdeki eksikliklerin segment bazında doldurulması:
    df_result["Maas Aylik(euro)"].fillna(
        df_result.groupby("Oyuncu_Segmenti")["Maas Aylik(euro)"].transform("median"), inplace = True)

    f = lambda x: x.mode().iloc[0]
    df_result["Sakatlik Riski"].fillna(
        df_result.groupby("Oyuncu_Segmenti")["Sakatlik Riski"].transform(f), inplace = True)

    df_result.drop(columns = ["Genel Mevki", "Mevki"], inplace = True)

    return df_result

def encode(dataframe):
    """
    Performs One-Hot Encoding and Label Encoding for categorical columns that will be included in model
    Parameters
    ----------
    dataframe: pandas DataFrame
        dataset resulting from feature engineering steps

    Returns
    -------
    dataframe_model: pandas DataFrame
        dataset ready for modelling steps
    """
    model_cols = ["Yas", "Teknik_Skor", "Mental_Skor", "Fiziksel_Skor", "BMI", "Oyuncu_Segmenti", "Kullandigi Ayak",
                  "Birincil Mevki"]

    dataframe_model = dataframe[model_cols]

    def label_encoder(dataframe, col):
        labelencoder = LabelEncoder()
        dataframe[col + '_labeled'] = labelencoder.fit_transform(dataframe[col])
        return dataframe

    dataframe_model = label_encoder(dataframe_model, "Oyuncu_Segmenti")
    # check: df_model[["Oyuncu_Segmenti", "Oyuncu_Segmenti_labeled"]].head()
    # check: df_model.columns

    one_hot_cols = ["Kullandigi Ayak", "Birincil Mevki"]

    def one_hot_encoder(dataframe, categorical_cols, drop_first = True):
        dataframe = pd.get_dummies(dataframe, columns = categorical_cols, drop_first = drop_first)
        return dataframe
    dataframe_model = one_hot_encoder(dataframe_model, one_hot_cols)
    dataframe_model.drop(columns = ["Oyuncu_Segmenti"], inplace = True)

    return dataframe_model

def hyperparameter_optimization(df):
    """
    Performs hyperparameter optimization for K-Means
    Parameters
    ----------
    df: pandas DataFrame

    Returns
    -------
    optimal_cluster_num: int
        Optimal number of clusters
    """
    sc = MinMaxScaler((0, 1))
    df_model = sc.fit_transform(df)
    kmeans = KMeans()
    elbow = KElbowVisualizer(kmeans, k = (1, 30))
    elbow.fit(df_model)
    elbow.show(block = True)
    optimal_cluster_num = elbow.elbow_value_
    return df_model, optimal_cluster_num

def modelling(model_df, elbow_val):
    kmeans = KMeans(n_clusters = elbow_val).fit(model_df)
    clusters = kmeans.labels_
    return clusters

def final_df(clusters):
    """
    Concatenates clusters with original dataset
    Parameters
    ----------
    clusters: int
        Optimal number of cluster resulting from K-Means algorithm

    Returns
    -------
    pandas Dataframe
    """
    df_final = read_data(excel_path)
    df_final.drop(df_final.loc[(df_final["Lig"].isnull()), :].index, inplace = True)
    df_final = prepare_data(df_final)
    df_final["Player_Cluster"] = clusters
    cat_cols, num_cols, cat_but_card_cols, num_but_cat_cols = grab_col_names(df_final, show_info = False)
    show_features_cols = ["Isim", "Bilgi", "Mevki", "Kisilik", "Medya Gozunde"] + num_cols
    return df_final[show_features_cols]

def recommend_player(df, player_name, sort_column, top_n):
    """
    Shows similar players to player_name according to player cluster which is calculated based on K-Means algorithm
    Parameters
    ----------
    df: pandas DataFrame
        finalized dataset
    player_name: str
    sort_column: str
        Name of the column for user to sort among similar players
    top_n: int
        Number of players to be recommended to user
    Returns
    -------
        Nothing, but prints
    """
    print(f"********** Selected Player Info: {player_name.upper()} ********** ")
    player = df[df["Isim"] == player_name]
    print(player, end = "\n\n")
    print(f"********** Top {top_n} Similar Players to {player_name.upper()} (sorted by {sort_column} in descending order)  ********** ",  end = "\n\n")
    player_cluster = df[df["Isim"] == player_name]["Player_Cluster"].values[0]
    df_others = df[~(df["Isim"] == player_name)]
    print(df_others[df_others["Player_Cluster"] == player_cluster].sort_values(by = sort_column, ascending = False).head(top_n))









