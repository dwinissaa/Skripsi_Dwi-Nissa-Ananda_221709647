from copy import deepcopy
from datetime import datetime
import pandas as pd
from nltk.corpus import stopwords 
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tensorflow.keras.utils import Progbar

def mergeData(df1,df2,index):
    dataframe_ = pd.concat([df1,df2]).drop_duplicates(subset=[index], keep='first')
    dataframe_.index = range(len(dataframe_))
    return dataframe_

# remove missing value on column content
def rmvMissCont(dataframes):
    dataframes_ = deepcopy(dataframes)
    for key in dataframes_:
        df = dataframes_[key]
        bool_ = df['Content'].apply(lambda x: False if pd.isnull(x) else False if x.strip()=='' else True)
        df = df[bool_]
        dataframes_[key] = df
    return dataframes_

# Berdasarkan survey, hanya ada 3 kolom yang terpenuhi: Link, Date Time, Title, Content. 
# Adapun kolom "Content" HARUS ADA. (yg lainnya sunnah)
def dropColumn(dataframes):
    dataframes_ = deepcopy(dataframes)
    for key in dataframes_:
        if key=="detik":
            dataframes_[key] = dataframes_[key].drop(columns=['Location','Category'])
        elif key=="jatimnow":
            dataframes_[key] = dataframes_[key].drop(columns=['Category'])
        elif key=="jawapos":
            dataframes_[key] = dataframes_[key].drop(columns=['Location'])
        elif key.startswith('radar'):
            dataframes_[key] = dataframes_[key].drop(columns=['Category'])
    return dataframes_

def convertDateTime(dataframes):
    bulan = {"Januari" : "January", "Februari": "February","Maret": "March", "April": "April", "Mei": "May",
             "Juni": "June", "Juli": "July", "Agustus": "August", "September": "September", "Oktober": "October",
             "November": "November", "Desember": "December"
            }
    bulan2 = {"Jan" : "January", "Feb": "February","Mar": "March", "Apr": "April", "Mei": "May",
             "Jun": "June", "Jul": "July", "Agu": "August", "Sep": "September", "Okt": "October",
             "Nov": "November", "Des": "December"
            }
    hari = {"Senin":"Monday", "Selasa":"Tuesday", "Rabu":"Wednesday", "Kamis":"Thursday", 
            "Jumat":"Friday", "Sabtu":"Saturday", "Minggu":"Sunday"
           }
    
    # Convert to structured
    def convstr_radar(unstruct):
        a = unstruct.split(" | EDITOR :",1)[0]
        a = a.split()
        str_ = ""
        str_+=a[0]+" "
        str_+=bulan[a[1].capitalize()]+" "
        str_+=a[2].replace(',','')+" "
        str_+=''.join(a[3:6])
        return str_
    def convstr_detik(unstruct):
        a = unstruct.replace('WIB','')
        a = a.split()
        str_ = ""
        str_+=hari[a[0].capitalize().replace(',','')]+" "
        str_+=a[1]+" "
        str_+=bulan2[a[2].capitalize()]+" "
        str_+=a[3]+" "
        str_+=a[4]
        return str_
    def convstr_jatimnow(unstruct):
        a = unstruct
        str_ = ""
        a = a.split("\n")
        a = ' '.join(a).split(':')
        str_+=hari[a[0].strip()]+" "
        b = a[1].split()
        str_+=b[0]+" "
        str_+=bulan[b[1]]+" "
        str_+=b[2]+" "
        str_+=a[2].strip()+":"
        str_+=a[3].strip()+":"
        str_+=a[4].strip()
        return str_
    def convstr_jawapos(unstruct):
        a = unstruct.replace(',','')
        a = a.replace('WIB','')
        str_ = ""
        a = a.split()
        str_+=a[0].strip()+" "
        str_+=bulan[a[1].strip()]+" "
        str_+=a[2].strip()+" "
        str_+=a[3].strip()
        return str_
    
    dataframes_ = deepcopy(dataframes)
    for key in dataframes_:
        df = dataframes_[key]
        
        # radarmadiun
        if key in ["radarmadiun","radarsidoarjo","radarjember","radarmalang","radargresik"]:
            for i in df.index:
                x = df['Date Time'].loc[i]
                try:
                    df['Date Time'].loc[i] = datetime.strptime(x, "%d %B %Y %H:%M %p")
                except:
                    try:
                        df['Date Time'].loc[i] = datetime.strptime(x, "%d-%b-%y")
                    except:
                        try:
                            df['Date Time'].loc[i] = datetime.strptime(x, "%m/%d/%Y %H:%M")
                        except:
                            print("An exception occurred in df_index:{}/{}".format(key,i))
                            return
        
        # radarbromo
        elif key=="radarbromo":
            df['Date Time'] = df['Date Time'].apply(lambda x: datetime.strptime(x, "%A, %d %B %Y"))

        # radarsurabaya
        elif key in ["radarsurabaya","radarmojokerto","radarmadura","radarjombang","radarkediri","radarbojonegoro","radarbanyuwangi","radartulungagung"]:
            df["Date Time"] = df["Date Time"].apply(lambda x: datetime.strptime(convstr_radar(x), "%d %B %Y %H:%M:%S"))
        
        # detik
        elif key=="detik":
            df['Date Time'] = df["Date Time"].apply(lambda x: datetime.strptime(convstr_detik(x), "%A %d %B %Y %H:%M"))
        
        # jatimnow
        elif key=="jatimnow":
            df["Date Time"] = df["Date Time"].apply(lambda x: datetime.strptime(convstr_jatimnow(x), "%A %d %B %Y %H:%M:%S"))
        
        # jawapos
        elif key=="jawapos":
            df["Date Time"] = df["Date Time"].apply(lambda x: datetime.strptime(convstr_jawapos(x), "%d %B %Y %H:%M:%S"))
    
    return dataframes_

def filterDate(dataframes):
    dataframes_ = deepcopy(dataframes)
    for key in dataframes:
        lower_boundary = datetime.strptime("1 January 2020 00:00","%d %B %Y %H:%M")
        upper_boundary = datetime.strptime("31 December 2020 23:59","%d %B %Y %H:%M")
        df = dataframes_[key].sort_values(by="Date Time",ascending = False)
        filt = [True if (df['Date Time'].iloc[i]>=lower_boundary and df['Date Time'].iloc[i]<=upper_boundary) else False for i in range(len(df))]
        dataframes_[key] = df[filt]
    return dataframes_

def mergeDatas(dataframes):
    dataframes_ = pd.DataFrame({})
    for key in dataframes:
        dataframes_ = pd.concat([dataframes_,dataframes[key]])
    dataframes_ = dataframes_.drop_duplicates(subset=['Link'], keep='first')
    dataframes_.index = range(len(dataframes_))
    return dataframes_

# rmv
def rmvAddText(txt):
    keywords = ['baca juga','tonton video','lihat juga video','simak video','simak juga','tonton juga',
                '[gambas:video 20detik]','foto:','editor:','reporter:','fotografer:','pewarta:',
                'foto :','editor :','reporter :','fotografer :','pewarta :']
    for keyword in keywords:
        arr = []
        ADD=False
        for x in sent_tokenize(txt):
            if keyword in x.strip():
                ADD = True
            if ADD: ADD = False; continue
            else:
                if x.strip()!='': arr.append(x.strip());
        txt = ' '.join(arr).strip()
    return txt

def rmvTags(txt):
    TAGS = False; txt_arr = sent_tokenize(txt); idx = []; last = txt_arr[-1].strip()
    if '\n' in last or last.startswith('('): TAGS = True
    if TAGS: out = ' '.join(txt_arr[:-1])
    else: out = ' '.join(txt_arr)
    return out

def rmvASCII(contentRaw):
    return ''.join(["" if ord(i) < 32 or ord(i) > 126 else i for i in contentRaw])

class PreProcessArticles:
    def __init__(self):
        self.stopw =  set(stopwords.words('indonesian')+list(string.punctuation)+list(['\'\'','--','``']))
        self.factory = StemmerFactory()
        self.stemmer = self.factory.create_stemmer()
    def processArticles(self, dat):
        self.data_out = deepcopy(dat)
        self.progbar = Progbar(len(dat))
        idx = self.data_out.index
        for i,j in enumerate(idx):
            self.data_out.loc[j,'Content'] = self._processArticle(self.data_out.loc[j,'Content'])
            self.progbar.update(i+1)
        print('Finished.')
        return self.data_out
    def _processArticle(self, article):
        article = article.lower()
        article = rmvTags(rmvAddText(article))
        article = rmvASCII(article)
        article = word_tokenize(article)
        article = [word for word in article if word not in self.stopw]
        article = ' '.join([self.stemmer.stem(word) for word in article])
        return article