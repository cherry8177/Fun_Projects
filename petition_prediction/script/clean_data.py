import pandas as pd
import re
import numpy as np
def str2float(s):  
    s = str(s).strip().replace(',','')
    s = re.sub('[^0-9]+', '0', s)
    return float(s)

def data_clean(raw_dict):
    def str2float(s):
        s = str(s).strip().replace(',','')
        s = re.sub('[^0-9]+', '0', s)
        return float(s)
    
    def remove_sw(words, sw):
        word = [w for w in words if not w in sw]
        return word
    
    def clean_text( col):
        letters_only=(col.apply(lambda x:re.sub(u"\xa0",u" ",x))
        #.apply(lambda x: BeautifulSoup(x).get_text())
        .apply(lambda x:re.sub("[^a-zA-Z]"," ",x))
                 )
        lower_case=letters_only.apply(lambda x: x.lower().split())
        from nltk.corpus import stopwords # Import the stop word list
        stopwords=set(stopwords.words("english"))
        clean_texts = []
        num_texts = col.size
        for i in range( 0, num_texts ):
        # Call our function for each one, and add the result to the list of
        # clean 
            clean_texts.append( " ".join(remove_sw(lower_case[i],stopwords)))
        return clean_texts
    

    
    df=pd.DataFrame(raw_dict)
    df=pd.DataFrame.from_dict(raw_dict,orient='index')
    df.reset_index(inplace=True)
    df.columns=['Petition_Url',"Title","Text","No_Supporters","progression","start"]
    df.start[df.start.str.len() == 0] = 'None\nNone'
    df.start = df.start.str.split('\n')
    df[['start_time', 'start_individual']] =  pd.DataFrame([x for x in df.start])
    df.No_Supporters=df.No_Supporters.apply(str2float)
    df['No_Supporters_log']=np.log10(df.No_Supporters)
    df.Text=clean_text(df.Text)
    df.Title=clean_text(df.Title)
    df['Text_len']=df.Text.str.split(' ').apply(lambda x: len(x))
    df['Title_len']=df.Title.str.split(' ').apply(lambda x: len(x))
    df['Text_str_len']=df.Text.str.split(' ').apply(lambda x: sum(len(w) for w in x)/ len(x))
    df['Title_str_len']=df.Title.str.split(' ').apply(lambda x: sum(len(w) for w in x)/ len(x))
    df['Text_len_p1']=(df['Text_len']<270)*df['Text_len']
    df['Text_len_p2']=(df['Text_len']>270)*df['Text_len']
    return df
    
    
    
