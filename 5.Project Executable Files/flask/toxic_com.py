from flask import Flask,render_template,request, url_for
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
import pickle

loaded = CountVectorizer(decode_error='replace',vocabulary=pickle.load(open(r'C:\Users\dadia\OneDrive\Desktop\abhi do\flask\word_feats.pkl','rb')))

app = Flask(__name__)

def clean_text(text):
    if text is None:
        return ""
    text = str(text)  # Ensure it's a string before proceeding
    text = text.lower()
    text=re.sub(r"what's","what is",text)
    text=re.sub(r"\'s"," ",text)
    text=re.sub(r"\'ve","have",text)
    text=re.sub(r"can't","cannot",text)
    text=re.sub(r"n't","not",text)
    text=re.sub(r"i'm","i am",text)
    text=re.sub(r"\'re","are",text)
    text=re.sub(r"\'d","would",text)
    text=re.sub(r"\'ll","will",text)
    text=re.sub(r"\'seuse","excuse",text)
    text=re.sub(r"\W"," ",text)
    text=re.sub(r"\S+"," ",text)
    text=text.strip(' ')
    return text

@app.route('/')
def homepage():
    flag=0
    return render_template('index.html')

@app.route('/predict')
def Classify_Now():
    return render_template('predict.html')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        comment = request.form['comment']
        new_row={'comment_text':comment}
        user_df=pd.DataFrame(columns=['comment_text'])
        user_df = pd.concat([user_df, pd.DataFrame([new_row])], ignore_index=False)
        user_features = loaded.transform(user_df['comment_text'])
        cols_target=['insult','toxic','severe_toxic','identity_hate','threat','obscene']
        lst =[]
        mapper ={}
        for label in cols_target:
            mapper[label] = loaded
            model=pickle.load(open(f'flask\{label}_model.sav','rb'))
            print('...processing {}'.format(label))
            user_y_prob=model.predict_proba(user_features)[:,1]
            print(label,":",user_y_prob[0])
            lst.append([label,user_y_prob])
        print(lst)
        final =[]
        flag=0
        for i in lst:
            if i[1]> 0.3:
                final.append(i[0])
                flag=2
        if not len(final):
            text ="yaayy!! The comment is clean"
            img_url = url_for('static', filename='img/emoji1.jpg')
            flag= 1
        else:
            text="The comment is "
            for i in final:
                text=text+i+" "
            img_url=url_for('static',filename='img/emoji2.jpg')
        print(text)
        return render_template('result.html',message=text,img_url=img_url,flag=flag)

if __name__ == "__main__":
    app.run(host='localhost',debug =True,threaded =False)
