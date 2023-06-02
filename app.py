import nltk
from flask import Flask,  jsonify, render_template, request
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
from collections import Counter

app = Flask(__name__, static_folder='static')

if not os.path.exists('uploaded_files'):
    os.makedirs('uploaded_files')

def extract_top_k(file_path, k):
    word_counts = Counter()
    with open(file_path, 'r', encoding= 'utf-8') as file:
        for line in file:
            words = line.strip().split()  # 줄을 단어로 분할
            word_counts.update(words)  # 단어들의 빈도를 업데이트

    # 불용어 데이터를 다운로드
    nltk.download('stopwords')
    
    # 불용어 제거를 위해 nltk.corpus.stopwords를 사용
    stopwords = set(nltk.corpus.stopwords.words('english'))
    
    # 불용어가 아닌 단어들만 선택
    non_stopwords = [(word, count) for word, count in word_counts.items() if word.lower() not in stopwords]
    
    # Counter 객체로 변환하여 most_common 함수를 사용
    counter_non_stopwords = Counter(dict(non_stopwords))
    
    # 빈도 기준으로 상위 k개의 단어를 추출
    top_k_words = counter_non_stopwords.most_common(k)
    
    # 상위 k개의 단어의 빈도를 스코어로 사용
    top_k_scores = [count for _, count in top_k_words]

    # 결과를 scores.txt 파일에 저장합니다
    with open('uploaded_files/scores.txt', 'w', encoding='utf-8') as output_file:
        for word, count in top_k_words:
            output_file.write(f"{word}\t{count}\n")
    
    return top_k_words

def generate_sentences_gpt2(scores):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    generated_sentences = []
    for score in scores:
        prompt = "Given a score of {}, generate a sentence:".format(score)
        inputs = tokenizer.encode(prompt, return_tensors='pt')
        outputs = model.generate(inputs, max_length=50, num_return_sequences=1, temperature=0.7)
        generated_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_sentences.append(generated_sentence)
        
        
    
    return '\n'.join(generated_sentences)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            file_path = 'uploaded_files/' + file.filename
            file.save(file_path)
            k = int(request.form['k'])

            top_scores = extract_top_k(file_path, k)
            generated_sentences = generate_sentences_gpt2(top_scores)

            return render_template('index.html', scores=top_scores, sentences=generated_sentences)
        else:
            return render_template('index.html', error='No file selected')
    else:
        return render_template('index.html')
    
@app.route('/generate_sentences', methods=['POST'])
def generate_sentences():
    k = request.form.get('k')  # 'k' 필드 값을 가져옴

    if k is not None and k.isdigit():  # 값이 존재하고, 숫자로 구성되어 있는지 확인
        k = int(k)  # 정수로 변환
        top_scores = extract_top_k('uploaded_files/scores.txt', k)
        generated_sentences = generate_sentences_gpt2(top_scores)
        return jsonify(sentences=generated_sentences.split('\n'))
    else:
        return jsonify(error='Invalid value for "k"')

if __name__ == '__main__':
    app.run(debug=True)

