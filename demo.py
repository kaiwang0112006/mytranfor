def parse_movie_name(file):
    movie_x = []
    movie_y = [] 

    data = []
    with open(file) as fin:
        cinema_reader = csv.DictReader(fin)
        for row in cinema_reader:
            row['film_name'] = re.sub("¡¤()", "",row['film_name'])
            seg_list = list(jieba.cut(row['film_name'] ))
            row['film_name'] = ' '.join(seg_list)
            row['type'] = film_type_code[row['film_code'][3]]
            movie_y.append(row['film_id'])
            data.append(row)
    
    vect = DictVectorizer()
    tokenize = CountVectorizer().build_tokenizer()
    movie_x = vect.fit_transform(features(tokenize,d) for d in data)
    return movie_x, movie_y, vect

def features(tokenize,document):
    terms = tokenize(document['film_name'])
    d = {'country': document['film_code'][:3],
         'type': document['type'],
         'year': document['film_code'][-4:]
         }

    for t in terms:
        d[t] = d.get(t, 0) + 1
    return d