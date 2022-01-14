
import datetime
from dateutil.tz import gettz
import csv
from tqdm import tqdm
import time
import operator
import os
import pickle


def prep_yoochoose(dataPATH, filename):

    dataset = dataPATH + '/' + filename
    
    
    print(f'원래 데이터의 training 세션 수 : 7966257')
    print(f"-- Starting @ %ss : {datetime.datetime.now(gettz('Asia/Seoul'))}")
    print('약 8분 30초')
    
    with open(dataset, "r") as f:
        reader = csv.DictReader(f, delimiter=',')
        sess_clicks = {}
        sess_date = {}
        ctr = 0
        curid = -1
        curdate = None
        for data in tqdm(reader):
            sessid = data['sessionId']
            if curdate and not curid == sessid:
                date = ''
                date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
                sess_date[curid] = date
            curid = sessid
    
            item = data['itemId']
            curdate = ''
            curdate = data['timestamp']
    
            if sessid in sess_clicks:
                sess_clicks[sessid] += [item]
            else:
                sess_clicks[sessid] = [item]
            ctr += 1
        date = ''
        date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
        sess_date[curid] = date
    
    

    # Filter out length 1 sessions
    for s in list(sess_clicks):
        if len(sess_clicks[s]) == 1:
            del sess_clicks[s]
            del sess_date[s]
    
    # Count number of times each item appears
    iid_counts = {}
    for s in sess_clicks:
        seq = sess_clicks[s]
        for iid in seq:
            if iid in iid_counts:
                iid_counts[iid] += 1
            else:
                iid_counts[iid] = 1
    
    sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))
    
    length = len(sess_clicks)
    for s in list(sess_clicks):
        curseq = sess_clicks[s]
        filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))
        if len(filseq) < 2:
            del sess_clicks[s]
            del sess_date[s]
        else:
            sess_clicks[s] = filseq
    
    # Split out test set based on dates
    dates = list(sess_date.items())
    maxdate = dates[0][1]
    
    for _, date in dates:
        if maxdate < date:
            maxdate = date
    
    # 7 days for test
    splitdate = 0
    splitdate = maxdate - 86400 * 1
    
    print('Splitting date', splitdate)      # Yoochoose: ('Split date', 1411930799.0)
    tra_sess = filter(lambda x: x[1] < splitdate, dates)
    tes_sess = filter(lambda x: x[1] > splitdate, dates)
    
    # Sort sessions by date
    tra_sess = sorted(tra_sess, key=operator.itemgetter(1))     # [(sessionId, timestamp), (), ]
    tes_sess = sorted(tes_sess, key=operator.itemgetter(1))     # [(sessionId, timestamp), (), ]

    return tra_sess, tes_sess, sess_clicks



def prep_diginetica(dataPATH, filename):

    
    dataset = dataPATH + '/' + filename

    print(f'원래 데이터의 training 세션 수 : 186670')
    print(f"-- Starting @ %ss : {datetime.datetime.now(gettz('Asia/Seoul'))}")
    print('약 26초')
    
    with open(dataset, "r") as f:
        reader = csv.DictReader(f, delimiter=';')
        sess_clicks = {}
        sess_date = {}
        ctr = 0
        curid = -1
        curdate = None
        for data in tqdm(reader):
            sessid = data['sessionId']
            if curdate and not curid == sessid:
                date = ''
                date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
                sess_date[curid] = date
            curid = sessid
            item = data['itemId'], int(data['timeframe'])
            curdate = ''
            curdate = data['eventdate']
    
            if sessid in sess_clicks:
                sess_clicks[sessid] += [item]
            else:
                sess_clicks[sessid] = [item]
            ctr += 1
        date = ''
        date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
        for i in list(sess_clicks):
            sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))
            sess_clicks[i] = [c[0] for c in sorted_clicks]
        sess_date[curid] = date
    
    
    
    print("-- Reading data @ %ss" % datetime.datetime.now())
    # Filter out length 1 sessions
    for s in list(sess_clicks):
        if len(sess_clicks[s]) == 1:
            del sess_clicks[s]
            del sess_date[s]
    
    # Count number of times each item appears
    iid_counts = {}
    for s in sess_clicks:
        seq = sess_clicks[s]
        for iid in seq:
            if iid in iid_counts:
                iid_counts[iid] += 1
            else:
                iid_counts[iid] = 1
    
    sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))
    
    length = len(sess_clicks)
    for s in list(sess_clicks):
        curseq = sess_clicks[s]
        filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))
        if len(filseq) < 2:
            del sess_clicks[s]
            del sess_date[s]
        else:
            sess_clicks[s] = filseq
    
    # Split out test set based on dates
    dates = list(sess_date.items())
    maxdate = dates[0][1]
    
    for _, date in dates:
        if maxdate < date:
            maxdate = date
    
    # 7 days for test
    splitdate = 0
    splitdate = maxdate - 86400 * 7
    
    print('Splitting date', splitdate)      # Yoochoose: ('Split date', 1411930799.0)
    tra_sess = filter(lambda x: x[1] < splitdate, dates)
    tes_sess = filter(lambda x: x[1] > splitdate, dates)
    
    # Sort sessions by date
    tra_sess = sorted(tra_sess, key=operator.itemgetter(1))     # [(sessionId, timestamp), (), ]
    tes_sess = sorted(tes_sess, key=operator.itemgetter(1))     # [(sessionId, timestamp), (), ]

    return tra_sess, tes_sess, sess_clicks


# print(len(tra_sess))    # 186670    # 7966257
# print(len(tes_sess))    # 15979     # 15324
# print(tra_sess[:3])
# print(tes_sess[:3])
# print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())


# Choosing item count >=5 gives approximately the same number of items as reported in paper
# Convert training sessions to sequences and renumber items to start from 1
# 아이템 번호 1번부터 새로 매기고, 5번 이상 나온 아이템만 살리고, 길이 1인 세션은 버림
def obtian_tra(sess_clicks, tra_sess):
    item_dict = {}
    train_ids = []
    train_seqs = []
    train_dates = []
    item_ctr = 1
    for s, date in tra_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]
                item_dict[i] = item_ctr
                item_ctr += 1
        if len(outseq) < 2:  # Doesn't occur
            continue
        train_ids += [s]
        train_dates += [date]
        train_seqs += [outseq]

    print(f'train 세션 수 : {len(train_seqs)}')
    flattened_sess_o = []
    for tra_s in train_seqs:
        flattened_sess_o += tra_s
    print(f'train 아이템 수 :  {len(set(flattened_sess_o))}')  # d : 43098, y : 37484
    
    return train_ids, train_dates, train_seqs, item_dict

# training에 등장하지 않는 아이템은 test에서 지운다
# Convert test sessions to sequences, ignoring items that do not appear in training set
def obtian_tes(sess_clicks, tes_sess, item_dict):
    test_ids = []
    test_seqs = []
    test_dates = []
    for s, date in tes_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
        if len(outseq) < 2:
            continue
        test_ids += [s]
        test_dates += [date]
        test_seqs += [outseq]
        
    print(f'test 세션 총 개수 : {len(test_seqs)}')
    flattened_sess_ot = []
    for tra_s in test_seqs:
        flattened_sess_ot += tra_s
    print(f'test 데이터 아이템 수 :  {len(set(flattened_sess_ot))}')
    
    return test_ids, test_dates, test_seqs

# tra_seqs_frac 저장하기
def save_tra_seqs_frac(experiment, y_or_d, frac, tra_seqs_frac):
    save_dir = f'exps/experiment{experiment}/{y_or_d}/{y_or_d[0]}{int(1/frac):03}_tra_seqs.pkl'
    print(f'tra_seqs save dir : {save_dir}')
    with open(save_dir, 'wb') as q:
        pickle.dump(tra_seqs_frac, q)





### frac데이터 crop하고 저장

def process_seqs(iseqs, idates):
    out_seqs = []
    out_dates = []
    labs = []
    ids = []
    for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            out_dates += [date]
            ids += [id]
            
    
    
    return out_seqs, out_dates, labs, ids




#%% 폴더 만들기

'''
da_for_sbr
    |-exps
    |   |-experiment1 
    |   |    |-yoochoose
    |   |    |-diginetica
    |   |    |-result_narm_yoochoose
    |   |    |    |-y064
    |   |    |    |-y128
    |   |    |    |-y256
    |   |    |    |-y512
    |   |    |-result_narm_diginetica
    |   |    |    |-d001
    |   |    |    |-d003
    |   |    |    |-d006
    |   |    |    |-d012
    |   |    |-result_srgnn_yoochoose
    |   |    |    |-y064
    |   |    |    |-y128
    |   |    |    |-y256
    |   |    |    |-y512
    |   |    |-result_srgnn_diginetica
    |   |    |    |-d001
    |   |    |    |-d003
    |   |    |    |-d006
    |   |    |    |-d012
    |   |-experiment(n)
    |   |-train_item_views.csv
    |   |-yoochoose-clicks-withHeader.dat
    |-narm_torch
    |-srgnn_torch
    |-main.ipynb
    |-preprocessing.py
    |-simmetric.py
'''



experiment=1


if not os.path.exists(f'exps/experiment{experiment}/yoochoose'):
    os.makedirs(f'exps/experiment{experiment}/yoochoose')
    
if not os.path.exists(f'exps/experiment{experiment}/diginetica'):
    os.makedirs(f'exps/experiment{experiment}/diginetica')
    
if not os.path.exists(f'exps/experiment{experiment}/result_narm_yoochoose'):
    os.makedirs(f'exps/experiment{experiment}/result_narm_yoochoose')
    
if not os.path.exists(f'exps/experiment{experiment}/result_narm_diginetica'):
    os.makedirs(f'exps/experiment{experiment}/result_narm_diginetica')
    
if not os.path.exists(f'exps/experiment{experiment}/result_srgnn_yoochoose'):
    os.makedirs(f'exps/experiment{experiment}/result_srgnn_yoochoose')
    
if not os.path.exists(f'exps/experiment{experiment}/result_srgnn_diginetica'):
    os.makedirs(f'exps/experiment{experiment}/result_srgnn_diginetica')

time.sleep(10)


for f in [1/64, 1/128, 1/256, 1/512]:
  if not os.path.exists(f'exps/experiment{experiment}/result_narm_yoochoose/y{int(1/f):03}'):
    os.makedirs(f'exps/experiment{experiment}/result_narm_yoochoose/y{int(1/f):03}')
  if not os.path.exists(f'exps/experiment{experiment}/result_srgnn_yoochoose/y{int(1/f):03}'):
    os.makedirs(f'exps/experiment{experiment}/result_srgnn_yoochoose/y{int(1/f):03}')

for f in [1/1, 1/3, 1/6, 1/12]:
  if not os.path.exists(f'exps/experiment{experiment}/result_narm_diginetica/d{int(1/f):03}'):
      os.makedirs(f'exps/experiment{experiment}/result_narm_diginetica/d{int(1/f):03}')
  if not os.path.exists(f'exps/experiment{experiment}/result_srgnn_diginetica/d{int(1/f):03}'):
      os.makedirs(f'exps/experiment{experiment}/result_srgnn_diginetica/d{int(1/f):03}')




#%% 원본데이터 불러오기, 전처리


# dataPATH = '/content/drive/MyDrive/015GithubRepos/da_for_sbr/exps'
# expPATH = '/content/drive/MyDrive/015GithubRepos/da_for_sbr/exps'


dataPATH = 'J:/내 드라이브/015GithubRepos/da_for_sbr/exps'
expPATH = 'J:/내 드라이브/015GithubRepos/da_for_sbr/exps'


# 만들 데이터 선택
y_or_d = 'yoochoose'


if y_or_d == 'yoochoose':
    filename = 'yoochoose-clicks-withHeader.dat'  # 원본 데이터 파일
    tra_sess, tes_sess, sess_clicks = prep_yoochoose(dataPATH, filename)
    fracs = [1/64, 1/128, 1/256, 1/512]
    
else:
    filename = 'train-item-views.csv'  # 원본 데이터 파일
    tra_sess, tes_sess, sess_clicks = prep_diginetica(dataPATH, filename)
    fracs = [1/1, 1/3, 1/6, 1/12]

# 원본데이터 전처리(통계량 확인)
print(f'원본데이터 통계량')
tra_ids_ori, tra_dates_ori, tra_seqs_ori, item_dict_ori = obtian_tra(sess_clicks, tra_sess)



#%% frac만큼 떼어내서 train과 test 만들어 저장
# train에 없는 아이템 test에서도 제거

'''
-마지막 아이템 떼서 라벨로
ex) yoochoose 1/128
    -> y128_train.txt : (train sequences, labels)
    -> y128_test.txt : (test sequences, labels)
'''

for f in fracs:

    split_frac = int(len(tra_sess) * f)
    tra_sess_frac = tra_sess[-split_frac:]
    
    print(f'----------------------------------------------')
    print(f'frac : 1/{int(1/f)}')
    tra_ids_frac, tra_dates_frac, tra_seqs_frac, item_dict_frac = obtian_tra(sess_clicks, tra_sess_frac)
    tes_ids, tes_dates, tes_seqs = obtian_tes(sess_clicks, tes_sess, item_dict_frac)
    save_tra_seqs_frac(experiment, y_or_d, f, tra_seqs_frac)

    # frac train
    tr_seqs_frac, tr_dates_frac, tr_labs_frac, tr_ids_frac = process_seqs(tra_seqs_frac, tra_dates_frac)
    te_seqs, te_dates, te_labs, te_ids = process_seqs(tes_seqs, tes_dates)
    tra_frac = (tr_seqs_frac, tr_labs_frac)
    
    # frac test
    tes = (te_seqs, te_labs)
    
    # save
    train_name = f'exps/experiment{experiment}/{y_or_d}/{y_or_d[0]}{int(1/f):03}_train.txt'
    print(f'train_name : {train_name}')
    test_name = f'exps/experiment{experiment}/{y_or_d}/{y_or_d[0]}{int(1/f):03}_test.txt'
    print(f'test_name : {test_name}')
    pickle.dump(tra_frac, open(train_name, 'wb'))
    pickle.dump(tes, open(test_name, 'wb'))


#%% most_sim_d 만들기


experiment=1
y_or_d = 'yoochoose'
if y_or_d == 'yoochoose':
    fracs = [1/64, 1/128, 1/256, 1/512]
else:
    fracs = [1/1, 1/3, 1/6, 1/12]
    
    
for frac in fracs:
    tra_seqs_frac = load_tra_seqs_frac(experiment, y_or_d, frac)
    
    co_mat_win = get_co_matrix_win(tra_seqs_frac, 5, True, True)
    save_mat_as_csr(co_mat_win, 'coo_mat', experiment, y_or_d, frac)
    
    pmi_mat = ppmi(co_mat_win, True)
    save_mat_as_csr(pmi_mat, 'pmi_mat', experiment, y_or_d, frac)
    
    jac_mat = get_jaccard_mat(tra_seqs_frac, 5, diag_freq = True, remove_dup = True)
    save_mat_as_csr(jac_mat, 'jac_mat', experiment, y_or_d, frac)
    
    tan_mat = get_tanimoto(tra_seqs_frac, window_size=5, diag_freq=True, remove_dup=True)
    save_mat_as_csr(tan_mat, 'tan_mat', experiment, y_or_d, frac)
    
    cos_mat_win = get_cosine(co_mat_win)
    save_mat_as_csr(cos_mat_win, 'cos_mat', experiment, y_or_d, frac)
    
    w2v_mat = get_w2v_model(tra_seqs_frac)
    save_mat_as_csr(w2v_mat, 'w2v_mat', experiment, y_or_d, frac)