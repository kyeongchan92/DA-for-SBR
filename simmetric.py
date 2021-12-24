from collections import defaultdict
import numpy as np
from gensim.models import Word2Vec
from scipy.sparse import csr_matrix
import pickle
from itertools import combinations

# 유사도지표 만들기 함수

### 1 동시발생행렬 만들기

#### apr (출현횟수가 중심에 놓일 경우만 카운)

def get_co_matrix_apr(sessions, window_size, diag_freq = False, remove_dup = False):
    # window_size: 본인 코드에서 lr_n_each와 동일한 값
    d = defaultdict(int)
    item_set = set()
    item_freq = defaultdict(int)
    for sess in sessions:
        # iterate over item
        for i in range(len(ses)):
            item = sess[i]
            item_freq[item] += 1
            item_set.add(item)  # add to item set
            next_item = sess[i+1 : i+1+window_size]
            if remove_dup: # 특정 item pair가 context 내에 존재하는지만 확인하는 경우
                next_item = set(next_item) 
            for ni in next_item:
                item_set.add(ni)
                key = tuple(sorted([item, ni]))
                d[key] += 1
    
    # formulate the dictionary into dataframe
    item_set = sorted(item_set) # sort item
    n_item = len(item_set)
    co_mat = np.zeros((n_item+1,n_item+1), dtype=np.int64)
    for key, value in d.items():
        co_mat[key[0],key[1]] = value
        co_mat[key[1],key[0]] = value

    if diag_freq:
        for key, value in item_freq.items():
            co_mat[key, key] = value
            
    return co_mat

# co_mat_apr = get_co_matrix_apr(tra_seqs_frac, 5, True, True)  # appear frequency


#### 윈도우 (출현횟수가 윈도우별로 카운팅)

def get_co_matrix_win(sessions, window_size, diag_freq = False, remove_dup = False):
    # window_size: 본인 코드에서 lr_n_each와 동일한 값
    d = defaultdict(int)
    item_set = set()
    item_freq = defaultdict(int)
    for sess in sessions:
        # iterate over item
        for i in range(len(sess)):
            item = sess[i]
            item_set.add(item)  # add to item set
            next_item = sess[i+1 : i+1+window_size]
            if remove_dup: # 특정 item pair가 context 내에 존재하는지만 확인하는 경우
                next_item = set(next_item) 
            if next_item:
              for ni in next_item:
                  item_set.add(ni)
                  key = tuple(sorted([item, ni]))
                  d[key] += 1
            # 윈도우 내 출현횟수 세기
            window = sess[max(0, i-window_size):i+1+window_size]
            windowset = set(window)
            for wi in windowset:
              item_freq[wi] += 1
    
    # formulate the dictionary into dataframe
    item_set = sorted(item_set) # sort item
    n_item = len(item_set)
    co_mat = np.zeros((n_item+1,n_item+1), dtype=np.int64)
    for key, value in d.items():
        co_mat[key[0],key[1]] = value
        co_mat[key[1],key[0]] = value
        
    if diag_freq:
        for key, value in item_freq.items():
            co_mat[key, key] = value
            
    return co_mat

# co_mat_win = get_co_matrix_win(tra_seqs_frac, 5, True, True)



#%% PPMI
def ppmi(C, verbose=False, eps=1e-8):
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0
    
    for i in range(C.shape[0]):
      for j in range(C.shape[1]):
          pmi = np.log2(C[i, j] * N / (S[j]*S[i]) + eps)  # np.log2(0)이 음의 무한대가 되는 것을 막기위해 eps 사용함
          M[i, j] = max(0, pmi)
            
      if i % 3000 == 0:
        print(f'{i / len(C) * 100} % 완료')
    
    return M

# start = time.time()  # 시작 시간 저장
# ppmi_mat = ppmi(co_mat_win, True)
# print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
  

#%% 3 jaccard 만들기(window)

def get_jaccard_mat(sessions, window_size, diag_freq = False, remove_dup = False):
  # window_size: 본인 코드에서 lr_n_each와 동일한 값
  d = defaultdict(int)
  item_set = set()
  item_freq = defaultdict(int)
  for sess in sessions:
      # iterate over item
      for i in range(len(sess)):
          item = sess[i]
          item_set.add(item)  # add to item set
          
          window = sess[max(0, i-window_size):i+1+window_size]
          windowset = set(window)
          
          # 윈도우 내 출현횟수 세기
          for wi in windowset:
            item_freq[wi] += 1
          
          # 윈도우 내 모든 조합
          cmb = list(combinations(windowset, 2))
          for pair in cmb:
            key = tuple(sorted(pair))
            d[key] += 1
          
              
  
  # formulate the dictionary into dataframe
  item_set = sorted(item_set) # sort item
  n_item = len(item_set)
  win_co_mat = np.zeros((n_item+1,n_item+1), dtype=np.int64)
  for key, value in d.items():
      win_co_mat[key[0],key[1]] = value
      win_co_mat[key[1],key[0]] = value
      
  if diag_freq:
      for key, value in item_freq.items():
          win_co_mat[key, key] = value


  M = np.zeros_like(win_co_mat, dtype=np.float32)

  for i in range(len(win_co_mat)):
    for j in range(len(win_co_mat)):
      if i == 0 or j == 0:
        M[i][j] = 0
      else:
        M[i][j] = win_co_mat[i][j] / (win_co_mat[i][i] + win_co_mat[j][j] - win_co_mat[i][j])

    if i % 1000 == 0:
      print(f'{i / len(win_co_mat) * 100} % 완료')

  return M





# jac_mat = get_jaccard_mat(tra_seqs_frac, 5, diag_freq = True, remove_dup = True)

#%% 4 Tanimoto 만들기


'''두 아이템이 모두 자기 자신과밖에 등장하지 않았을 경우 nan이 나온다
ex
[11, 11]과 [9544, 9544]
이 경우 0 / (0 + 0 - 0)

S[i] == 0일 경우 자기자신과만 나타난 경우이다. 즉 다른 아이템과 동시발생하는 경우 없는 경우이다.
'''

def get_tanimoto(sessions, window_size, diag_freq = False, remove_dup = False):

  # window_size: 본인 코드에서 lr_n_each와 동일한 값
  d = defaultdict(int)
  item_set = set()
  item_freq = defaultdict(int)
  for sess in sessions:
      # iterate over item
      for i in range(len(sess)):
          item = sess[i]
          item_set.add(item)  # add to item set
          
          window = sess[max(0, i-window_size):i+1+window_size]
          windowset = set(window)
          
          # 윈도우 내 출현횟수 세기
          for wi in windowset:
            item_freq[wi] += 1
          
          # 윈도우 내 모든 조합
          cmb = list(combinations(windowset, 2))
          for pair in cmb:
            key = tuple(sorted(pair))
            d[key] += 1
          
              
  
  # formulate the dictionary into dataframe
  item_set = sorted(item_set) # sort item
  n_item = len(item_set)
  win_co_mat = np.zeros((n_item+1,n_item+1), dtype=np.int64)
  for key, value in d.items():
      win_co_mat[key[0],key[1]] = value
      win_co_mat[key[1],key[0]] = value
      
  if diag_freq:
      for key, value in item_freq.items():
          win_co_mat[key, key] = value

  M = np.zeros_like(win_co_mat, dtype=np.float32)
  S = np.sum(win_co_mat, axis=0) - win_co_mat.diagonal()

  cnt = 0
  for i in range(len(win_co_mat)):
    for j in range(len(win_co_mat)):
      if (S[i] + S[j] - win_co_mat[i][j]) <= 0:  # 
        M[i][j] = 0
      else:
        M[i][j] = win_co_mat[i][j] / (S[i] + S[j] - win_co_mat[i][j])
      cnt += 1

      if cnt % (int((len(win_co_mat) ** 2)//10)) == 0:
        print(f'{cnt / (len(win_co_mat) ** 2) * 100:.2f}% 완료')

  return M
  



# tanimoto_mat = get_tanimoto(tra_seqs_frac, window_size=5, diag_freq=True, remove_dup=True)

#%% 5 cosine 만들기(윈도우)

def get_cosine(co_mat):
  M = np.zeros_like(co_mat, dtype=np.float32)

  cnt = 0
  for i in range(1, len(co_mat)):
    for j in range(1, len(co_mat)):
      M[i][j] = co_mat[i][j] / np.sqrt(co_mat[i][i] * co_mat[j][j])
      cnt += 1

      if cnt % (int((len(co_mat) ** 2)//10)) == 0:
        print(f'{cnt / (len(co_mat) ** 2) * 100:.2f}% 완료')

  return M

# cosine_mat_win = get_cosine(co_mat_win)

#%% 6 Word2vec 만들기

def get_w2v_model(tra_seqs_frac, nof_items):
  str_tra_seqs = []
  for s in tra_seqs_frac:
      str_tra_seqs.append(list(map(str, s)))
      
  model = Word2Vec(sentences=str_tra_seqs, size=100, window=5, min_count=1, workers=4, sg=0)

  mat = np.zeros((nof_items+1, nof_items+1))
  for i in range(1, nof_items+1):
    for j in range(1, nof_items+1):
      mat[i][j] = model.wv.similarity(str(i), str(j))

  return mat



#%%

# 유사도행렬 csr로 변환 후 저장
def save_mat_as_csr(matrix, matname, experiment, y_or_d, frac):
    csr_mat = csr_matrix(matrix)
    filename = f'exps/experiment{experiment}/{y_or_d}/{y_or_d[0]}{int(1/frac):03}_csr_{matname}.pkl'
    print(f'저장파일 : {filename}')
    with open(filename, 'wb') as f:
        pickle.dump(csr_mat, f)

# csr 유사도행렬 로드
def load_sim_mat(matname, experiment, y_or_d, frac):
    filename = f'exps/experiment{experiment}/{y_or_d}/{y_or_d[0]}{int(1/frac):03}_csr_{matname}.pkl'
    print(f'로드 유사도행렬 : {filename}')
    with open(filename, 'rb') as f:
        csr_matrix = pickle.load(f)
    matrix = csr_matrix.toarray()
    
    return matrix


#%% most_sim_d 만들기

#%%

def find_most_sim(sim_mat, item):

    # co-occurrence와 PPMI, jaccard 용
    mostsimitem = sim_mat[item].argsort()[-1]
    if mostsimitem == item:
        mostsimitem = sim_mat[item].argsort()[-2]
        
    # 첫번째 아니면 두번재로 바꿨는데 상호작용이 0이다? 자기 자신이랑만 상호작용있다.
    # 그러면 자기 자신으로 한다.
    if sim_mat[item][mostsimitem] == 0:
        mostsimitem = item
                
  
    return mostsimitem

#%% get most_sim_dictionary : 가장 유사한 아이템만 저장한 딕셔너리

def make_save_msd(simmetabbr, experiment, y_or_d, frac):
    
    sim_mat = load_sim_mat(f'{simmetabbr}_mat', experiment, y_or_d, frac)
    nof_items = len(sim_mat)-1
    
    d = {}
    for i in range(1, nof_items+1):
        m_sim_i = find_most_sim(sim_mat, i)
        if simmetabbr == 'coo':  # co-occurrence normalition, 자신 출현횟수로 나눔
            d[i] = (m_sim_i, sim_mat[i][m_sim_i] / sim_mat[i][i])
        else:
            d[i] = (m_sim_i, sim_mat[i][m_sim_i])    
    
    filename = f'exps/experiment{experiment}/{y_or_d}/{y_or_d[0]}{int(1/frac):03}_{simmetabbr}_msd.pkl'
    print(f'filename : {filename}')
    with open(filename, 'wb') as f:
        pickle.dump(d, f)

#%% most sim dict 불러오기

def load_msd(simmetabbr, experiment, y_or_d, frac):
  with open(f'exps/experiment{experiment}/{y_or_d}/{y_or_d[0]}{int(1/frac):03}_{simmetabbr}_msd.pkl', 'rb') as f:
    most_sim_d = pickle.load(f)

  return most_sim_d