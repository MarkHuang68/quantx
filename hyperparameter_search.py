# 檔案: hyperparameter_search.py

import numpy as np
import itertools
import math
import random

class SearchIterator:
    """
    通用參數迭代器：根據 Grid 或 Random 模式，生成特徵和模型參數的組合。
    
    支援格式: 
    1. [v1, v2, v3, ...] (離散值列表) (formate_types = discrete)
    2. [start, end, step] (區間步長，類似 range()) (formate_types = range)

    formate_type default auto, auto 陣列長度3自動變步進
    """
    
    def __init__(self, search_space: dict, search_type: str = 'grid', n_iter: int = None, format_types: dict = None):
        self.search_type = search_type.lower()
        self.param_names = list(search_space.keys())
        
        # 1. 解析所有參數範圍
        self._param_values = self._generate_all_values(search_space, format_types or {})
        
        # 2. 建立完整的網格 (Grid)
        self.full_grid = list(itertools.product(*self._param_values.values()))
        self.total_runs = len(self.full_grid)
        
        # 3. 根據模式建立迭代器
        if self.search_type == 'grid':
            self.iterator = iter(self.full_grid)
        elif self.search_type == 'random' and n_iter is not None:
            n_samples = min(n_iter, self.total_runs)
            self.random_samples = random.sample(self.full_grid, n_samples)
            self.iterator = iter(self.random_samples)
            self.total_runs = n_samples
        else:
            raise ValueError("search_type 必須是 'grid' 或 'random'，且 'random' 必須指定 n_iter。")

        print(f"--- 尋參迭代器初始化 ({search_type.upper()} 模式) ---")
        print(f"總共執行 {self.total_runs} 次訓練。")

    def _generate_range(self, start, end, step):
        """ 生成浮點數或整數範圍，並確保包含終點值。 """
        if isinstance(step, int) and step > 0:
            # 適用於整數步長：[2, 18, 6] -> [2, 8, 14]
            sequence = list(range(start, end + step, step))
        else:
            # 適用於浮點數步長：[0.01, 0.05, 0.01]
            sequence = [round(x, 8) for x in np.arange(start, end + step, step)]
        
        # 確保終點被涵蓋（如果步長不精確或設定不便）
        if sequence and sequence[-1] != end and start <= end:
            sequence.append(end)
        
        return sequence

    def _generate_all_values(self, search_space, format_types):
        """ 解析字典，將 [start, end, step] 轉換為值列表。"""
        param_values = {}
        for name, values in search_space.items():
            fmt = format_types.get(name, 'auto')
            if fmt == 'range' or (fmt == 'auto' and isinstance(values, list) and len(values) == 3):
                # 判斷為 [start, end, step] 格式
                param_values[name] = self._generate_range(values[0], values[1], values[2])
            else:
                # 判斷為離散列表格式 [v1, v2, ...]
                param_values[name] = values
        return param_values

    def next_config(self) -> dict:
        """ 獲取下一組參數組合 (Keys 是參數名)。"""
        try:
            param_tuple = next(self.iterator)
            # 將 tuple 轉換為字典
            return dict(zip(self.param_names, param_tuple))
        except StopIteration:
            return None

    def __iter__(self):
        return self
    
    def __next__(self):
        result = self.next_config()
        if result is None:
            raise StopIteration
        return result
    
    def get_total_runs(self):
        return self.total_runs

# --- (*** 用法範例：這就是您在 train_price_model.py 中所做的 ***) ---
if __name__ == '__main__':
    
    TEST_SPACE = {
        'max_depth': [3, 5, 7],                      # 離散值 (獨立項目)
        'macd_fast': [6, 18, 6],                     # 步長區間 (6, 12, 18)
        'learning_rate': [0.01, 0.05, 0.02],         # 浮點數步長 (0.01, 0.03, 0.05)
        'bbands_p': [2, 20, 8]                       # 步長不精確，但終點被強制包含 (2, 10, 18, 20)
    }

    # 1. 網格搜索模式 (測試所有組合)
    iterator = SearchIterator(TEST_SPACE, search_type='grid')
    print("\n--- 網格搜索結果 ---")
    for i, config in enumerate(iterator):
        print(f"Run {i+1}: {config}")
        
    # 2. 隨機搜索模式 (只測試 5 次)
    iterator_random = SearchIterator(TEST_SPACE, search_type='random', n_iter=5, format_types={'max_depth': 'discrete'})
    print("\n--- 隨機搜索結果 ---")
    for i, config in enumerate(iterator_random):
        print(f"Run {i+1}: {config}")