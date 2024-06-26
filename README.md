# simplex-method-py
輪読会で読んだ文献[1] の p.33--36「単体表による単体法」を実装した．

## 開発環境
- Python: 3.10.2
- numpy: 1.22.2

## 動かし方
1. `$ git clone ...`
1. `$ cd simplex-method-py/src/`
1. `$ python3 main.py < in/feasible-konno.in`

### 期待する出力
```txt
tableau:
[[0. 1. 3. 2. 4. 0. 0. 0.]
 [4. 0. 1. 1. 2. 1. 0. 0.]
 [5. 0. 2. 0. 2. 0. 1. 0.]
 [7. 0. 2. 1. 3. 0. 0. 1.]]
pivot (row, col)=(2, 2), pivot=2.0
[[-7.5  1.   0.   2.   1.   0.  -1.5  0. ]
 [ 1.5  0.   0.   1.   1.   1.  -0.5  0. ]
 [ 2.5  0.   1.   0.   1.   0.   0.5  0. ]
 [ 2.   0.   0.   1.   1.   0.  -1.   1. ]]
pivot (row, col)=(1, 3), pivot=1.0
[[-10.5   1.    0.    0.   -1.   -2.   -0.5   0. ]
 [  1.5   0.    0.    1.    1.    1.   -0.5   0. ]
 [  2.5   0.    1.    0.    1.    0.    0.5   0. ]
 [  0.5   0.    0.    0.    0.   -1.   -0.5   1. ]]
最適解に到達しました．
tableau:
[[-10.5   1.    0.    0.   -1.   -2.   -0.5   0. ]
 [  1.5   0.    0.    1.    1.    1.   -0.5   0. ]
 [  2.5   0.    1.    0.    1.    0.    0.5   0. ]
 [  0.5   0.    0.    0.    0.   -1.   -0.5   1. ]]
最適値 z = 21/2
最適解 x = {'x(1)': '5/2', 'x(2)': '3/2', 'x(6)': '1/2', 'x(3)': 0, 'x(4)': 0, 'x(5)': 0}
status=<Status.OPTIMAL: 'optimal'>
```

## instanceファイルのフォーマット
max型の等式標準形を想定する．
```txt
m
-z_0, -z, x_1 , ..., x_n , ..., x_(n+m)
 b_1, -z, a_11, ..., a_1n, ..., a_1(n+m)
 ...
 b_m, -z, a_m1, ..., a_mn, ..., a_m(n+m)
```

実際は，-z_0や-zは固定なので，係数部分だけを埋めればよい

```txt
m
0  , -1, x_1 , ...
b_1, 0 , a_11, ...
...
b_1, 0 , a_m1, ...
```

## 参考文献
1. 今野浩. 線形計画法. 日科技連. 1987.
1. 梅谷俊治. しっかり学ぶ数理最適化: モデルからアルゴリズムまで. 講談社. 2020.
1. Shun-Chen Niu. Introduction to Operations Research. The University of Texas at Dallas. <https://personal.utdallas.edu/~scniu/OPRE-6201/OPRE-6201_Course-Content.html>. 2024/05閲覧.
