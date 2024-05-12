import numpy as np
import fractions
import enum


"""
NOTE
- 各関数でどの変数を参照／操作するのかを明示するため，class変数にはあまり持たせず毎回引数で渡している
- 1段階の単体法なので，初期解（原点）が実行不能の場合は対応していない
- 最小添字規則 (Bland's rule) によってpivotを選択しているので，収束は遅いが巡回はしない
"""


class Status(enum.Enum):
    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"


class LPSolver:
    def __init__(self) -> None:
        self.z_offset = 2  # 0: z0, 1: z
        self.EPS = 1e-6
        return

    def is_infeasible(self, tableau: np.ndarray) -> bool:
        """ 基底解が実行不可能かどうか
        """
        # 非基底変数はゼロ
        # 基底変数が非負制約を満たさないならダメ
        return (tableau[:, 0] < 0).any()

    def is_optimal(self, tableau: np.ndarray) -> bool:
        """ 最適解の確認
        """
        # （最大化問題の場合）
        # 目的関数の係数が全て非正なら最適解
        return (tableau[0, self.z_offset:] <= 0).all()

    def get_pivot_column(self, tableau: np.ndarray) -> int:
        # （最大化問題の場合）
        # 目的関数の係数が正の変数から選ぶ
        pivot_candidates, *_ = np.where(tableau[0, self.z_offset:] > 0)
        # 最小添字規則
        pivot_col = min(pivot_candidates) + self.z_offset
        return pivot_col

    def is_unbounded(self, tableau: np.ndarray, pivot_col: int) -> bool:
        """ 非有界の確認
        """
        return (tableau[1:, pivot_col] <= 0).all()

    def get_pivot_row(self, tableau: np.ndarray, pivot_col: int) -> int:
        # ゼロ除算を防ぐ（np.nan に置き換え）
        ratio = np.divide(tableau[:, 0], tableau[:, pivot_col], where=tableau[:, pivot_col] != 0.0)
        # このあとminを取るので，無視するために np.nan を np.inf に変換
        ratio = np.nan_to_num(ratio, nan=np.inf)
        # 増加量の最大値
        # \frac{1}{inf} (1e-300とか) を無視するために，0.0じゃなくてEPS以上
        theta = min(ratio[ratio > self.EPS])
        pivot_candidates, *_ = np.where(ratio == theta)
        # 退化の検知
        is_degenerate = len(pivot_candidates) > 1
        if is_degenerate:
            print("退化が発生しました．まあ最小添え字規則を使ってるので気にしなくて良いですよ．")
        # 最小添字規則
        pivot_row = min(pivot_candidates)
        return pivot_row

    def pivot_operation(self, tableau: np.ndarray, pivot_row: int, pivot_col: int) -> np.ndarray:
        # pivot を 1 にする
        tableau[pivot_row, :] /= tableau[pivot_row, pivot_col]
        # 目的関数部分の操作
        tableau[0, :] -= tableau[0, pivot_col] * tableau[pivot_row, :]
        # 制約部分の操作
        h, _w = tableau.shape
        for row in range(1, h):
            if row == pivot_row:
                continue
            tableau[row, :] -= tableau[row, pivot_col] * tableau[pivot_row, :]
        return tableau

    def get_optimal_value(self, tableau: np.ndarray) -> float:
        """ 最適値の取得
        """
        # -z の値なので，符号を反転するのを忘れない
        # float型（0.333...とか）を Fraction型（1/3）にキャストして見た目を分かりやすく
        return fractions.Fraction(-tableau[0, 0]).limit_denominator()

    def get_optimal_solution(self, tableau: np.ndarray) -> np.ndarray:
        """ 最適解の取得
        """
        solution = dict()
        # 基底解
        B, *_ = np.where(tableau[0, self.z_offset:] == 0.0)
        tableau_B = list(map(lambda x: x + self.z_offset, B))
        display_B = list(map(lambda x: x + 1, B))  # 1-indexed
        for t_b, d_b in zip(tableau_B, display_B):
            r = np.where(tableau[:, t_b] == 1.0)
            r, *_ = r  # unpack
            r, *_ = r  # unpack
            solution[f"x({d_b})"] = fractions.Fraction(tableau[r, 0]).limit_denominator().__str__()
        # 非基底解
        N, *_ = np.where(tableau[0, self.z_offset:] < 0.0)
        display_N = list(map(lambda x: x + 1, N))  # 1-indexed
        for n in display_N:
            solution[f"x({n})"] = 0
        return solution

    def solve(self, tableau: np.ndarray) -> Status:
        print("tableau:")
        print(tableau)
        if self.is_infeasible(tableau):
            print("初期解が実行不能です．二段階単体法を検討してください．")
            return Status.INFEASIBLE
        while True:
            if self.is_optimal(tableau):
                print("最適解に到達しました．")
                break
            pivot_col = self.get_pivot_column(tableau)
            if self.is_unbounded(tableau, pivot_col):
                print("非有界です．")
                return Status.UNBOUNDED
            pivot_row = self.get_pivot_row(tableau, pivot_col)
            print(f"pivot (row, col)=({pivot_row}, {pivot_col}), pivot={tableau[pivot_row, pivot_col]}")
            tableau = self.pivot_operation(tableau, pivot_row, pivot_col)
            print(tableau)
        print("tableau:")
        print(tableau)
        print("最適値 z =", self.get_optimal_value(tableau))
        print("最適解 x =", self.get_optimal_solution(tableau))
        return Status.OPTIMAL


def build_tableau() -> np.ndarray:
    # format:
    # 今野浩. 線形計画法. 日科技連. の形式に従う
    # -z0, -z, x1, ..., x_{n+m}
    # max型の等式標準形を入力とする

    # 今野浩. 線形計画法. 日科技連. p.35
    tableau = np.array(
        # np.array() で np.ndarray型の配列を作成する．
        # np.ndarray() で生成してもエラーにはならないけど推奨されていないらしい
        # https://linus-mk.hatenablog.com/entry/numpy_array_ndarray_difference
        [
            [0, -1, 3, 2, 4, 0, 0, 0],
            [4, 0, 1, 1, 2, 1, 0, 0],
            [5, 0, 2, 0, 2, 0, 1, 0],
            [7, 0, 2, 1, 3, 0, 0, 1],
        ],
        dtype=float
    )

    # 梅谷俊治. しっかり学ぶ数理最適化: モデルからアルゴリズムまで. 講談社. p.27
    # tableau = np.array(
    #     [
    #         [0, -1, 1, 2, 0, 0, 0],
    #         [6, 0, 1, 1, 1, 0, 0],
    #         [12, 0, 1, 3, 0, 1, 0],
    #         [10, 0, 2, 1, 0, 0, 1],
    #     ],
    #     dtype=float
    # )

    return tableau


def read_tableau() -> np.ndarray:
    # format:
    # 今野浩. 線形計画法. 日科技連. の形式に従う
    # -z0, -z, x1, ..., x_{n+m}
    # max型の等式標準形を入力とする

    m = int(input())  # number of const.
    tableau = np.array(
        [list(map(int, input().split(","))) for _ in range(m + 1)],
        dtype=float
    )
    return tableau


if __name__ == '__main__':
    # tableau = build_tableau()
    tableau = read_tableau()
    solver = LPSolver()
    status = solver.solve(tableau)
    print(f"{status=}")
