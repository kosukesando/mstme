# %%
import enum
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


def check_over_thr(data, thr, n):
    """
  閾値を超えるデータ数がn以下になるように閾値を調整し、調整済みの配列を返す(あまり使用しない)

  Args:
      data (list): データ
      thr (int): 閾値
      n (int): データ数

  Returns:
      pot (list): 閾値を超えたデータ(これはn個以下になっている)
      thr (int): 更新後の閾値

  """
    thr_new = thr
    num = np.count_nonzero(data > thr_new)
    while True:
        if num < n:
            break
        thr_new += thr / 20
        num = np.count_nonzero(data > thr_new)
    pot = np.array(data[data > thr_new])
    return pot, thr_new


def calc_gl(n, xi, sgm, pot, error, thr):
    """
  各格子点周りの尤度を計算する（まだ対数尤度にはできていない)

  Args:
      n (int): 格子点の数の平方根(n*n個の格子点を作るので)
      xi (list): ξ
      sgm (list): σ
      pot (list): 閾値を超えたデータ
      error (list): 誤差(要素数は1 or n)
      thr (int): 閾値

  Returns:
      prob (2darray): prob[i, j]は格子点(i, j)周りの尤度

  """
    prob = np.zeros((n, n))
    prob2 = np.zeros((n, n))
    # for i, _xi in enumerate(xi):  # ξの添字
    #     for j, _sgm in enumerate(sgm):
    #         if _xi == 0:
    #             cdf1 = 1 - np.exp(
    #                 -(np.clip(pot - thr - error, a_min=0, a_max=None)) / _sgm
    #             )
    #             cdf2 = 1 - np.exp(
    #                 -(np.clip(pot - thr + error, a_min=0, a_max=None)) / _sgm
    #             )
    #         else:
    #             cdf1 = 1 - np.clip(
    #                 1 + _xi * np.clip(pot - thr - error, a_min=0, a_max=None) / _sgm,
    #                 a_min=0,
    #                 a_max=None,
    #             ) ** (-1 / _xi)
    #             cdf2 = 1 - np.clip(
    #                 1 + _xi * np.clip(pot - thr + error, a_min=0, a_max=None) / _sgm,
    #                 a_min=0,
    #                 a_max=None,
    #             ) ** (-1 / _xi)
    #         cdf = 10 ** len(pot) * np.prod(cdf2 - cdf1)
    #     prob[i, j] = cdf

    for i, _xi in enumerate(xi):  # ξの添字
        for j in range(n):  # σの添字
            cdf = 10 ** 100  # 最初に大きい値にしておく
            xi_ = xi[i]
            sgm_ = sgm[j]
            for k in range(len(pot)):
                pot_ = pot[k]
                error_ = error[k]
                # ξが0かどうかでCDFの式が変わる
                if xi_ == 0:
                    cdf1 = 1 - np.exp(-(max(0, pot_ - error_ - thr)) / sgm_)
                    cdf2 = 1 - np.exp(-(max(0, pot_ + error_ - thr)) / sgm_)
                else:
                    cdf1 = 1 - max(0, 1 + xi_ * max(0, pot_ - error_ - thr) / sgm_) ** (
                        -1 / xi_
                    )
                    cdf2 = 1 - max(0, 1 + xi_ * max(0, pot_ + error_ - thr) / sgm_) ** (
                        -1 / xi_
                    )
                cdf = cdf * (cdf2 - cdf1)
            prob2[i, j] = cdf
    # print("calcgl", np.max(prob2 - prob))
    # return prob
    return prob2


def set_param(min_par, max_par, n):
    """
  パラメータの設定（最小値から最大値までをn分割する)

  Args:
      min_par (int): 最小値
      max_par (int): 最大値(この値は含まない)
      n (int): 分割数

  Returns:
      params (list): 設定されたパラメータ

  """
    return np.linspace(min_par, max_par, n)


def lwm_gpd(data, error, thr, period, RP):
    from time import time

    """
  PPD(POSTERIOR PREDICTIVE DISTRIBUTION)の描画と、信頼区間ごとの尤度を描画する

  Args:
      data (list): データ
      error (list): 誤差
      thr (int): 閾値
      period (int): 期間
      RP (list): 再現期間(要素数は1 or n)

  Returns:
      描画する(x, ppd)
      描画する(ξ, logσ)
      Fval(list): 再現期待値(RPの要素数個分だけ出てくる)
  """
    pot, thr = check_over_thr(data, thr, 150)

    # 誤差は1つだけしか与えられなくても、1*nの配列に変換する
    if len(error) == 1:
        for i in range(len(pot) - 1):
            error.append(error[0])

    # 格子点の粒度
    N = 40
    # ξとσをセット
    xi_search = set_param(-5, 5, N)
    sgm_search = 10 ** set_param(np.log10(0.01), np.log10(10), N)
    prob_search = calc_gl(N, xi_search, sgm_search, pot, error, thr)
    # 最大尤度
    max_p = np.max(prob_search)
    print("最大尤度: ", max_p)
    print(
        "最大尤度を取るインデックス: ", np.unravel_index(np.argmax(prob_search), prob_search.shape)
    )
    # 最小尤度(これ以下の値は除外する→ξとσの範囲を絞るため)
    min_p = max_p / 10 ** 5
    # min_pよりも大きい値を取るindexのリスト
    # axis=1の方向（sigma）で1要素でも大きい値を取ればTrue
    xi_sub = xi_search[np.max(prob_search > min_p, axis=1)]
    print("xi_sub: ", xi_sub)
    max_xi = xi_sub.max()
    min_xi = xi_sub.min()
    if min_xi == max_xi:
        min_xi -= 0.3
        max_xi += 0.3
    print("max_xi, min_xi: ", max_xi, min_xi)
    # min_pよりも大きい値を取るindexのリスト
    # axis=0の方向（xi）で1要素でも大きい値を取ればTrue
    sgm_sub = sgm_search[np.max(prob_search > min_p, axis=0)]
    print("sgm_sub: ", sgm_sub)
    max_sgm = sgm_sub.max()
    min_sgm = sgm_sub.min()
    if min_sgm == max_sgm:
        min_sgm = min_sgm / 3
        max_sgm = max_sgm * 3
    print("max_sgm, min_sgm: ", np.log10(max_sgm), np.log10(min_sgm))

    # 粒度
    N = 200
    # パラメータの範囲を絞って、粒度を細かくした
    xi = set_param(min_xi, max_xi, N)
    sgm = 10 ** set_param(np.log10(min_sgm), np.log10(max_sgm), N)
    prob = calc_gl(N, xi, sgm, pot, error, thr)

    # 尤度の合計
    pp = np.sum(prob)
    # 尤度の最大値
    pm = np.max(prob)

    # Xの粒度
    N_x = 1001

    # # PPDの計算
    # x = np.linspace(0, 50, N_x, endpoint=True)
    # ppd = np.zeros(len(x))
    # # x は配列で、cdxも配列(サイズはN_x)
    # for i in range(N):  # ξの添字
    #     for j in range(N):  # σの添字
    #         if prob[i, j] > 0:
    #             # ξの正負で場合わけ
    #             if xi[i] == 0:
    #                 cdx = [max(0, 1 - np.exp((data - thr) / sgm[j])) for data in x]
    #             else:
    #                 cdx = [
    #                     1 - max(0, 1 + xi[i] * (data - thr) / sgm[j]) ** (-1 / xi[i])
    #                     for data in x
    #                 ]
    #             for k in range(len(x)):
    #                 ppd[k] = ppd[k] + cdx[k] * prob[i, j] / pp

    # # 閾値以下のところではPPDは0にする
    # xthr = np.where(x > thr)[0][0]
    # for k in range(xthr):
    #     ppd[k] = 0

    # # PPDのプロット
    # plt.plot(x, ppd)
    # plt.title("POSTERIOR PREDICTIVE DISTRIBUTION")
    # plt.xlabel("Hs[m]")
    # # plt.savefig("PPD.png")
    # # プロット表示(設定の反映)
    # plt.show()

    # # 再現期待値の算出(ppd > 1 - 1 / (RP * num) / period となるような最小のxを見つけて、それが再現期待値になる)
    # RPV_num = len(RP)
    # Fval = []
    # for i in range(RPV_num):
    #     RPV_index = len(x) - 1  # 更新されない場合はxの最後の要素が出力されるように設定
    #     for j in range(len(x)):
    #         if ppd[j] > 1 - 1 / (RP[i] * len(pot) / period):  # 条件を満たすxがあれば、それが再現期待値
    #             RPV_index = j
    #             Fval.append(x[RPV_index])
    #             print(f"{RP[i]}年再現期待値: {x[RPV_index]}")
    #             break

    sum = 1
    sum_prob = np.zeros((N, N))
    # 全ての格子点に対して、累積尤度的なものを計算する
    for _ in range(N * N):  # N*N回ループを回して, 全てのprob[i, j]に対して累積の尤度？てきなものを計算する
        max_value = np.max(prob) / pp
        max_index = np.unravel_index(np.argmax(prob), prob.shape)  # (1, 2)のような形で帰ってくる
        sum -= max_value
        prob[max_index[0], max_index[1]] = 0
        sum_prob[max_index[0], max_index[1]] = sum

    # 等高線の描画
    log_sgm = np.log10(sgm)
    X, Y = np.meshgrid(xi, log_sgm)
    Z = np.array([[sum_prob[i, j] for i in range(N)] for j in range(N)])
    print(Z)
    fig, ax = plt.subplots()
    # ax = fig.add_subplot(111, xlim=(-0.8, 0.2), ylim=(0, 1.5))
    cntr = ax.contour(X, Y, Z, levels=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9], colors="black")
    ax.clabel(cntr)
    # ax.set_aspect("equal")
    ax.set_title(f"Thr = {thr}")
    ax.set_xlabel("ξ")
    ax.set_ylabel("logσ")
    # plt.savefig("HPD.png")
    plt.show()

    # return Fval


# %%
if __name__ == "__main__":
    sample_data = [
        8.36,
        7.02,
        6.94,
        6.85,
        6.74,
        6.20,
        5.92,
        5.68,
        5.57,
        5.42,
        5.34,
        5.10,
        5.09,
        4.95,
        4.81,
        4.77,
        4.63,
        4.61,
        4.41,
        4.34,
        4.11,
    ]
    lwm_gpd(data=sample_data, error=[0.001], thr=4.0, period=21, RP=[20, 50, 100])


# %%
