from sklearn.feature_selection import SelectKBest, f_classif


def k_best_feature_selection(X, Y, knum):
    # 使用f_classif评分函数进行特征选择
    k_best_selector = SelectKBest(f_classif, k=knum)
    k_best_selector.fit_transform(X, Y)

    # 获取选择的特征的01矩阵
    selected_features = k_best_selector.get_support()

    return selected_features

