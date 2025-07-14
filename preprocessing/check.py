import pandas as pd

if __name__ == '__main__':
    # 读取数据时保留原始空值标记
    df = pd.read_csv('../data_set/preprocessed_AI_Human.csv', keep_default_na=False)
    initial_rows = len(df)
    # 步骤1：缺失值检测
    print("缺失值统计:")
    print(df.isnull().sum())  # 显示每列缺失值数量
    print("\n缺失值分布:")
    print(df.isnull().sum() / len(df) * 100)  # 显示缺失值百分比
    # 检查是否存在空字符串
    empty_string_mask = (df['text'] == '')
    empty_strings = (df['text'] == '').sum()
    print(f"空字符串数量: {empty_strings}")

    # 方法2：使用drop方法（更直观）
    df = df.drop(df[empty_string_mask].index)

    print(f"清洗后空字符串数量: {(df['text'] == '').sum()}")

    # 步骤2：删除包含缺失值的行
    df_clean = df.dropna(axis=0, how='any')  # 删除任何含缺失值的行
    cleaned_rows = len(df_clean)

    print(f"删除操作结果: ")
    print(f"原始行数: {initial_rows} → 清洗后行数: {cleaned_rows}")
    print(f"删除比例: {(initial_rows - cleaned_rows) / initial_rows * 100:.1f}%")

    # 步骤3：验证清洗结果
    assert df_clean.isnull().sum().sum() == 0, "仍存在缺失值！"
    print("\n验证通过：数据已无缺失值")

    # df.to_csv('../data_set/preprocessed_AI_Human.csv',
    #           index=False,  # 不保存行索引
    #           encoding='utf-8',  # 保持原始编码
    #           sep=',',  # 使用原始分隔符
    #           header=True)  # 保留列标题