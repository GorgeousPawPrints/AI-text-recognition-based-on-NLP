import joblib
from preprocessing.text import *



# 加载保存的模型
model = joblib.load('./model/text_classification_model.pkl')
# 示例文本
text ="The rapid advancement of artificial intelligence has revolutionized various industries. AI technology demonstrates exceptional capabilities in data analysis, enabling organizations to optimize decision-making processes. For instance, machine learning algorithms can identify patterns in large datasets that were previously undetectable by human analysts. Furthermore, these advancements have led to significant improvements in healthcare diagnostics, where AI systems achieve accuracy rates exceeding 95% in detecting early-stage diseases. However, the widespread adoption of AI raises ethical concerns regarding job displacement and data privacy. Experts emphasize the necessity of establishing regulatory frameworks to ensure responsible development. Ultimately, balancing technological progress with societal well-being remains a critical challenge for policymakers worldwide."
# 先对文本预处理
text = remove_tags(text)
# 去除符号干扰
text = remove_punc(text)
# 过滤掉对文本分析无实质贡献的常见词
text = remove_stopwords(text)
# 1是AI 0是人
# 执行预测
prediction = model.predict([text])
proba = model.predict_proba([text])
print(f"预测结果: {prediction[0]}, 是人写的概率为{proba[0][0]:.3f}, 是AI写的概率为{proba[0][1]:.3f}")  # 输出0或1

# texts = [
#     "服务态度很差，非常不满意",
#     "质量超出预期，物超所值",
#     "物流速度一般，希望改进"
# ]
#
# predictions = model.predict(texts)
# print(f"批量预测结果: {predictions}")  # 输出数组形式结果

