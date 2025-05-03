from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# إعداد Flask
app = Flask(__name__)
CORS(app)  # السماح بالتواصل مع الواجهة (مثل React)

# تحميل الموديل والبيانات
model = joblib.load("svm_income_model.joblib")  # تأكد إن الملف موجود
df = pd.read_csv("adult.csv")  # تأكد إن الملف موجود
df.replace("?", pd.NA, inplace=True)
df.dropna(inplace=True)

# الأعمدة المستخدمة
categorical_cols = [
    'workclass', 'education', 'marital.status', 'occupation',
    'relationship', 'race', 'sex', 'native.country'
]
numerical_cols = [
    'age', 'fnlwgt', 'education.num',
    'capital.gain', 'capital.loss', 'hours.per.week'
]

# API للتنبؤ بالدخل
@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_input = request.get_json()
        if not user_input:
            return jsonify({"error": "Empty input received"}), 400

        input_df = pd.DataFrame([user_input])

        # التحقق من الأعمدة
        missing_cols = [
            col for col in categorical_cols + numerical_cols if col not in input_df.columns
        ]
        if missing_cols:
            return jsonify({"error": f"Missing columns: {missing_cols}"}), 400

        # التنبؤ بالاحتمالات
        probabilities = model.predict_proba(input_df)[0]
        result = {
            "<=50K": round(probabilities[0] * 100, 2),
            ">50K": round(probabilities[1] * 100, 2)
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API لتحليل البيانات (EDA) - أرقام فقط
@app.route('/eda', methods=['GET'])
def eda():
    try:
        # الإحصائيات العامة
        summary = df.describe(include='all').fillna('').to_dict()

        # توزيع الدخل
        income_distribution = df['income'].value_counts(normalize=True)\
            .mul(100).round(2).to_dict()

        # توزيع الدخل حسب الجنس
        income_by_sex = df.groupby('income')['sex'].value_counts(normalize=True)\
            .unstack().mul(100).round(2).fillna(0).to_dict()

        # توزيع الدخل حسب التعليم
        income_by_education = df.groupby('education')['income'].value_counts(normalize=True)\
            .unstack().mul(100).round(2).fillna(0).to_dict()

        # معاملات الارتباط
        correlation = df.select_dtypes(include=['int64', 'float64'])\
            .corr().round(3).to_dict()

        return jsonify({
            "summary": summary,
            "income_distribution": income_distribution,
            "income_by_sex": income_by_sex,
            "income_by_education": income_by_education,
            "correlation": correlation
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# تشغيل السيرفر
if __name__ == '__main__':
    app.run(debug=True)
