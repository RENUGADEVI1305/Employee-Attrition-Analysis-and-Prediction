import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

st.set_page_config(page_title="Employee Attrition Analysis and Prediction")

st.sidebar.title("My Dashboard")
page=st.sidebar.radio('Visit',["Home", "Project Explanation", "Predicting Employee Attrition", "Predicting Employee Promotion Likelihood",  "Developer Info"])


   # page1 -  Home 

if page == "Home":
    
    st.header("Home")
    st.markdown(" ## Mini project 3:  ")
    st.markdown(" ### Title - Employee Attrition Analysis and Prediction")
    st.image("D:/GUVI/project3/PIC1.jpg", width=300)

  # page2 -  Project Explanation 

elif page == "Project Explanation":
    st.header("Project Explanation")
    #st.image("D:/GUVI/project1/images/pic2.jpg", width=300)
    st.write(""" **Problem Statement:** Employee turnover poses a significant challenge for organizations, 
             resulting in increased costs, reduced productivity, and team disruptions.
              Understanding the factors driving attrition and predicting at-risk employees is critical for effective retention strategies. 
             This project aims to analyze employee data, identify key drivers of attrition, and build predictive models to support 
             proactive decision-making in workforce management. """)
    
    st.write(""" **AIM:** 
             
             1. Predict whether an employee will leave the company (attrition).
             
             2. Predicting Employee Promotion Likelihood.
             
             """)
    
    st.subheader("Dataset")
    df=pd.read_csv("D:/GUVI/project3/employee_attrition_cleaned.csv") 
    st.dataframe(df)

   # page4 -  Predicting Employee Attrition 

elif page == "Predicting Employee Attrition":
    st.header("🧮 Predicting Employee Attrition")
    st.write("Provide employee details below:")
 


    MODEL_PATH = "D:/GUVI/project3/myenv/employee_attrition_pipeline.joblib"

    # ---- Load model and metadata ----
    if os.path.exists(MODEL_PATH):
        try:
            saved = joblib.load(MODEL_PATH)
            pipe = saved.get('pipeline')  # imblearn pipeline
            numeric_features = saved.get('numeric_features', [])
            categorical_features = saved.get('categorical_features', [])
            feature_names = saved.get('feature_names', [])
            st.success(f"✅ Loaded pipeline from {MODEL_PATH}")
        except Exception as e:
            st.error(f"Failed to load pipeline file: {e}")
            st.stop()
    else:
        st.warning(f"No model file found at '{MODEL_PATH}'. Please train & save a model first.")
        st.stop()

    # For inference we'll use the fitted preprocessor + classifier (avoid using SMOTE during predict)
    try:
        preprocessor = pipe.named_steps['preprocessor']
        classifier = pipe.named_steps['classifier']
    except Exception:
        st.error("Saved pipeline does not contain expected steps ('preprocessor' and 'classifier'). Make sure you saved the pipeline with these names.")
        st.stop()

    st.write("Model ready. Provide input as a single employee (manual) or upload CSV with same feature columns.")

    # ---- Helper: preprocess single/batch input ----
    def prepare_input(df_input: pd.DataFrame) -> np.ndarray:
        """
        1) Ensure required columns present (numeric + categorical)
        2) Use preprocessor.transform() to get numeric array ready for classifier
        """
        df = df_input.copy()

        # Check and fill missing expected columns
        expected_cols = list(set(numeric_features + categorical_features))  # set -> list (order not critical here)
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing expected columns: {missing}. Uploaded data must contain these columns: {expected_cols}")

        # Keep only the columns used by preprocessor, in any order (ColumnTransformer selects by name)
        # Preprocessor expects numeric_features and categorical ones (excl OverTime if you used label encoding)
        # So pass the DataFrame as-is (preprocessor will internally pick columns by name)
        X_prepared = preprocessor.transform(df)  # returns numpy array
        return X_prepared

    # ---- Option A: Manual single-row input ----
    st.subheader("Manual input (single employee)")
    with st.form(key='manual_input_form'):
        cols = st.columns(3)
        # numeric inputs
        age = cols[0].number_input("Age", min_value=18, max_value=60, value=30)
        monthly_income = cols[0].number_input("MonthlyIncome", min_value=1000, max_value=20000, value=5000)
        job_satisfaction = cols[1].selectbox("JobSatisfaction (1-4)", [1,2,3,4], index=2)
        years_at_company = cols[1].number_input("YearsAtCompany", min_value=0, max_value=18, value=3)
        distance_from_home = cols[2].number_input("DistanceFromHome", min_value=1, max_value=50, value=10)
        num_companies_worked = cols[2].number_input("NumCompaniesWorked", min_value=0, max_value=9, value=1)
       

        # categorical inputs
        marital_options = ['Single', 'Married', 'Divorced']
        overtime_options = ['Yes', 'No']

        marital = st.selectbox("MaritalStatus", marital_options)
        overtime = st.selectbox("OverTime", overtime_options)
        
        submit_manual = st.form_submit_button("Predict (manual)")

    if submit_manual:
        try:
            input_df = pd.DataFrame([{
                'Age': age,
                'MonthlyIncome': monthly_income,
                'JobSatisfaction': job_satisfaction,
                'YearsAtCompany': years_at_company,
                'DistanceFromHome': distance_from_home,
                'NumCompaniesWorked': num_companies_worked,
                'MaritalStatus': marital,
                'OverTime': overtime
            }])

            X_input = prepare_input(input_df)
            pred = classifier.predict(X_input)
            pred_proba = classifier.predict_proba(X_input)[:, 1] if hasattr(classifier, "predict_proba") else None

            # Map numeric prediction to readable label (we used 1=Yes,0=No)
            pred_label = "Yes" if int(pred[0]) == 1 else "No"
            prob = pred_proba[0] if pred_proba is not None else None

            # 💡 Stylish output card
            
            st.subheader("🎯 Prediction Result")

            if pred_label == "Yes":
               st.markdown(
                f"""
                <div style='background-color:#ffe6e6;padding:20px;border-radius:12px;text-align:center'>
                    <h3 style='color:#cc0000;'>⚠️ High Attrition Risk!</h3>
                    <p style='font-size:18px;'>This employee is <b>likely to leave</b> soon.</p>
                    <p><b>Probability:</b> {prob:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
               st.markdown(
                f"""
                <div style='background-color:#e6ffe6;padding:20px;border-radius:12px;text-align:center'>
                    <h3 style='color:#007700;'>✅ Low Attrition Risk!</h3>
                    <p style='font-size:18px;'>This employee is <b>likely to stay</b> with the company.</p>
                    <p><b>Probability:</b> {prob:.2%}</p>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # ---- Option B: Batch CSV upload ----
    
    st.subheader("Batch prediction (CSV)")
    batch_file = st.file_uploader("Upload CSV with columns matching training features", type=['csv'], key='batch_upload')

    if batch_file is not None:
        try:
            batch_df = pd.read_csv(batch_file)
            st.write("Preview of uploaded data:")
            st.dataframe(batch_df.head())

            if st.button("Run batch prediction"):
                try:
                    X_batch = prepare_input(batch_df)
                    preds = classifier.predict(X_batch)
                    probs = classifier.predict_proba(X_batch)[:, 1] if hasattr(classifier, "predict_proba") else None

                    # attach results
                    batch_results = batch_df.copy()
                    batch_results['Predicted_Attrition'] = ['Yes' if int(p)==1 else 'No' for p in preds]
                    if probs is not None:
                        batch_results['Attrition_Probability'] = probs

                    st.success("✅ Batch prediction completed!")
                    st.dataframe(batch_results)

                    # Provide download link
                    csv = batch_results.to_csv(index=False).encode('utf-8')
                    st.download_button(label="📥 Download predictions CSV", data=csv, file_name='attrition_predictions.csv', mime='text/csv')

                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")

        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")

    st.markdown("---")
    st.caption("📝 **Note:** Uploaded or manual input must contain the same feature columns used during model training.")



  # page5 -  Predicting Employee Promotion Likelihood 

elif page == "Predicting Employee Promotion Likelihood":
    st.header("🏆Predicting Employee Promotion Likelihood") 
    st.write("Provide employee details below to predict their likelihood of promotion.")

     # -------------------------------------------------
     # 📂 Load Saved Model
     # -------------------------------------------------
    MODEL_PATH = "D:/GUVI/project3/myenv/employee_promotion_pipeline.joblib"

    if os.path.exists(MODEL_PATH):
       try:
        saved = joblib.load(MODEL_PATH)
        pipe = saved.get('pipeline')  # imblearn pipeline
        numeric_features = saved.get('numeric_features', [])
        categorical_features = saved.get('categorical_features', [])
        feature_names = saved.get('feature_names', [])
        st.success(f"✅ Loaded pipeline from {MODEL_PATH}")
       except Exception as e:
        st.error(f"Failed to load pipeline file: {e}")
        st.stop()
    else:
      st.warning(f"⚠️ No model file found at '{MODEL_PATH}'. Please train & save a model first.")
      st.stop()

     # Extract steps
    try: 
      preprocessor = pipe.named_steps['preprocessor']
      classifier = pipe.named_steps['classifier']
    except Exception:
      st.error("Saved pipeline missing expected steps ('preprocessor' and 'classifier').")
      st.stop()

    st.write("Model ready. You can predict for a single employee or upload a CSV for batch prediction.")

# -------------------------------------------------
# 🧩 Helper: Input Preparation
# -------------------------------------------------
    def prepare_input(df_input: pd.DataFrame) -> np.ndarray:
       df = df_input.copy()
       expected_cols = list(set(numeric_features + categorical_features))
       missing = [c for c in expected_cols if c not in df.columns]
       if missing:
          raise ValueError(f"Missing expected columns: {missing}. Required columns: {expected_cols}")
       X_prepared = preprocessor.transform(df)
       return X_prepared

# -------------------------------------------------
# 🧍 Manual Prediction
# -------------------------------------------------
    st.subheader("🎯 Manual Input (Single Employee)")
    with st.form(key='manual_form'):
        cols = st.columns(3)

        years_in_role = cols[0].number_input("YearsInCurrentRole", min_value=0, max_value=15, value=3)
        num_companies = cols[0].number_input("NumCompaniesWorked", min_value=0, max_value=10, value=2)
        worklife_balance = cols[1].selectbox("WorkLifeBalance (1-4)", [1,2,3,4], index=2)
        job_level = cols[1].selectbox("JobLevel (1-5)", [1,2,3,4,5], index=1)
        total_work_years = cols[2].number_input("TotalWorkingYears", min_value=0, max_value=30, value=5)
        education = cols[2].selectbox("Education (1-Basic, 5-Doctor)", [1,2,3,4,5], index=2)
        job_satisfaction = cols[0].selectbox("JobSatisfaction (1-4)", [1,2,3,4], index=2)
        salary_hike = cols[1].number_input("PercentSalaryHike", min_value=0, max_value=25, value=5)

        submit_manual = st.form_submit_button("🔍 Predict Promotion")

# -------------------------------------------------
# 🎨 Display Prediction (Manual)
# -------------------------------------------------
    if submit_manual:
      try:
        input_df = pd.DataFrame([{
            'YearsInCurrentRole': years_in_role,
            'NumCompaniesWorked': num_companies,
            'WorkLifeBalance': worklife_balance,
            'JobLevel': job_level,
            'TotalWorkingYears': total_work_years,
            'Education': education,
            'JobSatisfaction': job_satisfaction,
            'PercentSalaryHike': salary_hike
        }])

        X_input = prepare_input(input_df)
        pred = classifier.predict(X_input)
        pred_proba = classifier.predict_proba(X_input) if hasattr(classifier, "predict_proba") else None

        # Extract predicted label and probabilities
        pred_label = "Yes" if int(pred[0]) == 1 else "No"
        prob_yes = pred_proba[0][1] if pred_proba is not None else None
        prob_no = pred_proba[0][0] if pred_proba is not None else None

        

            # 💡 Stylish output card
            
        st.subheader(" 🏆📈 Promotion Prediction Result")

        if pred_label == "Yes":
           st.markdown(f"""
               <div style='background-color:#e6f7ff;padding:20px;border-radius:12px;text-align:center;'>
            <p style='font-size:18px;'>
                <b>Predicted PromotionSoon:</b> 
                <span style='color:#009933;font-weight:bold;'>Yes ✅</span>
            </p>
            <p style='font-size:16px;'>Probability of PromotionSoon (Yes): 
                <b style='color:#009933;'>{prob_yes:.2%}</b>
            </p>
            <p style='font-size:16px;'>Probability of Not Soon (No): 
                <b style='color:#cc0000;'>{prob_no:.2%}</b>
            </p>
                 </div>
             """,
             unsafe_allow_html=True
            )

        else:
            st.markdown(f"""
            <div style='background-color:#fff0f0;padding:20px;border-radius:12px;text-align:center;'>
            <p style='font-size:18px;'>
                <b>Predicted PromotionSoon:</b> 
                <span style='color:#cc0000;font-weight:bold;'>No ❌</span>
            </p>
            <p style='font-size:16px;'>Probability of PromotionSoon (Yes): 
                <b style='color:#009933;'>{prob_yes:.2%}</b>
            </p>
            <p style='font-size:16px;'>Probability of Not Soon (No): 
                <b style='color:#cc0000;'>{prob_no:.2%}</b>
            </p>
            </div>
            """,
            unsafe_allow_html=True
            )
           

      except Exception as e:
        st.error(f"Prediction failed: {e}")

# -------------------------------------------------
# 📂 Batch Prediction
# -------------------------------------------------
    st.subheader("📊 Batch Prediction (Upload CSV)")
    uploaded_file = st.file_uploader("Upload CSV with same feature columns", type=['csv'], key='batch_upload')

    if uploaded_file is not None:
      try:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(batch_df.head())

        if st.button("🚀 Run Batch Promotion Prediction"):
            try:
                X_batch = prepare_input(batch_df)
                preds = classifier.predict(X_batch)
                probs = classifier.predict_proba(X_batch) if hasattr(classifier, "predict_proba") else None

                batch_results = batch_df.copy()
                batch_results['Predicted_PromotionSoon'] = ['Yes' if int(p) == 1 else 'No' for p in preds]

                if probs is not None:
                   batch_results['Promotion_Probability_Yes'] = probs[:, 1]
                   batch_results['Promotion_Probability_No'] = probs[:, 0]

                st.success("✅ Batch Prediction Completed Successfully!")
                st.dataframe(batch_results)

                csv = batch_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Predictions",
                    data=csv,
                    file_name='promotion_predictions.csv',
                    mime='text/csv'
                )
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")

      except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")

    st.markdown("---")
    st.caption("💡 Note: Model predicts the estimated *Years Since Last Promotion*. Lower values imply the employee is likely due for a promotion soon.")

   # page6 -  Developer Info
 
elif page == "Developer Info":
    st.header("Developer Info")
    st.markdown("""
    **Developed by:** T RENUGADEVI 

    **Course:** Data Science                        
    **Skills:** Python, Pandas, EDA, Machine Learning Model Development, Streamlit""", True)

    st.snow()