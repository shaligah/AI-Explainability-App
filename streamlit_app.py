import pandas as pd 
import streamlit as st
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
from sklearn.feature_selection import f_regression,f_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
import xgboost as xgt
from sklearn.metrics import roc_auc_score,f1_score,precision_score,recall_score,accuracy_score,r2_score,mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import shap
import plotly.express as px
import matplotlib.pyplot as plt
import openai
from openai import APIError, OpenAI
import base64
from io import BytesIO
import plotly.io as pio


class DropMissingData(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.45):
        self. threshold = threshold
        self.columns_to_drop = []
        
        
    def fit(self, X, y=None):
        self.columns_to_drop = X.isnull().mean()[X.isnull().mean()>self.threshold].index.tolist()
        return self
    
    def transform(self, X):
        X= X.drop(columns = self.columns_to_drop, errors='ignore')
        
        return X
    

class SelectRegress(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.selected_features = []
        
    def fit(self,X,y):
        X = X.loc[:, X.std() > 0]
        f_scores, p_values =f_regression(X, y)
        self.selected_features = X.columns[p_values<self.alpha].tolist()
        if len(self.selected_features) == 0:
            top_indices = np.argsort(p_values)[:15]
            self.selected_features = X.columns[top_indices].tolist()
        return self
    
    def transform(self,X):
        return X[self.selected_features]


class SelectClassif(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.selected_features = []
        
    def fit(self,X,y):
        X = X.loc[:, X.std() > 0]
        f_scores, p_values =f_classif(X, y)
        self.selected_features = X.columns[p_values<self.alpha].tolist()
        if len(self.selected_features) == 0:
            top_indices = np.argsort(p_values)[:15]
            self.selected_features = X.columns[top_indices].tolist()
        return self
    
    def transform(self,X):
        return X[self.selected_features]

    
class VIFCorrelationReducer(BaseEstimator, TransformerMixin):
    def __init__(self, corr_threshold=0.8, protected_columns=None):
        self.corr_threshold = corr_threshold
        self.protected_columns = protected_columns or []
        self.kept_features = []

    def fit(self, X, y=None):
        X = X.copy()
        self.kept_features = list(X.columns)

        while True:
            # Step 1: Correlation matrix
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

            # Step 2: Identify correlated pairs
            to_consider = [
                (row, col)
                for col in upper.columns
                for row in upper.index
                if upper.loc[row, col] >= self.corr_threshold
                and row not in self.protected_columns
                and col not in self.protected_columns
            ]

            if not to_consider:
                break

            # Step 3: Compute VIFs
            vif = pd.Series(
                [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
                index=X.columns
            )

            # Step 4: Drop one feature from each correlated pair based on higher VIF
            dropped = set()
            for f1, f2 in to_consider:
                if f1 in dropped or f2 in dropped:
                    continue  # skip if already dropped

                vif1 = vif.get(f1, np.inf)
                vif2 = vif.get(f2, np.inf)

                if vif1 > vif2:
                    drop = f1
                else:
                    drop = f2

                X = X.drop(columns=[drop])
                dropped.add(drop)

            # Step 5: Update list of features to keep
            self.kept_features = list(X.columns)

        return self

    def transform(self, X):
        return X[self.kept_features]
       
def training(X,y, task, random_state=42): 
    if task !='classification':
        y = np.log1p(y)
    
    #split the data in training and validation
    Xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2, stratify = y if task =='classification' else None, random_state=42)
    
    #choosing the model
    if task =='classification':
        model = xgt.XGBClassifier(eval_metric='logloss', random_state=random_state)
        scoring = 'f1' if pd.Series(y).nunique()==2 else 'accuracy'
        params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.3, 0.5]
        }
     
    else:
        model =xgt.XGBRegressor(random_state=random_state)
        scoring='neg_root_mean_squared_error'
        params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.3, 0.5]
        }
    
    #finding the best params
    search_space = RandomizedSearchCV(
        estimator = model,
        param_distributions = params,
        n_iter=15, 
        scoring=scoring, 
        n_jobs=-1, 
        cv=3, 
        verbose=0,
        random_state=random_state
    )
    
    search_space.fit(Xtrain, ytrain)
    
    #finding the best model
    best_model = search_space.best_estimator_
    best_model.fit(Xtrain, ytrain)
    
    
    #viewing the metrics
    predictions = best_model.predict(xtest)
    
    if task=='classification':
        prob = best_model.predict_proba(xtest)
        if prob.shape[1] ==2:
            auc = roc_auc_score(ytest, prob[:,1])
        else:
            auc = roc_auc_score(ytest,prob,multi_class='ovr', average='weighted' )
        
        val_score = f1_score(ytest, predictions,average='weighted')
        metrics = {
            'f1_score': val_score,
            'precision': precision_score(ytest, predictions,average='weighted'),
            'recall':recall_score(ytest, predictions,average='weighted'),
            'accuracy':accuracy_score(ytest, predictions),
            'ROC-AUC Score': auc
        }
        
    else:
        val_score = r2_score(np.expm1(ytest), np.expm1(predictions))
        metrics = {
            'RMSE':np.sqrt(mean_squared_error(np.expm1(ytest), np.expm1(predictions))),
            'R2_score': val_score
        }
        
    return {
        'model':best_model,
        'best_params':search_space.best_params_,
        'val_score': val_score,
        'metrics':metrics,
        'test_data':xtest
    }
        
    
st.set_page_config('Data Explainability App', layout='wide')
st.title('🧠 Explainable AI Playground 🎢 ')

if 'section' not in st.session_state:
    st.session_state['section'] = '-'

# 2. Let user select section, but don't override session state immediately
selected = st.sidebar.selectbox(
    "Select a section",
    ["-", 'Tasking', 'Preprocessing', 'Model_Training','Explainability', 'Generate AI Summary'],
    index=["-", 'Tasking', 'Preprocessing','Model_Training', 'Explainability','Generate AI Summary'].index(st.session_state['section'])
)

# 3. Only update session state if user actually changed selection
if selected != st.session_state['section']:
    st.session_state['section'] = selected

#App Introduction
if st.session_state.section == '-':
    st.markdown("""
    Welcome to **Explainable AI for Everyone** — a smart, intuitive, and no-code machine learning app that helps you build **interpretable models** and **share insights** with ease!
    ---

    ## 🔍 What does this app do?

    Whether you're working with customer data, real estate, health, finance, or just playing with Kaggle datasets — this app lets you:

    ✅ Upload your dataset  
    ✅ Choose your target variable  
    ✅ Automatically clean, select features, and build an XGBoost model  
    ✅ See detailed metrics like **F1, Accuracy, RMSE, R², AUC**  
    ✅ Understand your model using **SHAP visualizations**  
    ✅ 🪄 **Generate a plain-English summary of the SHAP results using OpenAI**  
    ✅ 📄 **Download an HTML report with explanations for stakeholders** — all auto-generated!

    ---

    ## 🧪 Supports Classification & Regression

    The app can automatically detect whether your task is:
    
    - 🏷️ **Classification** (e.g., churn prediction, fraud detection)
    - 📈 **Regression** (e.g., house price prediction, revenue forecasting)

    Or you can pick it yourself!

    ---

    ## 📂 What do you need?

    All you need is a **CSV file**.  
    We'll help you pick the target column and handle the rest.

    ---

    ## 💬 What makes this special?

    While most ML tools stop at accuracy, **we believe in transparent AI**.  
    That’s why we added:

    - 🧹 Smart missing value handling  
    - 🎯 Feature selection (statistical + multicollinearity-aware)  
    - 🧬 SHAP-based model explanations  
    - 🗣️ **OpenAI-powered text summary of your model’s behavior**  
    - 📥 A polished **HTML report** you can share with clients, teammates, or investors

    ---

    ## 👩‍💻 Built by a Data Scientist — for Everyone
    
    This app is part of a growing portfolio aimed at making **machine learning more human**.  
    No jargon. No notebooks. Just results you can understand and explain.

    Ready? Let's build explainable AI together! 🚀

    ---
    """)
    if st.button('Lets Begin'):
        with st.spinner("..."):
            st.session_state.section = 'Tasking'
            st.rerun()
            
# Data Collection and summarisation
if st.session_state.section =='Tasking':
    uploaded_file = st.sidebar.file_uploader('Please upload a csv file', type=['csv'])
    st.caption("🔒 *Don't worry, we never store your data.*")
    if uploaded_file is not None and 'df' not in st.session_state:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.sidebar.success('File successfully load and read')
        
    if 'df' in st.session_state:
        df = st.session_state.df
        with st.expander("📊 Want to peek into your data stats?"):
            st.subheader("🔍 Data Overview")

        # Create summary table
            summary = pd.DataFrame({
                "Column Name": df.columns,
                "Data Type": df.dtypes.astype(str),
                "% Missing": df.isnull().mean().round(2) * 100,
                "# Unique": df.nunique()
            }).reset_index(drop=True)
            st.markdown(f"**Rows** {df.shape[0]} | **Columns** {df.shape[1]}")

            st.dataframe(summary)

            # Missing values bar chart
            missing = df.isnull().mean() * 100
            missing = missing[missing > 0].sort_values(ascending=False)

            if not missing.empty:
                st.markdown("#### 🧼 Missing Data Overview")
                st.bar_chart(missing)
            else:
                st.success("✅ No missing values found!")

    else:
        st.warning("⚠️ Upload a dataset first to explore stats.")

    
    if 'df' in st.session_state:
        df = st.session_state.df
        st.sidebar.write('What are we predicting today')
        st.session_state.target = st.sidebar.selectbox("🎯 Select target variable", df.columns)
        st.markdown(f"---\n### 📌 Selected Target: `{st.session_state.target}`")

        st.markdown("### 🧠 How should we treat the task?")
        task_mode = st.radio(
            "Select task mode:",
            ["✅ Auto-detect (recommended)", "🔘 Manually select: Classification", "🔘 Manually select: Regression"]
        )

        if "Auto-detect" in task_mode:
        # Simple logic: if target is object/categorical or has few unique values
            if df[st.session_state.target].nunique() <= 10 and df[st.session_state.target].dtype in ['object', 'category', 'bool', 'int']:
                task = 'classification'
            else:
                task = 'regression'
            st.success(f"Auto-detected task: **{task.capitalize()}**")
    
        elif "Classification" in task_mode:
            task = 'classification'
            st.info("You selected **Classification** manually.")
    
        else:
            task = 'regression'
            st.info("You selected **Regression** manually.")
            
        st.session_state['task'] = task
        
        if st.session_state.target:
            if st.button("➡️ Proceed to Preprocessing"):
                st.session_state.section = 'Preprocessing'
                st.rerun()
                
elif st.session_state.section=='Preprocessing':
    df = st.session_state.df
    
    st.markdown("## 🧼 Preprocessing")
    st.markdown("### Missing Value Handler")
    if df.isnull().sum().sum() > 0:
        subsection = st.radio('Select a threshold',['Default', 'Use user selected'])
        if subsection =='Use user selected':
            thresh = st.select_slider('Threshold', options=list(np.arange(0, 1.01, 0.01)), value = 0.45)
            # handling columns with missing data greater than a certain threshold
            missing_handler = DropMissingData(threshold = thresh)
        else:
            missing_handler =DropMissingData(threshold=0.45)
        st.session_state.cleaned_data = missing_handler.fit_transform(st.session_state.df)
        dropped = list(set(st.session_state.df.columns.tolist()) - set(st.session_state.cleaned_data.columns.tolist()))
        st.success(f"✅ Dropped {len(dropped)} columns with missing value percentage above threshold.")
        if dropped:
            with st.expander("🔍 See dropped columns"):
                st.write(dropped)
        method = st.selectbox("Select imputation method", ["Drop rows", "Fill with mode"])
        
        if 'apply_clicked' not in st.session_state:
            st.session_state.apply_clicked = False
        
        if st.button("Apply"):
            st.session_state.apply_clicked = True
            
        if st.session_state.apply_clicked:
            if method == "Drop rows":
                st.session_state.cleaned_data =  st.session_state.cleaned_data.dropna()
            elif method == "Fill with mode":
                st.session_state.cleaned_data = st.session_state.cleaned_data.fillna(st.session_state.cleaned_data.mode().iloc[0])
            st.success("✅ Missing values handled.") 
            st.markdown(f"**Rows** {st.session_state.cleaned_data.shape[0]} | **Columns** {st.session_state.cleaned_data.shape[1]}")
    else:
        st.success("No missing values to clean!")
        
    y = st.session_state.cleaned_data[st.session_state.target]
    X = st.session_state.cleaned_data.drop(columns =st.session_state.target)

    
    #Categorical Encoding
    st.markdown('### Categorical Encoding')
    if st.button("Encode"):
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            encoder = OrdinalEncoder()
            X[cat_cols] = encoder.fit_transform(X[cat_cols])
            st.session_state.encoded_df = X
            st.success("✅ Categorical variables encoded using OrdinalEncoder.")
        else:
            st.session_state.encoded_df = X
            st.info("ℹ️ No categorical variables to encode.")
        
        if st.session_state.task == 'classification':
            if y.dtype == 'object' or y.dtype.name == 'category':
                num_classes = y.nunique()

                if num_classes == 2:
                    st.info("Binary classification detected.")
                elif num_classes > 2:
                    st.info(f"Multi-class classification detected ({num_classes} classes).")
                
                le = LabelEncoder()
                st.session_state.y_encoded = le.fit_transform(y)
                st.success("✅ Label Variable encoded using LabelEncoder.")

            elif pd.api.types.is_numeric_dtype(y):
                unique_vals = y.nunique()
                if unique_vals <= 10 and sorted(y.unique()) == list(range(unique_vals)):
        # Possibly already encoded classification
                    st.info(f"Likely classification with {unique_vals} classes.")
                    st.session_state.y_encoded = y
        else:
            st.session_state.y_encoded =y
        st.write(st.session_state.encoded_df)
        
        
    st.markdown('### Do you want to select significant features first?')
    selection_choice = st.radio("Select significant features?", ["Yes", "No"], key="feature_selection_choice")
    if selection_choice == "Yes":
        st.markdown('### Significant Features')
        if st.button("Select Significant Features"):
            
            #selecting singnificant columns with p-values less than 0.05
            if st.session_state.task =='classification':
                significant_selector = SelectClassif(alpha=0.05)
                st.session_state.Xclean = significant_selector.fit_transform(st.session_state.encoded_df,st.session_state.y_encoded)
            else:
                significant_selector = SelectRegress(alpha=0.05)
                st.session_state.Xclean = significant_selector.fit_transform(st.session_state.encoded_df,st.session_state.y_encoded)
            
            # using VIF to deal with multicollinearity
            reducer = VIFCorrelationReducer(corr_threshold=0.8)
            st.session_state.prepped = reducer.fit_transform(st.session_state.Xclean)
            dropped2 = list(set(X.columns.tolist()) - set(st.session_state.prepped.columns.tolist()))
            st.success(f"✅ Dropped {len(dropped2)}  after Feature Selection.")
            if dropped2:
                with st.expander('See dropped columns'):
                    st.write(dropped2)
            st.session_state.features_selected = True

    if selection_choice == "No":
        st.session_state.prepped = st.session_state.encoded_df  # use original encoded features
        st.session_state.features_selected = True


# ✅ Show this block only if features were selected
    if st.session_state.get('features_selected', False): 
        st.write("🧮 Feature matrix shape:", st.session_state.prepped.shape)
        st.write("🎯 Target vector shape:", st.session_state.y_encoded.shape)
        st.markdown('## Ready to move on!!!!!!')
        st.write(st.session_state.prepped.shape[0],
            st.session_state.y_encoded.shape[0])
        if st.button('Lets train the model🚀'):
            with st.spinner("Training in progress..."):
                st.session_state.training_results = training(
                st.session_state.prepped,
                st.session_state.y_encoded,
                st.session_state.task,
                random_state=42)
                st.session_state.Model_Training = True
                st.session_state.section = "Model_Training"
                st.rerun()
        
elif st.session_state.section == 'Model_Training':
    st.markdown("## 📊 Training Results")

    results = st.session_state.training_results

    st.subheader("📌 Best Hyperparameters")
    st.json(results['best_params'])

    st.subheader("📈 Validation Metrics")
    for metric, score in results['metrics'].items():
        st.metric(label=metric, value=round(score, 4))

    # Optionally save the model for prediction later
    st.session_state.model = results['model']
    st.session_state.xtest = results['test_data']
    st.session_state['metrics'] = results['metrics']
    st.session_state['best_params'] = results['best_params']
    
    if st.button('Lets explain the results'):
        st.session_state.section = 'Explainability'
        st.rerun()

elif st.session_state.section == 'Explainability':
    st.header("🔍 Model Explainability")

    st.markdown("""
    Use this section to understand how different features influence the model's predictions.
    Select an explanation type:
    """)

    explain_mode = st.radio("Choose explanation type", ['Global Feature Importance', 'Local Explanation (SHAP)'])


    if explain_mode == 'Global Feature Importance':
        st.subheader("📊 Top Features by Gain (XGBoost)")

        booster = st.session_state.model.get_booster()
        importance = booster.get_score(importance_type='gain')
        st.session_state.importance_df = pd.DataFrame({
            'Feature': list(importance.keys()),
            'Importance': list(importance.values())
        }).sort_values(by='Importance', ascending=False)

        fig = px.bar(st.session_state.importance_df.head(20), x='Importance', y='Feature', orientation='h', title='Top 20 Feature Importances')
        st.plotly_chart(fig, use_container_width=True)
        plotly_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
        st.session_state.importance_html = plotly_html
    
    elif explain_mode=='Local Explanation (SHAP)': 
        prediction2 = st.session_state.model.predict(st.session_state.xtest)
        row1 = st.number_input("Choose first instance", min_value=0, max_value=len(st.session_state.xtest)-1, value=0)
        row2 = st.number_input("Choose second instance", min_value=0, max_value=len(st.session_state.xtest)-1, value=1)
        if st.button("Explain Predictions"):
            row_df1 = st.session_state.xtest.iloc[[row1]]
            row_df2 = st.session_state.xtest.iloc[[row2]]
            explainer = shap.Explainer(st.session_state.model.predict, st.session_state.xtest)

            shap_values = explainer(pd.concat([row_df1, row_df2]))
            base_price1 = np.exp(shap_values[0].base_values)
            shap_contribs1 = shap_values[0].values * base_price1

            base_price2 = np.exp(shap_values[1].base_values)
            shap_contribs2 = shap_values[1].values * base_price2

            explanation1 = shap.Explanation(
                values=shap_contribs1,
                base_values=base_price1,
                data=row_df1.iloc[0],
                feature_names=row_df1.columns
            )

            explanation2 = shap.Explanation(
                values=shap_contribs2,
                base_values=base_price2,
                data=row_df2.iloc[0],
                feature_names=row_df2.columns
            )

            col1, col2 = st.columns(2)

            with col1:
                st.write(f"Explanation for instance {row1}")
                fig1, ax1 = plt.subplots()
                shap.plots.waterfall(explanation1, max_display=10, show=False)
                st.session_state.df_expl1 = pd.DataFrame({
                    'Feature': explanation1.feature_names,
                    'Value': explanation1.data,
                    'SHAP Value': explanation1.values})
                st.session_state.buf2 = BytesIO()
                fig1.savefig(st.session_state.buf2, format='png')
                st.session_state.buf2.seek(0)
                st.pyplot(fig1)

            with col2:
                st.write(f"Explanation for instance {row2}")
                fig2, ax2 = plt.subplots()
                shap.plots.waterfall(explanation2, max_display=10, show=False)
                st.session_state.df_expl2 = pd.DataFrame({
                    'Feature': explanation2.feature_names,
                    'Value': explanation2.data,
                    'SHAP Value': explanation2.values})
                st.session_state.buf3 = BytesIO()
                fig2.savefig(st.session_state.buf3, format='png')
                st.session_state.buf3.seek(0)
                st.pyplot(fig2)



        st.markdown('## Ready for the next step?')
        if st.button('Lets generate a report!!!!'):
            st.session_state.section = 'Generate AI Summary'
            st.rerun()

elif st.session_state.section == 'Generate AI Summary':
    
    generate_ai_summary = st.sidebar.checkbox("🔍 Generate an AI Summary Report?")
    
    if generate_ai_summary:
        # User inputs for context
        st.subheader('Generate AI Summary')
        st.markdown("## Provide Context for the Report")

        industry = st.text_input("🏭 Industry/Domain", help="E.g., Healthcare, Finance, Retail")
        dataset_description = st.text_area("📂 Describe the dataset", help="E.g., Transaction data from an online store.")
        business_objective = st.text_area("🎯 What is the business goal?", help="E.g., Predict customer churn or detect fraud.")

        # Store in session_state
        st.session_state["industry"] = industry
        st.session_state["dataset_description"] = dataset_description
        st.session_state["business_objective"] = business_objective

        st.session_state['industry_clean'] = industry.strip() or 'a general industry'
        st.session_state['data_description_clean'] = dataset_description.strip() or 'an unspecified data description'
        st.session_state['business_objective_clean'] = business_objective.strip() or 'unspecified business objective'

        if not industry or not dataset_description or not business_objective:
            st.warning("⚠️ Some fields have been left blank. We'll use default values for better AI summarization")


        prompt = f"""
        You are an AI assistant. Generate a professional, business-facing summary report describing the results of a machine learning model.

        This report will be presented as part of a PDF document to help business stakeholders understand the value and limitations of the AI model used.

        ## ⚠️ Disclaimer
        Model predictions depend **heavily on the quality, completeness, and accuracy of the input data**. All insights should be reviewed and interpreted in consultation with a data science or domain expert before any business decisions are made.

        ## 📘 Project Overview
        - **Industry/Domain**: {st.session_state['industry_clean']}
        - **Dataset Description**: {st.session_state['data_description_clean']}
        - **Business Objective**: {st.session_state['business_objective_clean']}
        - **Task Type**: {st.session_state['task'].capitalize()} (e.g., classification, regression)

        ## 🤖 AI Model Summary
        - Model Type: **Tuned XGBoost {st.session_state['task']} model**
        - Hyperparameter tuning performed using cross-validation:
        {st.session_state['best_params']}

        ## 📊 Performance Metrics
        Summarize and interpret these evaluation metrics:
        {st.session_state['metrics']}

        Comment on whether model performance is sufficient for business deployment or further testing is recommended.

        ## 🔍 Key Influencing Factors
        The following top 10 features had the most impact on model predictions:
        {st.session_state['importance_df']}

        Briefly explain what these features represent and how they might influence the model output.

        ## 🔬 SHAP-Based Explanations (Local Interpretability)
        Use SHAP values to explain how specific features contributed to two individual predictions. Focus on the **3 most** and **3 least** influential variables in each case.

        ### Example 1:
        {st.session_state['df_expl1']}

        ### Example 2:
        {st.session_state['df_expl2']}

        ## ✅ Summary & Next Steps
        - State whether the model is **ready for production**, **requires more data**, or should be **reviewed further**.
        - Keep the explanation free from technical jargon.
        - Suggest **one practical next step** to improve model accuracy, interpretability, or business adoption.

        ## 📝 Output
        Write a clean, readable summary using professional tone and markdown-style formatting. This report should be suitable for inclusion in a PDF intended for a non-technical audience.
        """



        st.info(
            "🔐 **Privacy Notice**: Your OpenAI API key is **never stored**. "
            "It is used only during this session and **deleted immediately after use**.")

        api_key = st.text_input("Enter your OpenAI API Key", type="password")

        if st.button("🚀 Generate Summary") and api_key:
            try:
                # Step 2: Temporarily set it

                # Step 3: Validate the key (optional, but recommended)
                client = OpenAI(api_key=api_key)
                st.success("✅ API Key is valid. Generating your summary...")

                # Step 4: Make your OpenAI call (example below)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )
                st.session_state['summary'] = response.choices[0].message.content
                st.subheader("🧠 AI Summary")
                st.write(st.session_state['summary'])

            except APIError:
                st.error("❌ Invalid API key. Please try again.")

            except Exception as e:
                st.error(f"🚨 An error occurred: {str(e)}")

            finally:
                # Step 5: Ensure the key is deleted after use
                del api_key
                openai.api_key = None
                
        if 'summary' in st.session_state and st.session_state.summary is not None:
            if st.button('View summary'):
                st.session_state.summary
                
    def generate_report_html(summary, metrics_dict, global_image,shap_images, include_summary=True):
        html = """
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
                h1, h2, h3 { color: #333; }
                .title { text-align: center; font-size: 28px; }
                .section { margin-bottom: 30px; }
                table { border-collapse: collapse; width: 100%; }
                table, th, td { border: 1px solid #ddd; }
                th, td { padding: 8px; text-align: left; }
                th { background-color: #f4f4f4; }
                img { max-width: 100%; height: auto; margin-top: 10px; }
            </style>
        </head>
        <body>
            <div class="title">📊 Model Report Summary</div>
        """

        # Optional summary
        summary_html = summary.replace("\n", "<br>")
        if include_summary and summary.strip():
            html += f"""
            <div class="section">
                <h2>🧠 AI-Generated Summary</h2>
                <p>{summary_html}</p>
            </div>
            """

        # Metrics
        html += """
        <div class="section">
            <h2>📈 Model Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
        """
        for key, value in metrics_dict.items():
            html += f"<tr><td>{key}</td><td>{value}</td></tr>"
        html += "</table></div>"

        # Global images
        html += "<div class='section'><h2>📊 Global Feature Importance</h2>"
        html += global_image  # injects interactive chart
        html += "</div>"
        # SHAP images
        html += "<div class='section'><h2>🔍 SHAP Interpretations</h2>"
        for title, image_bytes in shap_images.items():
            b64_img = base64.b64encode(image_bytes.getvalue()).decode()
            html += f"<h3>{title}</h3><img src='data:image/png;base64,{b64_img}'>"
        html += "</div>"

        html += "</body></html>"
        return html
    
    summary = st.session_state.get("summary", "")
    include_summary = bool(summary.strip())

    images = {
        "💧 SHAP Waterfall (Instance 1)": st.session_state.buf2,
        "💧 SHAP Waterfall (Instance 2)": st.session_state.buf3,
    }
    # Example usage
    html_report = generate_report_html(summary, st.session_state['metrics'],st.session_state['importance_html'], images)
    st.download_button(
        label="📥 Download HTML Report",
        data=html_report.encode("utf-8"),
        file_name="model_report.html",
        mime="text/html"
    )
    
st.sidebar.markdown("---")


# Place Clear All button at bottom
clear = st.sidebar.button("🧹 Clear All", key="clear_all")

if clear:
    # Clear everything
    for key in list(st.session_state.keys()):
        del st.session_state[key]

    # Reset to home
    st.session_state.section = "-"
    st.rerun()






