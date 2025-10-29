import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pandas.plotting import parallel_coordinates
import numpy as np

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="💉Diabetes Data Visualization", layout="wide")

st.title("💉 Diabetes  Data Visualization Dashboard")
st.markdown("### Diabetes Data Visualization & Insights using Machine Learning")

# ------------------- LOAD DATA -------------------
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    return df.head(200)

df = load_data()

# ------------------- SIDEBAR -------------------
st.sidebar.title("📊 Navigation")
option = st.sidebar.radio("Choose Option", ["Dataset Overview", "Data Visualization", "Algorithm Performance"])

# ------------------- 1️⃣ DATASET OVERVIEW -------------------
if option == "Dataset Overview":
    st.subheader("📘 Dataset Overview (First 200 Rows)")
    st.dataframe(df)
    st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.markdown("**Missing Values:**")
    st.write(df.isnull().sum())

    st.markdown("**Data Types:**")
    st.write(df.dtypes)

# ------------------- 2️⃣ DATA VISUALIZATION -------------------
elif option == "Data Visualization":
    st.subheader("📊 Exploratory Data Analysis")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Distributions", 
        "📦 Boxplots", 
        "🔗 Correlations", 
        "📉 Trends & Relationships",
        "🧠 Advanced Visualizations"
    ])

    # ----------- Tab 1: Distributions -----------
    with tab1:
        st.markdown("### Outcome Distribution")
        fig1, ax1 = plt.subplots()
        df['Outcome'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#90ee90','#ffcccb'], ax=ax1)
        ax1.set_ylabel('')
        st.pyplot(fig1)

        feature = st.selectbox("Select a numerical feature to view distribution:", df.columns[:-1])
        fig2, ax2 = plt.subplots()
        sns.histplot(df[feature], kde=True, color='skyblue', ax=ax2)
        st.pyplot(fig2)

    # ----------- Tab 2: Boxplots -----------
    with tab2:
        st.markdown("### Feature Boxplots (Detect Outliers)")
        feature_box = st.selectbox("Select feature for Boxplot:", df.columns[:-1])
        fig3, ax3 = plt.subplots()
        sns.boxplot(x=df[feature_box], color='lightcoral', ax=ax3)
        st.pyplot(fig3)

        st.markdown("#### Multiple Boxplots")
        fig4, ax4 = plt.subplots(figsize=(10, 4))
        sns.boxplot(data=df.drop('Outcome', axis=1), palette='Set2', ax=ax4)
        plt.xticks(rotation=45)
        st.pyplot(fig4)

    # ----------- Tab 3: Correlations -----------
    with tab3:
        st.markdown("### Correlation Heatmap")
        fig5, ax5 = plt.subplots(figsize=(7, 5))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax5)
        st.pyplot(fig5)

        st.markdown("### Pairplot (Feature Relationships)")
        st.info("⚠️ Note: Pairplot may take a few seconds on 200 rows.")
        sns.pairplot(df, hue='Outcome', diag_kind='kde', palette='husl')
        st.pyplot(plt.gcf())

    # ----------- Tab 4: Trends & Relationships -----------
    with tab4:
        st.markdown("### Relationship Between Two Features")
        x_feature = st.selectbox("Select X-axis feature:", df.columns[:-1], index=1)
        y_feature = st.selectbox("Select Y-axis feature:", df.columns[:-1], index=5)
        fig6, ax6 = plt.subplots()
        sns.scatterplot(data=df, x=x_feature, y=y_feature, hue='Outcome', palette='coolwarm', s=70, ax=ax6)
        st.pyplot(fig6)

        st.markdown("### Trend of Glucose or BMI by Age")
        feature_line = st.selectbox("Select feature for trend analysis:", ['Glucose', 'BMI'])
        fig7, ax7 = plt.subplots()
        sns.lineplot(data=df.sort_values('Age'), x='Age', y=feature_line, color='green', ax=ax7)
        st.pyplot(fig7)

    # ----------- Tab 5: Advanced Visualizations -----------
    with tab5:
        st.markdown("### 🌈 Advanced Visualization Gallery")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📊 Bar Chart")
            fig = px.bar(df, x='Age', y='Glucose', color='Outcome', title="Age vs Glucose")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### 📈 Line Chart")
            fig = px.line(df.sort_values('Age'), x='Age', y='BMI', color='Outcome', title="BMI Trend by Age")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### 🫧 Bubble Chart")
            fig = px.scatter(df, x='Age', y='Glucose', size='BMI', color='Outcome', hover_name='Age', title="Age vs Glucose (Bubble Size=BMI)")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### 🧩 Tree Map")
            fig = px.treemap(df, path=['Outcome'], values='Glucose', color='BMI', title="Tree Map by Outcome")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### 🧬 Parallel Coordinates")
            fig, ax = plt.subplots(figsize=(8, 4))
            parallel_coordinates(df[['Pregnancies','Glucose','BMI','Age','Outcome']], 'Outcome', colormap='cool')
            st.pyplot(fig)

        with col2:
            st.markdown("#### 🍩 Donut Chart")
            donut = df['Outcome'].value_counts()
            fig = go.Figure(data=[go.Pie(labels=donut.index, values=donut.values, hole=0.5)])
            fig.update_traces(marker=dict(colors=['#4CAF50', '#FF7043']))
            fig.update_layout(title="Donut Chart of Outcome")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### 📊 Gauge Chart (Accuracy Demo)")
            value = np.random.uniform(70, 95)
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=value,
                title={'text': "Model Accuracy (%)"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "green"}}
            ))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### 🔶 Hexbin Plot")
            fig, ax = plt.subplots()
            hb = ax.hexbin(df['Age'], df['Glucose'], gridsize=25, cmap='coolwarm')
            plt.colorbar(hb, ax=ax)
            ax.set_xlabel('Age')
            ax.set_ylabel('Glucose')
            ax.set_title('Hexbin: Age vs Glucose')
            st.pyplot(fig)

            st.markdown("#### 🎻 Violin Plot")
            fig, ax = plt.subplots()
            sns.violinplot(x='Outcome', y='BMI', data=df, palette='muted', ax=ax)
            st.pyplot(fig)

# ------------------- 3️⃣ ALGORITHM PERFORMANCE -------------------
elif option == "Algorithm Performance":
    st.subheader("⚙️ Machine Learning Model Performance")

    if st.button("🚀 Run Algorithms"):
        X = df.drop("Outcome", axis=1)
        y = df["Outcome"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=500),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(n_estimators=100)
        }

        results = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[name] = acc

            st.write(f"### 🔹 {name}")
            st.write(f"**Accuracy:** {acc:.2f}")
            st.text(classification_report(y_test, y_pred))

            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f"{name} - Confusion Matrix")
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
            st.markdown("---")

        # Comparison chart
        st.markdown("### 📈 Accuracy Comparison")
        fig8, ax8 = plt.subplots()
        ax8.bar(results.keys(), results.values(), color=['#4e79a7','#f28e2b','#76b7b2'])
        ax8.set_ylabel("Accuracy")
        ax8.set_ylim(0, 1)
        st.pyplot(fig8)

        st.markdown("### 🧾 Summary Table")
        result_df = pd.DataFrame(list(results.items()), columns=["Algorithm", "Accuracy"])
        st.dataframe(result_df)
