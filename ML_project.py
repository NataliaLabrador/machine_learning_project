import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV, cross_val_score
from collections import Counter

# App title and description
st.markdown("<h1 style='text-align: center; font-size: 55px;'>&#127973; &#128137;</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; font-size: 40px;'>Detección de pacientes con diabetes</h2>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; font-size: 35px;'>a través de Machine Learning &#128187; &#x1F4CA;</h2>", unsafe_allow_html=True)
st.write(" ")
st.write(" ")

# Sidebar for model selection and configuration
st.sidebar.title("Configuración")
st.sidebar.markdown("Aquí podrás configurar el modelo según tus necesidades.")

# File uploader
def load_data():
    uploaded_file = st.sidebar.file_uploader("Seleccionar un archivo CSV", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, sep=',', quotechar='"')
            st.write("### Previsualización de los datos")
            st.write("Datos de nuestros pacientes:")
            st.write(df.head())
            return df
        except ValueError as e:
            st.error(f"Error al leer el archivo: {e}")
            return None

# Data cleaning
def clean_data(df):
    st.write("### Limpieza de Datos")
    st.write("Descripción de los datos originales:")
    st.write(df.describe())

    # Handle missing values
    df = df.dropna()

    # Scale features
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])

    st.write("Datos limpios:")
    st.write(df.head())

    return df

def plot_outcome_distribution(data):
    # Calcular el recuento de valores únicos en la columna "Outcome"
    outcome_counts = data['Outcome'].value_counts()

    # Crear un gráfico de barras
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=outcome_counts.index, y=outcome_counts.values, palette='Set2', ax=ax)
    ax.set_title('Distribución de la columna "Outcome"')
    ax.set_xlabel('Outcome')
    ax.set_ylabel('Cantidad')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No diabetes', 'Diabetes'])
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)

def undersample_data(data):
    # Contar la cantidad de cada clase en la columna "Outcome"
    class_counts = data['Outcome'].value_counts()

    # Determinar la clase minoritaria y su cantidad
    minority_class = class_counts.idxmin()
    minority_class_count = class_counts.min()

    # Determinar la clase mayoritaria y su cantidad
    majority_class = class_counts.idxmax()
    majority_class_count = class_counts.max()

    # Realizar undersampling para igualar la cantidad de ambas clases
    if minority_class_count < majority_class_count:
        # Seleccionar índices de la clase mayoritaria para mantener
        majority_indices = data[data['Outcome'] == majority_class].index

        # Muestrear aleatoriamente la clase mayoritaria para igualar la cantidad con la clase minoritaria
        sampled_majority_indices = np.random.choice(majority_indices, minority_class_count, replace=False)

        # Seleccionar todos los índices de la clase minoritaria
        minority_indices = data[data['Outcome'] == minority_class].index

        # Combinar los índices de la clase mayoritaria muestreada con los índices de la clase minoritaria
        undersampled_indices = np.concatenate([sampled_majority_indices, minority_indices])

        # Obtener los datos undersampled
        undersampled_data = data.loc[undersampled_indices]

        return undersampled_data
    else:
        return data

def plot_undersampling_result(data):
    # Contar la cantidad de cada clase en la columna "Outcome" después del undersampling
    undersampled_class_counts = data['Outcome'].value_counts()

    # Crear un gráfico de barras para mostrar el resultado del undersampling
    fig, ax = plt.subplots(figsize=(8, 6))
    undersampled_class_counts.plot(kind='bar', color='coral', ax=ax)
    ax.set_title('Resultado del Undersampling')
    ax.set_xlabel('Outcome')
    ax.set_ylabel('Cantidad')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No diabetes', 'Diabetes'])
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)

def choose_model():
    model_option = st.sidebar.selectbox("Elige un modelo", 
                                        ["KNN", "Regresión Logística", "Regresión Lineal", "Árbol de decisión", 
                                         "Bagging", "Random Forest", 
                                         "AdaBoost", "Gradient Boosting"])

    param_grid = {}
    model_description = ""
    if model_option == "KNN":
        n_neighbors = st.sidebar.slider("Número de vecinos", 1, 20, 3)
        model = KNeighborsClassifier()
        param_grid = {'n_neighbors': np.arange(1, 21)}
        model_type = 'classifier'
        model_description = "KNN (K-Nearest Neighbors): Clasifica un dato nuevo basándose en la mayoría de las clases de sus vecinos más cercanos en el espacio de características."
    elif model_option == "Regresión Logística":
        model = LogisticRegression()
        param_grid = {'C': np.logspace(-4, 4, 20), 'solver': ['liblinear']}
        model_type = 'classifier'
        model_description = "Regresión Logística: Predice la probabilidad de que una observación pertenezca a una clase determinada usando una función logística."
    elif model_option == "Regresión Lineal":
        model = LinearRegression()
        model_type = 'regressor'
        model_description = "Regresión Lineal: Predice un valor numérico basado en la relación lineal entre las características de entrada y la variable objetivo."
    elif model_option == "Árbol de decisión":
        max_depth = st.sidebar.slider("Max depth", 1, 20, 3)
        model = DecisionTreeClassifier()
        param_grid = {'max_depth': np.arange(1, 21)}
        model_type = 'classifier'
        model_description = "Árbol de Decisión: Realiza clasificaciones o predicciones utilizando un modelo basado en reglas de decisión extraídas de los datos."
    elif model_option == "Bagging":
        model = BaggingClassifier(random_state=42)
        model_type = 'classifier'
        model_description = "Bagging (Bootstrap Aggregating): Mejora la precisión de los modelos de clasificación o regresión al entrenar múltiples modelos en subconjuntos aleatorios de los datos y promediar sus predicciones."
    elif model_option == "Random Forest":
        n_estimators = st.sidebar.slider("Número de estimadores", 10, 100, 10)
        model = RandomForestClassifier(random_state=42)
        param_grid = {'n_estimators': np.arange(10, 101, 10)}
        model_type = 'classifier'
        model_description = "Random Forest: Combina múltiples árboles de decisión entrenados en diferentes subconjuntos de los datos para mejorar la precisión y reducir el sobreajuste."
    elif model_option == "AdaBoost":
        n_estimators = st.sidebar.slider("Número de estimadores", 10, 100, 50)
        model = AdaBoostClassifier()
        param_grid = {'n_estimators': np.arange(10, 101, 10)}
        model_type = 'classifier'
        model_description = "AdaBoost (Adaptive Boosting): Combina múltiples clasificadores débiles secuencialmente, poniendo más peso en los errores de las clasificaciones anteriores, para crear un clasificador fuerte."
    else:
        n_estimators = st.sidebar.slider("Número de estimadores", 10, 100, 50)
        model = GradientBoostingClassifier()
        param_grid = {'n_estimators': np.arange(10, 101, 10)}
        model_type = 'classifier'
        model_description = "Gradient Boosting: Iterativamente mejora un modelo combinando predicciones de modelos más débiles, minimizando el error en cada paso mediante la optimización del gradiente."

    st.markdown(f"<h3 style='text-align: center;'>¿Qué hace este modelo?</h3>", unsafe_allow_html=True)
    st.write(model_description)

    return model, model_type, param_grid

# Dictionary to store metrics of each model
metrics_dict = {}

def train_and_evaluate_model(model, model_type, param_grid, X_train, X_test, y_train, y_test):
    best_model = model
    # Dividir la pantalla en dos columnas
    col1, col2 = st.columns(2)



    # Búsqueda de Hiperparámetros
    with col1:
        if param_grid:
            st.write("### Búsqueda de Hiperparámetros")
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            st.write("Mejores Parámetros:")
            st.table(grid_search.best_params_)

    # Validación Cruzada
    with col2:
        st.write("### Validación Cruzada")
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)  # Utiliza el mejor modelo encontrado
        st.write("Puntuaciones de Validación Cruzada:", cv_scores)
        st.write("Media de Puntuaciones de Validación Cruzada:", cv_scores.mean())
    
    # Train the model with the entire training set
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    if model_type == 'classifier':
        st.write("### Matriz de confusión")
        cm = confusion_matrix(y_test, y_pred)
        st.write(cm)

        st.write("### Reporte de Clasificación")
        cr = classification_report(y_test, y_pred)
        st.text(cr)

        st.write("### Precisión del Modelo")
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy:", accuracy)

        # Actualizar el diccionario de métricas
        metrics_dict[model.__class__.__name__] = {'Confusion Matrix': cm, 'Classification Report': cr, 'Accuracy': accuracy}

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        st.pyplot(fig)

    else:
        st.write("### Métricas de Regresión")
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f"Mean Absolute Error:", mae)
        st.write(f"Mean Squared Error:", mse)
        st.write(f"R^2 Score:", r2)

        # Actualizar el diccionario de métricas
        metrics_dict[model.__class__.__name__] = {'Mean Absolute Error': mae, 'Mean Squared Error': mse, 'R^2 Score': r2}

        fig, ax = plt.subplots()
        sns.scatterplot(x=y_test, y=y_pred, ax=ax)
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        st.pyplot(fig)

        st.write("### Gráfico de errores")
        errors = y_test - y_pred
    
        fig, ax = plt.subplots()
        sns.histplot(errors, kde=True, ax=ax)
        ax.set_title('Error Distribution')
        ax.set_xlabel('Error')
        st.pyplot(fig)

def main():
    data = load_data()

    if data is not None:
        data = clean_data(data)

        # Llamar a la función para mostrar la distribución de "Outcome"
        plot_outcome_distribution(data)

        # Opción para aplicar undersampling
        apply_undersampling = st.sidebar.checkbox("Aplicar Undersampling", value=False)

        if apply_undersampling:
            # Llamar a las funciones para mostrar la distribución del Undersampled Data
            undersampled_data = undersample_data(data)
            plot_undersampling_result(undersampled_data)
            data = undersampled_data

        # Assume the last column is the target
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        # Convert target column for classification if necessary
        model, model_type, param_grid = choose_model()

        if model_type == 'classifier':
            if y.dtype != object:
                y = pd.cut(y, bins=2, labels=["sin diabetes", "diabetes"])
        else:
            if y.dtype == object:
                y = y.astype('category').cat.codes

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        if st.sidebar.button("Train and Evaluate Model"):
            with st.spinner('Training and evaluating...'):
                train_and_evaluate_model(model, model_type, param_grid, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()














