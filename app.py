import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import category_encoders as ce
import os
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error , mean_absolute_error
from flask import jsonify

app = Flask(__name__)

# Charger les données initiales
data = pd.read_csv('profitentr.csv')
current_data = data.copy()

# Variables globales
trained_model = None
X_test = None
y_test = None

def get_data_stats(df):
    instance_count = df.shape[0]
    return {
        'instance_count': instance_count,
        'attribute_count': df.shape[1],
        'data_preview': df.head(n=instance_count).to_html(classes='data-table', index=False),
        'missing_rows_count': df.isnull().any(axis=1).sum(),
        'missing_columns': df.columns[df.isnull().any()].tolist(),
        'corr_image': None
    }

@app.route('/')
def home():
    stats = get_data_stats(current_data)
    return render_template('index.html', **stats)

@app.route('/clean', methods=['POST'])
def clean_data():
    global current_data
    strategy = request.form.get('missingStrategy')
    
    if strategy == 'mean':
        current_data = data.fillna(data.mean(numeric_only=True))
    elif strategy == 'median':
        current_data = data.fillna(data.median(numeric_only=True))
    elif strategy == 'drop':
        current_data = data.dropna()
    
    stats = get_data_stats(current_data)
    return render_template('index.html', **stats)

@app.route('/encode_normalize', methods=['POST'])
def encode_normalize():
    global current_data
    encoding_method = request.form.get('encodingMethod')
    normalization_method = request.form.get('normalizationMethod')
    
    if encoding_method == 'onehot':
        encoder = OneHotEncoder(drop=None, sparse_output=False)
        encoded_state = encoder.fit_transform(current_data[['State']]).astype(np.int32)
        encoded_columns = [f"State_{cat.replace(' ', '_')}" for cat in encoder.categories_[0]]
        encoded_df = pd.DataFrame(encoded_state, columns=encoded_columns)
        current_data = pd.concat([current_data.drop('State', axis=1), encoded_df], axis=1)
    elif encoding_method == 'label':
        le = LabelEncoder()
        current_data['State'] = le.fit_transform(current_data['State']).astype(np.int32)
    elif encoding_method == 'binary':
        encoder = ce.BinaryEncoder(cols=['State'])
        current_data = encoder.fit_transform(current_data).astype(np.int32)
    
    numeric_features = current_data.select_dtypes(include=['float64', 'int64']).columns.drop('Profit', errors='ignore')
    
    if normalization_method == 'standard':
        scaler = StandardScaler()
    elif normalization_method == 'minmax':
        scaler = MinMaxScaler()
    elif normalization_method == 'maxabs':
        scaler = MaxAbsScaler()
    elif normalization_method == 'robuste':
        scaler = RobustScaler()
    else:
        scaler = None
    
    if scaler is not None and not numeric_features.empty:
        current_data[numeric_features] = scaler.fit_transform(current_data[numeric_features])

    stats = get_data_stats(current_data)
    return render_template('index.html', **stats)

@app.route('/generate_corr', methods=['POST'])
def generate_corr():
    global current_data
    
    excluded_columns = ['Profit'] + [col for col in current_data.columns if col.startswith('State')]
    features = [col for col in current_data.columns if col not in excluded_columns]
    
    corr_matrix = current_data[features].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    img_path = f'static/corr_matrix_{timestamp}.png'
    plt.savefig(img_path)
    plt.close()
    
    stats = get_data_stats(current_data)
    stats['corr_image'] = img_path
    
    return render_template('index.html', **stats)

@app.route('/apply_pca', methods=['POST'])
def apply_pca():
    global current_data

    excluded_columns = ['Profit'] + [col for col in current_data.columns if col.startswith('State')]
    features = [col for col in current_data.columns if col not in excluded_columns]
    
    corr_matrix = current_data[features].corr()
    
    sorted_corrs = corr_matrix.abs().unstack().sort_values(ascending=False)
    top_pairs = sorted_corrs[sorted_corrs < 1].index[:3]
    top_features = list(set([col for pair in top_pairs for col in pair]))

    if len(top_features) < 2:
        return "Pas assez de colonnes pour l'ACP.", 400

    n_components = int(request.form.get('pcaComponents', 2))
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(current_data[top_features])
    
    current_data = current_data.drop(columns=top_features)
    pca_columns = [f'PC{i+1}' for i in range(n_components)]
    current_data[pca_columns] = principal_components

    plt.figure(figsize=(8, 6))
    if n_components == 2:
        sns.scatterplot(x=current_data['PC1'], y=current_data['PC2'])
    else:
        sns.histplot(current_data['PC1'], kde=True)
    plt.title('Projection après ACP')
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    img_path = f'static/pca_{timestamp}.png'
    plt.savefig(img_path)
    plt.close()

    stats = get_data_stats(current_data)
    stats['pca_image'] = img_path

    return render_template('index.html', **stats)

@app.route('/train_model', methods=['POST'])
def train_model():
    global trained_model, X_test, y_test
    
    train_split = int(request.form.get('trainSplit', 80))
    test_size = (100 - train_split) / 100.0

    X = current_data.drop('Profit', axis=1)
    y = current_data['Profit']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=42
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    trained_model = model
    
    return render_template('index.html',
        train_message=f"✅ Modèle entraîné ({train_split}%) !",
        **get_data_stats(current_data))

@app.route('/test_model', methods=['POST'])
def test_model():
    global trained_model, X_test, y_test
    
    # Vérifier si le modèle est entraîné
    if not trained_model:
        return render_template('index.html', 
            test_message="❌ Entraînez le modèle d'abord !",
            **get_data_stats(current_data))
    
    # Faire des prédictions
    y_pred = trained_model.predict(X_test)
    
    # Calculer les métriques
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Générer un graphique
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, label='Prédictions')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Valeurs idéales')
    plt.xlabel('Valeurs Réelles')
    plt.ylabel('Valeurs Prédites')
    plt.title('Comparaison Réelles vs Prédites')
    plt.legend()
    
    # Sauvegarder le graphique
    plot_path = f'static/pred_plot_{datetime.now().strftime("%Y%m%d%H%M%S")}.png'
    plt.savefig(plot_path)
    plt.close()
    
    # Afficher les résultats
    return render_template('index.html',
        r2=r2,
        mse=mse,
        mae=mae,
        plot_image=plot_path,
        **get_data_stats(current_data))

@app.route('/predict', methods=['POST'])
def predict():
    if not trained_model:
        return render_template('index.html', test_message="❌ Entraînez le modèle d'abord !", **get_data_stats(current_data))
    
    # Récupérer les données du formulaire
    rd_spend = float(request.form['rdSpend'])
    administration = float(request.form['administration'])
    marketing_spend = float(request.form['marketingSpend'])
    state = request.form['state']
    
    # Appliquer les mêmes transformations que lors de l'entraînement
    # Encodage de l'état
    if encoding_method == 'onehot':
        state_encoded = encoder.transform([[state]]).toarray()
    elif encoding_method == 'label':
        state_encoded = le.transform([state])
    elif encoding_method == 'binary':
        state_encoded = encoder.transform([state])
    
    # Normalisation des valeurs
    features = np.array([[rd_spend, administration, marketing_spend] + state_encoded.tolist()])
    features_normalized = scaler.transform(features)
    
    # Prédiction
    predicted_profit = trained_model.predict(features_normalized)
    
    return render_template('index.html', predicted_profit=predicted_profit[0], **get_data_stats(current_data))

if __name__ == '__main__':
    app.run(debug=True)