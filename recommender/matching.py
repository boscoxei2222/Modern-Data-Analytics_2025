import numpy as np
import pickle
# from sentence_transformers import SentenceTransformer, models
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def get_top_matches(proposal_text, project_data, top_n=10, scivoc_weight=0.5):
    """
    Get the most matching projects for a proposal
    
    Args:
        proposal_text (str): Research proposal text
        project_data (pd.DataFrame): Full project data, including SCV_ columns
        top_n (int): Number of matches to return
        scivoc_weight (float): Weight for EuroSciVoc topic encoding (0-1)
    
    Returns:
        tuple: (top_match_ids_scores, theme_similarities)
            - top_match_ids_scores: List containing project IDs and similarity scores
            - theme_similarities: DataFrame containing project theme similarity scores
    """
    # Load model
    with open("models/embedding_model_name.txt", "r") as f:
        model_name = f.read().strip()
    
    # # Manually configure sentence-transformers
    # word_embedding_model = models.Transformer(model_name)
    # pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    # model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    model = SentenceTransformer('models/all-MiniLM-L6-v2', local_files_only=True)
    
    # Load project encodings
    project_embeddings = np.load("models/project_embeddings.npy")
    with open("models/project_ids.pkl", "rb") as f:
        project_ids = pickle.load(f)
    
    # Encode proposal text
    proposal_embedding = model.encode([proposal_text])[0]
    
    # Calculate text similarity
    text_similarities = cosine_similarity(
        [proposal_embedding], 
        project_embeddings
    )[0]
    
    # Extract SCV_ columns from project_data as theme encodings
    scivoc_vectors = project_data.filter(regex="^SCV_").values
    theme_labels = [col.replace("SCV_", "") for col in project_data.filter(regex="^SCV_").columns]
    
    # Extract theme labels from proposal
    proposal_themes = set()
    for theme in theme_labels:
        if theme.lower() in proposal_text.lower():
            proposal_themes.add(theme)
    
    # Convert proposal themes to one-hot encoding
    proposal_theme_encoding = np.zeros(len(theme_labels))
    for theme in proposal_themes:
        idx = np.where(theme_labels == theme)[0]
        if len(idx) > 0:
            proposal_theme_encoding[idx[0]] = 1
    
    # Calculate theme similarity
    theme_similarities = []
    for project_id in project_ids:
        project_idx = project_data[project_data["projectID"] == project_id].index[0]
        project_theme_encoding = scivoc_vectors[project_idx]
        intersection = np.sum(np.minimum(proposal_theme_encoding, project_theme_encoding))
        union = np.sum(np.maximum(proposal_theme_encoding, project_theme_encoding))
        theme_sim = intersection / union if union > 0 else 0
        theme_similarities.append(theme_sim)
    
    theme_similarities = np.array(theme_similarities)
    
    # Combine similarity scores
    combined_similarities = (1 - scivoc_weight) * text_similarities + scivoc_weight * theme_similarities
    
    # Get top_n matches
    top_indices = np.argsort(combined_similarities)[-top_n:][::-1]
    
    # Prepare theme similarity DataFrame
    theme_df = pd.DataFrame({
        'projectID': [project_ids[i] for i in top_indices],
        'theme_similarity': theme_similarities[top_indices],
        'text_similarity': text_similarities[top_indices],
        'combined_similarity': combined_similarities[top_indices]
    })
    
    # Get acronym from project_data
    theme_df = theme_df.merge(
        project_data[['projectID', 'acronym']], 
        on='projectID',
        how='left'
    )
    
    return [(project_ids[i], combined_similarities[i]) for i in top_indices], theme_df
