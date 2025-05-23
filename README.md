# EU Project Matching System

This is a Shiny application for matching research proposals with similar EU-funded projects.

## Features

- Research proposal matching with similar EU projects
- Interactive project filtering by year
- Detailed project information display
- Organization profiles and collaboration networks
- Funding mechanism analysis
- Geographic visualization of project partners

## Prerequisites

- Python 3.9 or higher
- Git LFS (Large File Storage)

### Installing Git LFS

#### On macOS:
```bash
brew install git-lfs
git lfs install
```

#### On Ubuntu/Debian:
```bash
sudo apt install git-lfs
git lfs install
```

#### On Windows:
Download and install from https://git-lfs.github.com/

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Modern-Data-Analytics_2025.git
cd Modern-Data-Analytics_2025
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required data and model files:
   - The repository uses Git LFS to manage large files
   - After cloning, run `git lfs pull` to download all large files
   - If you encounter any issues, you can manually download the files from the releases page

## Usage

Run the application:
```bash
shiny run app_v2.py
```

The application will be available at http://127.0.0.1:8000

## Project Structure

- `app_v2.py`: Main application file
- `data_1/processed/`: Processed project and organization data
  - `project_merged.csv`: Merged project data
  - `org_unique_detailed.csv`: Detailed organization information
  - `scivoc_summary.csv`: EuroSciVoc topic summaries
- `recommender/`: Project matching algorithm implementation
- `models/`: Pre-trained models and embeddings
- `requirements.txt`: Python package dependencies

## Data

The application uses processed data from EU-funded projects, including:
- Project details
- Organization information
- Funding mechanisms
- EuroSciVoc topics

## Troubleshooting

1. If you encounter SSL-related warnings, you can safely ignore them as they don't affect the application's functionality.

2. If you have issues with large files:
   - Make sure Git LFS is properly installed
   - Run `git lfs pull` to download all large files
   - Check if you have enough disk space (at least 500MB free)

3. If the application fails to start:
   - Make sure all dependencies are installed correctly
   - Check if you're using the correct Python version
   - Ensure you're running the application from the virtual environment 