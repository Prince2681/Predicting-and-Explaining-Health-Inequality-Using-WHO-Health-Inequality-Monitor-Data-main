#!/bin/bash
set -e

# If arguments are passed to docker run, execute them
if [ "$#" -gt 0 ]; then
    exec "$@"
fi

echo "====================================================================="
echo " Starting WHO Health Inequality Monitor Container"
echo "====================================================================="

# Check if model exists; if not, run training pipeline
if [ ! -f "models/module6_random_forest.pkl" ]; then
    echo "Trained model not found. Running ML pipeline to train and evaluate models..."
    python main.py
else
    echo "Found existing trained model in models/module6_random_forest.pkl"
fi

echo "Launching Streamlit Web Application on port 8501..."
exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0
