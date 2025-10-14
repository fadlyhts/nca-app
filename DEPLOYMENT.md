# Easypanel Deployment Guide

## Prerequisites
- GitHub repository: https://github.com/fadlyhts/nca-app.git
- Easypanel account (https://easypanel.io/)
- Docker and Dockerfile configured

## Deployment Steps

### 1. Login to Easypanel
1. Go to your Easypanel dashboard
2. Select or create a project

### 2. Create a New Service
1. Click **"Create Service"** or **"+ New Service"**
2. Choose **"App"** type
3. Select **"GitHub"** as source

### 3. Configure GitHub Repository
1. Connect your GitHub account if not already connected
2. Select repository: `fadlyhts/nca-app`
3. Select branch: `main`
4. Build method: **Dockerfile**

### 4. Configure Build Settings
- **Build Path**: `/` (root directory)
- **Dockerfile Path**: `Dockerfile`
- **Port**: `8501` (Streamlit default port)

### 5. Environment Variables (Optional)
You can add environment variables if needed:
```
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### 6. Domain Configuration
1. Easypanel will provide a default domain: `your-app.easypanel.host`
2. You can also add a custom domain if you have one

### 7. Deploy
1. Click **"Deploy"** or **"Create"**
2. Easypanel will:
   - Clone your repository
   - Build Docker image
   - Deploy the container
   - Expose it on port 8501

### 8. Monitor Deployment
- Check the **Logs** tab for build progress
- Wait for deployment to complete (may take 3-5 minutes)
- Look for message: "You can now view your Streamlit app in your browser"

## Alternative: Manual Docker Configuration

If you prefer manual configuration:

### Docker Build Settings:
```dockerfile
Build Command: docker build -t nca-app .
Run Command: docker run -p 8501:8501 nca-app
Port: 8501
```

### Resource Allocation (Recommended):
- **CPU**: 0.5 - 1 core
- **Memory**: 1-2 GB (recommended 2GB for ML models)
- **Storage**: 2-5 GB

## Post-Deployment

### Access Your Application
- Open the provided URL (e.g., `https://your-app.easypanel.host`)
- You should see the Streamlit dashboard

### Testing
1. Upload a CSV file in the EDA tab
2. Try the prediction functionality
3. Check that model files are loaded correctly

### Troubleshooting

#### If deployment fails:
1. Check logs in Easypanel dashboard
2. Verify all dependencies in requirements.txt
3. Ensure model files are in the repository

#### Common issues:
- **Port mismatch**: Verify port 8501 is exposed
- **Model files missing**: Check if model files are tracked in git (not in .gitignore)
- **Memory issues**: Increase memory allocation to 2GB
- **Build timeout**: ML dependencies may take time, be patient

#### Model Files Warning:
Check your `.gitignore` - it currently excludes:
- `model/` directory
- `*.h5` files
- `*.json` files in models_deployment

**Important**: If your model files are needed for deployment, you should either:
1. Remove them from .gitignore and commit them to GitHub
2. Or upload them separately in Easypanel as persistent volumes
3. Or download them during container startup

### Update Deployment
After pushing changes to GitHub:
1. Easypanel will auto-deploy if auto-deploy is enabled
2. Or manually click **"Redeploy"** in Easypanel dashboard

## Model Files Consideration

Your `.gitignore` currently excludes model files. For deployment, you have 3 options:

### Option 1: Include Model Files in Git (Recommended for small models)
Update `.gitignore` to allow model files:
```bash
# Remove or comment these lines from .gitignore:
# model/
# *.h5
# *.keras
```

Then commit and push:
```bash
git add model/
git commit -m "Add model files for deployment"
git push origin main
```

### Option 2: Use Persistent Storage
1. In Easypanel, add a **Volume**
2. Upload model files to the volume
3. Mount volume to `/app/model`

### Option 3: Download Models at Runtime
Add to Dockerfile before ENTRYPOINT:
```dockerfile
RUN curl -o model/multi_output_lstm_h5_lookback5.h5 YOUR_MODEL_URL
```

## Support
- Easypanel Documentation: https://easypanel.io/docs
- GitHub Issues: https://github.com/fadlyhts/nca-app/issues
