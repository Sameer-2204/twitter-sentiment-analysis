import subprocess
import sys

# List of required packages
required_packages = [
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "wordcloud",
    "scikit-learn",
    "nltk",
    "lightgbm",
    "tensorflow",
    "transformers",
    "flask",       # for API / UI
    "fastapi",     # optional, if you want to try modern API
    "uvicorn"      # to serve FastAPI
]

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Installed: {package}")
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install: {package}")

if __name__ == "__main__":
    print("🔧 Setting up environment...")
    for pkg in required_packages:
        install_package(pkg)
    print("🎉 Environment setup complete!")
