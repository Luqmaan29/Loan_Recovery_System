#!/usr/bin/env python3
"""
Smart Digital Lending Recommendation System Launcher
"""

import subprocess
import sys
import os

def check_environment():
    """Check if virtual environment is activated."""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment is activated")
        return True
    else:
        print("⚠️  Virtual environment not detected")
        print("Please activate your virtual environment first:")
        print("  source loan_system_env/bin/activate  # On macOS/Linux")
        print("  loan_system_env\\Scripts\\activate     # On Windows")
        return False

def check_dependencies():
    """Check if required packages are installed."""
    try:
        import streamlit
        import pandas
        import numpy
        import plotly
        import xgboost
        import lightgbm
        import sklearn
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def launch_dashboard():
    """Launch the Streamlit dashboard."""
    try:
        print("🚀 Starting Smart Digital Lending System...")
        print("📊 Dashboard will open at: http://localhost:8501")
        print("🔄 Press Ctrl+C to stop the server")
        print("-" * 50)
        
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "dashboard.py",
            "--server.headless", "false",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        return False

def main():
    """Main launcher function."""
    print("🏦 Smart Digital Lending Recommendation System")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        return
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Launch dashboard
    launch_dashboard()

if __name__ == "__main__":
    main()



