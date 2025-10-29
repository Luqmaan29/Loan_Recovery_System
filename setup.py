#!/usr/bin/env python3
"""
Setup script for Smart Digital Lending Recommendation System
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def setup_environment():
    """Setup the project environment."""
    print("🏦 Smart Digital Lending Recommendation System Setup")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create virtual environment
    if not os.path.exists("loan_system_env"):
        if not run_command("python -m venv loan_system_env", "Creating virtual environment"):
            return False
    else:
        print("✅ Virtual environment already exists")
    
    # Install requirements
    pip_command = "loan_system_env/bin/pip" if os.name != 'nt' else "loan_system_env\\Scripts\\pip"
    if not run_command(f"{pip_command} install -r requirements.txt", "Installing dependencies"):
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Activate virtual environment:")
    print("   source loan_system_env/bin/activate  # On macOS/Linux")
    print("   loan_system_env\\Scripts\\activate     # On Windows")
    print("2. Run the system:")
    print("   python run_system.py")
    print("   # or")
    print("   streamlit run dashboard.py")
    
    return True

if __name__ == "__main__":
    setup_environment()






