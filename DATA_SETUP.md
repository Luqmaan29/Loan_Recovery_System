# 📊 Dataset Setup Instructions

## 🎯 Getting the Home Credit Default Risk Dataset

The large data files are not included in this repository due to GitHub's file size limitations. Follow these steps to set up the complete dataset:

### **📥 Download Instructions**

1. **Visit the Kaggle Competition Page**:
   - Go to: [Home Credit Default Risk Competition](https://www.kaggle.com/c/home-credit-default-risk/data)

2. **Download the Required Files**:
   - `application_train.csv` (158.44 MB)
   - `application_test.csv` (48.74 MB)
   - `bureau.csv` (162.14 MB)
   - `bureau_balance.csv` (358.19 MB)
   - `credit_card_balance.csv` (404.91 MB)
   - `installments_payments.csv` (689.62 MB)
   - `POS_CASH_balance.csv` (374.51 MB)
   - `previous_application.csv` (386.21 MB)

3. **Place Files in Data Directory**:
   ```
   Loan_System/
   └── data/
       ├── application_train.csv
       ├── application_test.csv
       ├── bureau.csv
       ├── bureau_balance.csv
       ├── credit_card_balance.csv
       ├── installments_payments.csv
       ├── POS_CASH_balance.csv
       ├── previous_application.csv
       ├── HomeCredit_columns_description.csv ✅ (included)
       └── sample_submission.csv ✅ (included)
   ```

### **🚀 Quick Setup Script**

Run the following command to check if all data files are present:

```bash
python -c "
import os
required_files = [
    'application_train.csv',
    'application_test.csv', 
    'bureau.csv',
    'bureau_balance.csv',
    'credit_card_balance.csv',
    'installments_payments.csv',
    'POS_CASH_balance.csv',
    'previous_application.csv'
]

missing_files = []
for file in required_files:
    if not os.path.exists(f'data/{file}'):
        missing_files.append(file)

if missing_files:
    print('❌ Missing files:')
    for file in missing_files:
        print(f'  - {file}')
    print('\n📥 Please download these files from Kaggle and place them in the data/ directory')
else:
    print('✅ All required data files are present!')
    print('🚀 You can now run the system with: streamlit run dashboard.py')
"
```

### **📋 Alternative: Use Sample Data**

If you want to test the system without downloading the full dataset, the system includes sample data generation capabilities:

```python
# The system will automatically create sample datasets for testing
# when the full dataset is not available
```

### **🔧 System Requirements**

- **Minimum RAM**: 8GB (for full dataset processing)
- **Storage Space**: ~3GB for all data files
- **Python Version**: 3.13+

### **📞 Support**

If you encounter any issues with dataset setup:
1. Check the [Kaggle Competition Page](https://www.kaggle.com/c/home-credit-default-risk/data)
2. Ensure you have a Kaggle account and API access
3. Verify all files are placed in the correct `data/` directory

---

**Note**: The dataset is publicly available through Kaggle and is used for educational and research purposes in credit risk assessment.
