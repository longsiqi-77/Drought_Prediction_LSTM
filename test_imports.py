print("Testing imports...")
try:
    import numpy
    print(f"✅ numpy {numpy.__version__}")
    
    import pandas
    print(f"✅ pandas {pandas.__version__}")
    
    import streamlit
    print(f"✅ streamlit {streamlit.__version__}")
    
    import plotly
    print(f"✅ plotly {plotly.__version__}")
    
    import torch
    print(f"✅ torch {torch.__version__}")
    
    print("\n✅ All imports successful!")
    
except Exception as e:
    print(f"\n❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
