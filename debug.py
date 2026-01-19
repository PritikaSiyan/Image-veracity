import os
print("CWD:", os.getcwd())
print("BASE_DIR:", os.path.dirname(__file__) if '__file__' in globals() else 'no file')
