FROM python:3.9

COPY Objects/ /usr/src/Objects
COPY Article_Euler_main.py .
COPY Article_Euler_Number_2D_And_3D_ML.py .
COPY Article_Euler_Number_2D_General.py .
COPY Article_Euler_Number_3D_General.py .
COPY Article_Euler_Number_Create_Data.py .
COPY Article_Euler_Number_Libraries.py .
COPY Article_Euler_Number_Utilities.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "./Article_Euler_main.py"]
