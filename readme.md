Model for the landmark competition hosted in kaggle. A pre train model made to weight less than 100MB and a API made using flask and connection.

Starting the server:
 python app1.py
How to use the endpoing:
  curl -o coordinates_face3.jpg -F "file=@face_3.jpg" -X POST localhost:5000/
