import sys
from flask import Flask, request, render_template

from src.logger import logging
from src.exception import CustomException

from src.pipeline.predict_pipeline import PredictPipeline, CustomData

application = Flask(__name__)
app = application

# @app.route('/')
# def index():
#     return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def predict_data():
    try:

        if request.method == 'GET':
            return render_template('home.html')
        else:
            logging.info("prediction post method called")
            data=CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('writing_score')),
                writing_score=float(request.form.get('reading_score'))

            )
            pred_df=data.get_data_as_data_frame()

            logging.info("Before Prediction")
            logging.info(str(pred_df))

            predict_pipeline=PredictPipeline()
            
            results=predict_pipeline.predict(pred_df)
            logging.info("Prediction: {0}".format(results))
            
            return render_template('home.html',results=results[0])

    except Exception as e:
        raise CustomException(e, sys)


if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)