import flask
import pandas as pd
from joblib import dump, load

with open(f'model/housepriceprediction.joblib', 'rb') as f:
    model = load(f)

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return flask.render_template('main.html')

    if flask.request.method == 'POST':
        # Update the variable names and form fields accordingly
        lot_frontage = flask.request.form['lot_frontage']
        lot_area = flask.request.form['lot_area']
        year_built = flask.request.form['year_built']
        first_floor_sf = flask.request.form['first_floor_sf']
        second_floor_sf = flask.request.form['second_floor_sf']
        # Add more variables as needed

        input_variables = pd.DataFrame(
            [[lot_frontage, lot_area, year_built, first_floor_sf, second_floor_sf]],
            columns=['LotFrontage', 'LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF'],
            dtype='float',
            index=['input']
        )

        predictions = model.predict(input_variables)[0]
        print(predictions)

        return flask.render_template('main.html', 
                                     original_input={'LotFrontage': lot_frontage, 'LotArea': lot_area, 'YearBuilt': year_built, '1stFlrSF': first_floor_sf, '2ndFlrSF': second_floor_sf},
                                     result=predictions)

if __name__ == '__main__':
    app.run(debug=True)
