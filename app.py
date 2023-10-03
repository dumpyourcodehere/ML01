import pandas as pd
from flask import Flask, render_template, request
import model

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get data from the form
        data = {
            "Location": request.form.get("location"),
            "Emp. Group": request.form.get("emp_group"),
            "Function": request.form.get("function"),
            "Gender ": request.form.get("gender"),
            "Tenure Grp.": request.form.get("tenure_grp"),
            "Experience (YY.MM)": request.form.get("experience"),  # Add this input to the form
            "Marital Status": request.form.get("marital_status"),
            "Age in YY.": request.form.get("age"),  # Add this input to the form
            "Hiring Source": request.form.get("hiring_source"),
            "Promoted/Non Promoted": request.form.get("promotion"),
            "Job Role Match": request.form.get("job_role_match")
        }

        input_data = pd.DataFrame([data])
        prediction = model.predict(input_data)[0]
    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
