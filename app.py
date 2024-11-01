from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for macOS compatibility
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

def generate_data(N, mu, sigma2):
    X = np.random.rand(N)
    Y = mu + np.random.normal(0, np.sqrt(sigma2), N)
    return X, Y

def linear_regression(X, Y):
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), Y)
    slope = model.coef_[0]
    intercept = model.intercept_
    return slope, intercept

def generate_plot(X, Y, slope, intercept):
    plt.figure()
    plt.scatter(X, Y, color='blue', label='Data Points')
    plt.plot(X, slope * X + intercept, color='red', label=f'Fitted Line: y = {intercept:.2f} + {slope:.2f}x')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.title("Generated Plot")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()
    return f"data:image/png;base64,{plot_url}"

def generate_histogram(slopes, intercepts, base_slope, base_intercept):
    plt.figure()
    plt.hist(slopes, bins=20, alpha=0.5, label="Slopes", color="blue")
    plt.hist(intercepts, bins=20, alpha=0.5, label="Intercepts", color="orange")
    plt.axvline(base_slope, color="blue", linestyle="--", label=f"Slope: {base_slope:.2f}")
    plt.axvline(base_intercept, color="orange", linestyle="--", label=f"Intercept: {base_intercept:.2f}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Histogram of Slopes and Intercepts")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()
    return f"data:image/png;base64,{plot_url}"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        S = int(request.form["S"])

        # Generate base dataset
        X, Y = generate_data(N, mu, sigma2)
        base_slope, base_intercept = linear_regression(X, Y)
        plot1 = generate_plot(X, Y, base_slope, base_intercept)

        # Simulations for slopes and intercepts
        slopes, intercepts = [], []
        for _ in range(S):
            _, Y_sim = generate_data(N, mu, sigma2)
            slope, intercept = linear_regression(X, Y_sim)
            slopes.append(slope)
            intercepts.append(intercept)

        # Calculate proportions
        slope_extreme = np.mean([1 if abs(s) > abs(base_slope) else 0 for s in slopes]) * 100
        intercept_extreme = np.mean([1 if abs(i) > abs(base_intercept) else 0 for i in intercepts]) * 100

        # Generate histogram plot
        plot2 = generate_histogram(slopes, intercepts, base_slope, base_intercept)

        return render_template("index.html", plot1=plot1, plot2=plot2, slope_extreme=slope_extreme, intercept_extreme=intercept_extreme)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
