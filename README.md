# Student Exam Score Predictor

A machine learning project that predicts a student's exam score from study hours, attendance, and sleep hours using Linear Regression, with statistical analysis of the data.

![Actual vs Predicted](actual_vs_predicted.png)

## Problem Statement
Can we predict a student's exam score from simple daily habits? This project builds a regression model to answer that, and measures how much each habit contributes to the score.

## Dataset
A synthetic dataset of 200 students generated with NumPy (random seed = 42, so results are reproducible).

| Feature | Description | Range |
|---|---|---|
| study_hours | Hours studied per day | 1 to 10 |
| attendance | Attendance percentage | 50 to 100 |
| sleep_hours | Hours slept per night | 4 to 9 |
| exam_score (target) | Final exam score | 0 to 100 |

The score is generated from this formula plus random noise (standard deviation 5):
`score = 5 x study_hours + 0.3 x attendance + 1.5 x sleep_hours + noise`

Because the data is synthetic, this project demonstrates the machine learning workflow, not real student behaviour.

## Tools and Libraries
Python, Pandas, NumPy, Matplotlib, scikit-learn

## Methodology
1. **Data generation:** created 200 student records with NumPy.
2. **Statistical analysis:** summary statistics (mean, standard deviation, min, max) and correlation with the target.
3. **Train-test split:** 80% training, 20% testing (random_state = 42).
4. **Model training:** fitted a Linear Regression model on the training set.
5. **Evaluation:** measured performance on the unseen test set using R2 and RMSE.
6. **Visualization:** plotted actual vs predicted scores.
7. **Prediction:** predicted the score for a new student.

## Results

| Metric | Value |
|---|---|
| R2 score | 0.92 |
| RMSE | 4.15 marks |
| Mean exam score | 58.95 |
| Standard deviation of score | 14.65 |

**Correlation with exam score**

| Feature | Correlation (r) |
|---|---|
| study_hours | 0.89 |
| attendance | 0.23 |
| sleep_hours | 0.11 |

**Model coefficients (marks gained per unit increase)**

| Feature | Coefficient |
|---|---|
| study_hours | 4.92 |
| attendance | 0.28 |
| sleep_hours | 1.88 |

## Key Insights
- Study hours have by far the strongest relationship with exam score (r = 0.89).
- Each extra hour of study adds about 4.9 marks, holding other factors constant.
- The model explains about 92% of the variation in scores, and predictions are off by about 4 marks on average.
- The learned coefficients (4.92, 0.28, 1.88) are close to the true values used to generate the data (5, 0.3, 1.5), which shows the model recovered the underlying pattern.
- Example: a student studying 6 hours with 90% attendance and 7 hours of sleep is predicted to score about 67.

## How to Run
1. Clone the repository:
```
git clone https://github.com/shafikacode/student-score-predictor.git
cd student-score-predictor
```
2. Install the libraries:
```
pip install pandas numpy matplotlib scikit-learn
```
3. Run the script:
```
python student_score_predictor.py
```
The script prints the statistics and results, and saves `actual_vs_predicted.png`.

## Project Structure
```
student-score-predictor/
|-- student_score_predictor.py
|-- actual_vs_predicted.png
|-- README.md
```

## Limitations
- The data is synthetic, so the high R2 reflects how the data was generated, not real-world accuracy.
- Linear Regression assumes a straight-line relationship between features and score.
- Only three features are used. Real exam performance depends on many more factors.

## Future Improvements
- Test the model on a real student performance dataset.
- Compare with other models (Decision Tree, Random Forest) and use cross-validation.
- Add a residual plot to check model assumptions.
- Build a simple web app where a user enters their habits and sees a predicted score.

## Author
Shafika S
B.Sc. Computer Science student, Chennai
GitHub: github.com/shafikacode | LinkedIn: linkedin.com/in/shafika-s-datascientist
