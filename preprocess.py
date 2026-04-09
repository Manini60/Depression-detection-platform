import pandas as pd

def preprocess():
    df = pd.read_csv("student_depression_dataset.csv")

    df = df[[
        'Gender',
        'Age',
        'Academic Pressure',
        'Study Satisfaction',
        'Sleep Duration',
        'Dietary Habits',
        'Financial Stress',
        'Depression'
    ]]

    # Convert numeric safely
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['Academic Pressure'] = pd.to_numeric(df['Academic Pressure'], errors='coerce')
    df['Financial Stress'] = pd.to_numeric(df['Financial Stress'], errors='coerce')
    df['Study Satisfaction'] = pd.to_numeric(df['Study Satisfaction'], errors='coerce')

    # Discretize Age
    df['Age'] = pd.cut(
        df['Age'],
        bins=[15, 20, 25, 30, 40],
        labels=['Teen', 'Young', 'Adult', 'Senior']
    )

    # Discretize Academic Pressure (1-5 scale)
    df['Academic Pressure'] = pd.cut(
        df['Academic Pressure'],
        bins=[0, 2, 3.5, 5],
        labels=['Low', 'Medium', 'High']
    )

    # Discretize Financial Stress (1-5 scale)
    df['Financial Stress'] = pd.cut(
        df['Financial Stress'],
        bins=[0, 2, 3.5, 5],
        labels=['Low', 'Medium', 'High']
    )

    # Discretize Study Satisfaction (1-5 scale)
    df['Study Satisfaction'] = pd.cut(
        df['Study Satisfaction'],
        bins=[0, 2, 3.5, 5],
        labels=['Low', 'Medium', 'High']
    )

    # Clean text columns
    df['Sleep Duration'] = df['Sleep Duration'].astype(str).str.strip().str.strip("'")
    df['Dietary Habits'] = df['Dietary Habits'].astype(str).str.strip()
    df['Gender'] = df['Gender'].astype(str).str.strip()
    df['Depression'] = df['Depression'].astype(str).str.strip()

    # Drop NaN rows
    df = df.dropna()

    # pgmpy requires category dtype
    for col in df.columns:
        df[col] = df[col].astype('category')

    return df


def get_raw_stats():
    """Return raw dataset statistics for dashboard."""
    df = pd.read_csv("student_depression_dataset.csv")
    total = len(df)
    depressed = int(df['Depression'].sum())
    not_depressed = total - depressed
    pct = round(depressed / total * 100, 1)

    # Age distribution
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    age_bins = pd.cut(df['Age'], bins=[15, 20, 25, 30, 40],
                      labels=['Teen (15-20)', 'Young (21-25)', 'Adult (26-30)', 'Senior (31-40)'])
    age_dist = age_bins.value_counts().sort_index().to_dict()

    # Gender split
    gender_dist = df['Gender'].value_counts().to_dict()

    # Sleep distribution
    sleep_dist = df['Sleep Duration'].astype(str).str.strip("'").value_counts().to_dict()

    # Dietary habits
    diet_dist = df['Dietary Habits'].value_counts().to_dict()

    return {
        'total': total,
        'depressed': depressed,
        'not_depressed': not_depressed,
        'pct': pct,
        'age_dist': age_dist,
        'gender_dist': gender_dist,
        'sleep_dist': sleep_dist,
        'diet_dist': diet_dist,
    }
