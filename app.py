import os
import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request, jsonify
from preprocess import preprocess, get_raw_stats
from model import create_model, draw_network, confusion, get_model_stats
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import numpy as np
import json

# ── Anthropic client (server-side text analysis) ──────────────────────────────
try:
    import anthropic as _anthropic_lib
    _anthropic_client = _anthropic_lib.Anthropic()   # reads ANTHROPIC_API_KEY env var
    ANTHROPIC_AVAILABLE = True
    print("✓ Anthropic SDK ready")
except Exception as e:
    ANTHROPIC_AVAILABLE = False
    print(f"⚠ Anthropic SDK not available: {e}")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, "templates"),
            static_folder=os.path.join(BASE_DIR, "static"))

# ── Boot: load data and model ──────────────────────────────────────────────────
print("⟳ Loading dataset...")
data = preprocess()
stats = get_raw_stats()

print("⟳ Building Bayesian Network...")
model = create_model(data)
draw_network(model)
confusion()

infer = VariableElimination(model)
print("✓ Model ready.")


# ── Charts ─────────────────────────────────────────────────────────────────────
def depression_chart():
    counts = data['Depression'].value_counts()
    vals = [counts.get('0', 0), counts.get('1', 0)]
    colors = ['#1e6b4a', '#c0392b']

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    wedges, texts, autotexts = ax.pie(
        vals, labels=['Not Depressed', 'Depressed'], colors=colors,
        autopct='%1.1f%%', startangle=90, pctdistance=0.75,
        wedgeprops=dict(width=0.6, edgecolor='#0d1117', linewidth=3)
    )
    for t in texts:   t.set_color('white');  t.set_fontsize(12)
    for at in autotexts: at.set_color('white'); at.set_fontsize(11); at.set_fontweight('bold')

    ax.set_title("Depression Distribution", color='white', fontsize=14, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig("static/depression_chart.png", dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()


def risk_gauge(yes_prob, risk):
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#111827')

    bars = ax.bar(['Not Depressed', 'Depressed'],
                  [1 - yes_prob, yes_prob],
                  color=['#1e6b4a', '#c0392b'], width=0.45, edgecolor='none')

    for bar, val in zip(bars, [1 - yes_prob, yes_prob]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', va='bottom',
                color='white', fontsize=14, fontweight='bold')

    ax.set_ylim(0, 1.2)
    ax.set_ylabel('Probability', color='#8b9dc3', fontsize=11)
    ax.set_title(f'Prediction: {risk}', color='white', fontsize=14, fontweight='bold')
    for spine in ['top', 'right']: ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']: ax.spines[spine].set_color('#2a3f5f')
    ax.tick_params(colors='#8b9dc3')

    plt.tight_layout()
    plt.savefig("static/probability.png", dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    depression_chart()
    s = stats
    return render_template('index.html',
                           total=s['total'], yes=s['depressed'], no=s['not_depressed'], pct=s['pct'],
                           age_dist=json.dumps({str(k): v for k, v in s['age_dist'].items()}),
                           gender_dist=json.dumps(s['gender_dist']),
                           sleep_dist=json.dumps(s['sleep_dist']),
                           diet_dist=json.dumps(s['diet_dist']))


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None; risk = None; yes_prob = None; no_prob = None
    combined_prob = None; combined_risk = None; tips = []
    cgpa = None; work_hours = None; user_text = ''

    if request.method == 'POST':
        evidence = {
            'Academic Pressure':  request.form['ap'],
            'Financial Stress':   request.form['fs'],
            'Sleep Duration':     request.form['sleep'],
            'Study Satisfaction': request.form['ss'],
            'Dietary Habits':     request.form['diet'],
            'Gender':             request.form['gender'],
            'Age':                request.form['age'],
        }
        cgpa       = request.form.get('cgpa', '').strip()
        work_hours = request.form.get('work_hours', '5')
        user_text  = request.form.get('user_text', '').strip()

        result   = infer.query(variables=['Depression'], evidence=evidence)
        no_prob  = float(result.values[0])
        yes_prob = float(result.values[1])

        # CGPA adjustment
        cgpa_penalty = 0.0
        try:
            cv = float(cgpa)
            cgpa_penalty = 0.08 if cv < 5.0 else (0.03 if cv < 7.0 else -0.03)
        except Exception: pass

        # Hours adjustment
        hours_penalty = 0.0
        try:
            hrs = float(work_hours)
            hours_penalty = 0.07 if hrs >= 10 else (0.03 if hrs >= 7 else (0.02 if hrs <= 3 else 0.0))
        except Exception: pass

        # Text sentiment (submitted as hidden field from JS after /api/analyse-text)
        text_adjustment = 0.0
        try:
            text_adjustment = float(request.form.get('text_sentiment_score', '0'))
        except Exception: pass

        combined_prob = min(max(yes_prob + cgpa_penalty + hours_penalty + text_adjustment, 0.0), 1.0)
        combined_risk = "HIGH RISK" if combined_prob > 0.7 else ("MEDIUM RISK" if combined_prob > 0.4 else "LOW RISK")
        risk          = "HIGH RISK" if yes_prob > 0.7       else ("MEDIUM RISK" if yes_prob > 0.4       else "LOW RISK")

        # Tips
        if evidence['Sleep Duration'] in ['Less than 5 hours', '5-6 hours']:
            tips.append("🛏 Aim for 7–8 hours of sleep per night to improve mental health.")
        if evidence['Dietary Habits'] == 'Unhealthy':
            tips.append("🥗 A balanced diet reduces stress hormones and supports brain health.")
        if evidence['Academic Pressure'] == 'High':
            tips.append("📚 Break study sessions into smaller blocks and take regular breaks.")
        if evidence['Financial Stress'] == 'High':
            tips.append("💰 Consider speaking to a financial advisor or student support services.")
        if evidence['Study Satisfaction'] == 'Low':
            tips.append("🎯 Explore new study techniques or speak to a counsellor about your goals.")
        try:
            if float(work_hours) >= 9: tips.append("⏱ Studying 9+ hours daily increases burnout risk — schedule proper rest.")
        except Exception: pass
        try:
            if float(cgpa) < 6.0: tips.append("📊 Low CGPA can increase academic anxiety. Consider speaking to a mentor.")
        except Exception: pass

        prediction = f"P(Depressed) = {yes_prob:.2%}"
        risk_gauge(combined_prob, combined_risk)

    return render_template('predict.html',
                           prediction=prediction, risk=risk,
                           yes_prob=yes_prob, no_prob=no_prob,
                           combined_prob=combined_prob, combined_risk=combined_risk,
                           tips=tips, cgpa=cgpa, work_hours=work_hours, user_text=user_text)


# ── AI Text Analysis — server-side Anthropic call ─────────────────────────────
@app.route('/api/analyse-text', methods=['POST'])
def analyse_text():
    """Calls Anthropic API server-side and returns keyword/sentiment JSON."""
    body = request.get_json(silent=True) or {}
    text = (body.get('text') or '').strip()

    if not text or len(text) < 5:
        return jsonify({'error': 'Text too short'}), 400

    # ── Fallback: pure keyword matching (no API key needed) ──────────────────
    if not ANTHROPIC_AVAILABLE:
        return _keyword_fallback(text)

    prompt = (
        "You are a mental health text analyser. Analyse the following student's self-report for emotional signals.\n\n"
        "Return ONLY a valid JSON object with these exact keys — no markdown, no explanation:\n"
        '{\n'
        '  "stress_keywords": ["word/phrase", ...],\n'
        '  "positive_keywords": ["word/phrase", ...],\n'
        '  "neutral_keywords": ["word/phrase", ...],\n'
        '  "distress_score": 0.0,\n'
        '  "summary": "One sentence describing the overall emotional tone."\n'
        '}\n\n'
        "Rules:\n"
        "- stress_keywords: up to 8 words/short phrases indicating stress, anxiety, sadness, overwhelm, hopelessness, fatigue\n"
        "- positive_keywords: up to 6 words/short phrases indicating resilience, hope, positivity, support\n"
        "- neutral_keywords: up to 5 neutral topic words (study, classes, etc.)\n"
        "- distress_score: float 0.0–1.0 (0=very positive/calm, 1=severe distress)\n"
        "- summary: one clear, empathetic sentence about the overall emotional tone\n\n"
        f"Text: \"{text[:600]}\""
    )

    try:
        message = _anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = message.content[0].text.strip()
        raw = raw.replace('```json', '').replace('```', '').strip()
        result = json.loads(raw)
        return jsonify(result)

    except json.JSONDecodeError:
        return _keyword_fallback(text)
    except Exception as e:
        print(f"Anthropic error: {e}")
        return _keyword_fallback(text)


def _keyword_fallback(text):
    """Pure Python keyword analysis — works without any API key."""
    text_lower = text.lower()

    stress_words   = ['stress', 'anxious', 'anxiety', 'overwhelmed', 'depressed', 'sad', 'lonely',
                      'tired', 'exhausted', 'hopeless', 'pressure', 'worried', 'fear', 'panic',
                      'struggle', 'fail', 'failing', 'hopelessness', 'cry', 'crying', 'empty',
                      'helpless', 'worthless', 'unmotivated', 'burnt out', 'burnout', 'insomnia',
                      "can't sleep", 'no energy', 'no motivation', 'giving up', 'hate']

    positive_words = ['happy', 'good', 'great', 'motivated', 'hopeful', 'better', 'support',
                      'friends', 'family', 'love', 'enjoy', 'excited', 'confident', 'grateful',
                      'improving', 'relaxed', 'calm', 'positive', 'okay', 'fine', 'well']

    neutral_words  = ['study', 'class', 'exam', 'assignment', 'college', 'university', 'work',
                      'project', 'lecture', 'course', 'semester', 'grade', 'cgpa']

    found_stress   = [w for w in stress_words   if w in text_lower][:8]
    found_positive = [w for w in positive_words if w in text_lower][:6]
    found_neutral  = [w for w in neutral_words  if w in text_lower][:5]

    # Score: ratio of stress hits to total signal words
    total = len(found_stress) + len(found_positive) + 1
    distress_score = round(min(len(found_stress) / total, 1.0), 2)

    if distress_score > 0.6:
        summary = "The text shows significant emotional distress with multiple stress signals."
    elif distress_score > 0.3:
        summary = "The text reflects moderate stress mixed with some positive elements."
    elif found_positive:
        summary = "The text has a largely positive or neutral emotional tone."
    else:
        summary = "The text appears emotionally neutral or descriptive in nature."

    return jsonify({
        'stress_keywords':   found_stress,
        'positive_keywords': found_positive,
        'neutral_keywords':  found_neutral,
        'distress_score':    distress_score,
        'summary':           summary,
        'fallback':          True
    })


@app.route('/model')
def model_page():
    return render_template('model.html', model_info=get_model_stats(model))

@app.route('/metrics')
def metrics():
    return render_template('metrics.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/api/quick-predict', methods=['POST'])
def quick_predict():
    try:
        body = request.get_json()
        evidence = {k: v for k, v in body.items() if v}
        result = infer.query(variables=['Depression'], evidence=evidence)
        return jsonify({'yes': float(result.values[1]), 'no': float(result.values[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
