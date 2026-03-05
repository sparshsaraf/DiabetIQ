import matplotlib
matplotlib.use("Agg")  # non-GUI backend -> no Tkinter errors

from fpdf import FPDF
import matplotlib.pyplot as plt
import os
FONT_PATH = os.path.join(os.getcwd(), "fonts", "DejaVuSans.ttf")

PRIMARY_COLOR = (40, 90, 255)      # blue
SUCCESS_COLOR = (0, 180, 90)       # green
DANGER_COLOR  = (220, 50, 50)      # red
LIGHT_GRAY    = (240, 240, 240)

# ---------------------------
# Create chart helper
# ---------------------------
def create_chart(title, labels, values, filepath):
    plt.figure(figsize=(4, 2.5))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylim(0,100)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


# ---------------------------
# Modern PDF Report
# ---------------------------
def generate_report(
        patient_name,
        patient_info,
        prediction_result,
        face_image_path=None,
        filepath="uploads/report.pdf"
):

    # --- Temp folder for charts ---
    temp_dir = "temp_charts"
    os.makedirs(temp_dir, exist_ok=True)

    # Probability chart
    prob_chart = os.path.join(temp_dir, "probability.png")
    create_chart(
        "Risk Probability (%)",
        ["Probability"],
        [prediction_result["probability"] * 100],
        prob_chart
    )

    # Health metrics chart
    h_labels, h_values = [], []
    for feature in ["Glucose", "BMI", "Insulin"]:
        if feature in patient_info:
            h_labels.append(feature)
            h_values.append(patient_info[feature])

    health_chart = None
    if h_labels:
        health_chart = os.path.join(temp_dir, "health.png")
        create_chart("Key Health Indicators", h_labels, h_values, health_chart)

    # ---------------------------
    # Start PDF
    # ---------------------------
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # register unicode font
    pdf.add_font("DejaVu", "", FONT_PATH, uni=True)
    pdf.set_font("DejaVu", "", 14)

    # --- HEADER BAR ---
    pdf.set_fill_color(*PRIMARY_COLOR)
    pdf.rect(0, 0, 210, 25, "F")

    pdf.set_text_color(255, 255, 255)
    pdf.set_font("DejaVu", "", 20)
    pdf.set_xy(10, 8)
    pdf.cell(0, 10, "Smart Health - Diagnostic Report", ln=True)

    # FACE IMAGE
    from PIL import Image

    if face_image_path and os.path.exists(face_image_path):
        try:
            temp_face = os.path.join(temp_dir, "face.png")
            img = Image.open(face_image_path).convert("RGB")
            img.save(temp_face)
            pdf.image(temp_face, x=165, y=5, w=35)
            print("[INFO] Face image added to report")
        except Exception as e:
            print("[ERROR] Could not embed face image:", e)


    pdf.ln(20)

    # ---------------------------
    # Patient Name
    # ---------------------------
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("DejaVu", "", 14)
    pdf.cell(0, 10, f"Patient: {patient_name}", ln=True)

    # ---------------------------
    # Prediction Box
    # ---------------------------
    label = prediction_result["label"]
    outcome = "Diabetic" if label == 1 else "Not Diabetic"
    prob = prediction_result["probability"] * 100

    color = DANGER_COLOR if label == 1 else SUCCESS_COLOR

    pdf.set_fill_color(*LIGHT_GRAY)
    pdf.rect(10, pdf.get_y(), 190, 18, "F")

    pdf.set_xy(10, pdf.get_y() + 2)
    pdf.set_font("DejaVu", "", 12)
    pdf.cell(0, 8, f"Prediction: {outcome}", ln=True)

    pdf.set_font("DejaVu", "", 11)
    pdf.set_text_color(*color)
    pdf.cell(0, 6, f"Risk Probability: {prob:.2f}%", ln=True)
    pdf.set_text_color(0, 0, 0)

    pdf.ln(10)

    # ---------------------------
    # Probability Chart
    # ---------------------------
    pdf.set_font("DejaVu", "", 12)
    pdf.cell(0, 10, "Risk Probability Chart:", ln=True)
    pdf.image(prob_chart, x=20, w=160)
    pdf.ln(8)

    # ---------------------------
    # Health Chart
    # ---------------------------
    if health_chart:
        pdf.set_font("DejaVu", "", 12)
        pdf.cell(0, 10, "Health Metrics Chart:", ln=True)
        pdf.image(health_chart, x=20, w=160)
        pdf.ln(8)

    # ---------------------------
    # Patient Input Table
    # ---------------------------
    pdf.set_font("DejaVu", "", 12)
    pdf.cell(0, 10, "Patient Submitted Values:", ln=True)

    pdf.set_font("DejaVu", "", 11)
    for k, v in patient_info.items():
        pdf.cell(0, 7, f"{k}: {v}", ln=True)

    pdf.ln(5)

    # ---------------------------
    # Explanation
    # ---------------------------
    pdf.set_font("DejaVu", "", 12)
    pdf.cell(0, 10, "How This Result Was Determined:", ln=True)

    pdf.set_font("DejaVu", "", 11)
    pdf.multi_cell(
        0, 6,
        "This prediction is generated using a machine learning model that analyzes "
        "key health indicators such as glucose, BMI, and insulin. Higher values in these "
        "metrics often correlate with increased diabetes risk. The probability score above "
        "represents the model's confidence in its prediction."
    )

    pdf.ln(5)

    # ---------------------------
    # Recommendations
    # ---------------------------
    pdf.set_font("DejaVu", "", 12)
    pdf.cell(0, 10, "Recommendations:", ln=True)

    pdf.set_font("DejaVu", "", 11)

    if label == 1:
        recs = [
            "• Follow a low-sugar, low-carb diet",
            "• Monitor blood sugar daily",
            "• Engage in 30–45 mins exercise",
            "• Maintain healthy BMI",
            "• Consult a specialist for regular checkups"
        ]
    else:
        recs = [
            "• Maintain active lifestyle",
            "• Keep BMI in healthy range",
            "• Limit sugar-heavy foods",
            "• Routine glucose check yearly",
            "• Stay hydrated and eat balanced meals"
        ]

    for r in recs:
        pdf.cell(0, 6, r, ln=True)

    # ---------------------------
    # Save PDF
    # ---------------------------
    pdf.output(filepath)
    print(f"[INFO] PDF Exported → {filepath}")
