from flask import Flask, render_template, request, send_file, url_for, jsonify
import os
import pandas as pd
from modules.model_loader import load_artifacts
from modules.preprocess import prepare_input
from modules.predictor import predict_from_array
from modules.report_engine import generate_report
from modules.face_module import find_match   # ← ADD THIS


app = Flask(__name__)

# -------------------------
# PATHS
# -------------------------
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

FACE_DB = os.path.join(UPLOAD_FOLDER, "faces_db")
os.makedirs(FACE_DB, exist_ok=True)

PATIENT_CSV = os.path.join("saved_models", "patients.csv")


# Load model once
model, scaler, selected_features = load_artifacts()
predict_fn = predict_from_array(model)


# ---------------------------
#         HOME PAGE
# ---------------------------
@app.route("/")
def index():
    return render_template("index.html", features=selected_features)


# ---------------------------
#     FACE UPLOAD & MATCH
# ---------------------------
# ---------------------------
#     FACE UPLOAD & MATCH
# ---------------------------
@app.route("/upload_face", methods=["POST"])
def upload_face():
    file = request.files.get("face")
    if not file:
        return jsonify({"error": "No file received"}), 400

    # -----------------------------
    # save file with unique temp name
    # -----------------------------
    import uuid
    ext = os.path.splitext(file.filename)[1]  # keep original extension
    temp_name = f"{uuid.uuid4().hex}{ext}"    # 8acbd6a0dd.jpg etc.
    temp_path = os.path.join(app.config["UPLOAD_FOLDER"], temp_name)
    file.save(temp_path)

    # attempt face matching
    matched = find_match(temp_path)

    # -----------------------------------
    # CASE 1: no match -> NEW USER
    # -----------------------------------
    if not matched:
        return jsonify({
            "found": False,
            "temp_img": temp_name
        })

    # -----------------------------------
    # CASE 2: match found -> EXISTING USER
    # -----------------------------------
    matched_name = os.path.splitext(os.path.basename(matched))[0]  # remove .jpg

    # load previous patient data if exists
    if os.path.exists(PATIENT_CSV):
        df = pd.read_csv(PATIENT_CSV)
        rec = df[df["name"] == matched_name]
        if not rec.empty:
            details = rec.iloc[0].to_dict()
            details.pop("name", None)
            return jsonify({
                "found": True,
                "name": matched_name,
                "details": details,
                "temp_img": temp_name 
            })

    return jsonify({
        "found": True,
        "name": matched_name,
        "details": {},
        "temp_img": temp_name
    })




# ---------------------------
#         PREDICT
# ---------------------------

@app.route("/predict", methods=["POST"])
def predict():
    form = request.form.to_dict()

    # DEBUG BLOCK — must be immediately after "form"
    patient_name = form.get("patient_name", "").strip()
    uploaded_face = form.get("uploaded_face_filename", "").strip()

    print("\n========= DEBUG FACE IMPORT =========")
    print("patient_name →", patient_name)
    print("uploaded_face filename →", uploaded_face)
    print("=====================================\n")

    # Convert medical inputs → floats
    input_dict = {}
    for feat in selected_features:
        val = form.get(feat, "0")
        try:
            input_dict[feat] = float(val)
        except:
            input_dict[feat] = 0.0


    # ---- SAVE FACE IF NEW USER UPLOADED ONE ----
    uploaded_face = form.get("uploaded_face_filename", "").strip()
    if patient_name and uploaded_face:
        temp_path = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_face)

        if os.path.exists(temp_path):
            # keep original extension (.jpg, .jpeg, .png, .webp)
            ext = os.path.splitext(uploaded_face)[1]
            final_path = os.path.join(FACE_DB, f"{patient_name}{ext}")

            # delete previous images of the same user
            for f in os.listdir(FACE_DB):
                if f.startswith(patient_name):
                    os.remove(os.path.join(FACE_DB, f))

            os.rename(temp_path, final_path)
            print(f"[INFO] Saved new face to FACE_DB: {final_path}")

    # ---- ML Prediction ----
    X = prepare_input(input_dict, selected_features, scaler)
    label, prob = predict_fn(X)
    pred = {"label": label, "probability": prob}

    # ---- Save patient record to CSV ----
    new_record = {"name": patient_name}
    new_record.update(input_dict)

    df = pd.DataFrame([new_record])
    df.to_csv(
        PATIENT_CSV,
        mode="a",
        header=not os.path.exists(PATIENT_CSV),
        index=False
    )

    # ---- Find face image for PDF ----
    face_image_path = None
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        check = os.path.join(FACE_DB, f"{patient_name}{ext}")
        if os.path.exists(check):
            face_image_path = check
            break

    # ---- Generate PDF report ----
    pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], "report.pdf")
    generate_report(
        patient_name,
        patient_info=input_dict,
        prediction_result=pred,
        face_image_path=face_image_path,
        filepath=pdf_path
    )

    # redirect to result page
    return render_template(
        "result.html",
        result=pred,
        patient_name=patient_name,
        pdf_url=url_for("download_report")
    )


# ---------------------------
#     DOWNLOAD PDF
# ---------------------------
@app.route("/download_report")
def download_report():
    path = os.path.join(app.config["UPLOAD_FOLDER"], "report.pdf")
    return send_file(path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
