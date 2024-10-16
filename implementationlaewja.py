import os
import random
import inference
import cv2
import csv

# Set the directory and image selection parameters
directory = "test_nai_hong"
num_images = 10
ic = {
    "AND" : "74LS08",
    "OR": "74LS32",
    "NOT": "74LS04",
    "XOR": "74LS86",
    "NAND": "74LS00",
    "NOR": "74LS02",
    "XNOR": "74LS266",
}

# Get model for inference
model = inference.get_model("digital-schematics-detection/1")

# List all files in the directory and select random 10 images
all_files = os.listdir(directory)
image_files = [f for f in all_files if f.endswith(('.png', '.jpg', '.jpeg'))]  # filter image files
selected_images = random.sample(image_files, num_images)

# Create CSV file to store results
csv_filename = "used_ic_results.csv"
csv_headers = ["Image Name", "AND", "OR", "NOT", "XOR", "NAND", "NOR", "XNOR"]

with open(csv_filename, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(csv_headers)  # Write the header row

    for img_name in selected_images:
        # Load image
        img_path = os.path.join(directory, img_name)
        image = cv2.imread(img_path)

        # Perform inference
        result = model.infer(image=img_path)

        # Initialize gate counts
        gates = {
            "AND": 0,
            "OR": 0,
            "NOT": 0,
            "XOR": 0,
            "NAND": 0,
            "NOR": 0,
            "XNOR": 0,
        }

        # Bounding boxes and classification processing
        bboxes = []
        for pred in result[0].predictions:
            x_center = pred.x
            y_center = pred.y
            width = pred.width
            height = pred.height

            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            gates[pred.class_name] += 1
            bboxes.append({
                "class": pred.class_name,
                "pos": (x1, y1, x2, y2),
                "conf": pred.confidence
            })

        # Draw bounding boxes on the image
        debug_image = image.copy()
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox["pos"]
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{bbox['class']}: {bbox['conf']:.2f}"
            cv2.putText(debug_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the image
        cv2.imshow("Detected Gates", debug_image)
        cv2.waitKey(0)

        # Print used ICs in the terminal
        print(f"Image: {img_name}")
        for gate in gates:
            if gates[gate] > 0:
                used_ics = gates[gate] // 4 if gates[gate] % 4 == 0 else gates[gate] // 4 + 1
                print(f"{ic[gate]}: {used_ics}")
        print(gates)

        # Export ICs used to CSV
        writer.writerow([
            img_name, 
            gates["AND"], 
            gates["OR"], 
            gates["NOT"], 
            gates["XOR"], 
            gates["NAND"], 
            gates["NOR"]
        ])

cv2.destroyAllWindows()

print(f"Results saved to {csv_filename}")
