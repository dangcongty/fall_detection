import re

import matplotlib.pyplot as plt

# Raw text input
text = """
URFall - Le2i
Event-Based Evaluation at IoU: 0.1
  Precision: 0.581
  Recall:    1.000
  F1 Score:  0.703
Event-Based Evaluation at IoU: 0.2
  Precision: 0.506
  Recall:    0.900
  F1 Score:  0.620
Event-Based Evaluation at IoU: 0.3
  Precision: 0.406
  Recall:    0.800
  F1 Score:  0.520
Event-Based Evaluation at IoU: 0.4
  Precision: 0.277
  Recall:    0.500
  F1 Score:  0.342
Event-Based Evaluation at IoU: 0.5
  Precision: 0.160
  Recall:    0.250
  F1 Score:  0.183
Event-Based Evaluation at IoU: 0.6
  Precision: 0.050
  Recall:    0.050
  F1 Score:  0.050
Event-Based Evaluation at IoU: 0.7
  Precision: 0.000
  Recall:    0.000
  F1 Score:  0.000
Event-Based Evaluation at IoU: 0.8
  Precision: 0.000
  Recall:    0.000
  F1 Score:  0.000
Event-Based Evaluation at IoU: 0.9
  Precision: 0.000
  Recall:    0.000
  F1 Score:  0.000
"""

# Extract data using regular expressions
ious = [float(i.group(1)) for i in re.finditer(r"IoU: ([0-9.]+)", text)]
precision = [float(p.group(1)) for p in re.finditer(r"Precision: ([0-9.]+)", text)]
recall = [float(r.group(1)) for r in re.finditer(r"Recall:\s+([0-9.]+)", text)]
f1_score = [float(f.group(1)) for f in re.finditer(r"F1 Score:\s+([0-9.]+)", text)]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(ious, precision, marker='o', label='Precision')
plt.plot(ious, recall, marker='s', label='Recall')
plt.plot(ious, f1_score, marker='^', label='F1 Score')

plt.title('Event-Based Evaluation Metrics vs IoU Thresholds (Le2i - Urfall)')
plt.xlabel('IoU Threshold')
plt.ylabel('Score')
plt.ylim(0, 1.05)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('PR-curved.jpg')
