from sklearn.metrics import classification_report

# Dummy predictions for illustration
true = [0, 1, 1, 0]
pred = [0, 1, 0, 0]

print(classification_report(true, pred))