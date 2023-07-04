from sklearn.model_selection import cross_val_score

# Menentukan rentang nilai K yang akan dievaluasi
k_values = range(1, 10)

# Melakukan validasi silang untuk setiap nilai K
accuracy_scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    accuracy_scores.append(scores.mean())

# Mencari nilai K terbaik
best_k = k_values[accuracy_scores.index(max(accuracy_scores))]
print("Nilai K terbaik: ", best_k)

# Melatih model KNN dengan nilai K terbaik
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train, y_train)

# Memprediksi kelas dari data uji dengan model KNN terbaik
y_pred_best = best_knn.predict(X_test)

# Menghitung akurasi prediksi dengan model KNN terbaik
accuracy_best = accuracy_score(y_test, y_pred_best)
print("Akurasi dengan K terbaik: {:.2f}%".format(accuracy_best * 100))
