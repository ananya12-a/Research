from load_and_train import load_data, full_model

X_train, X_add, y_train, y_add,X_test, X_test_final, y_test, y_test_final, input_shape= load_data(5/6)
accuracy, model_int = full_model(X_train, y_train, X_test, y_test, "model_digit_before", input_shape, iter_nam="initial", toPlot=False, toStoreQuery=True, num_epoch=10)
print("Initial accuracy: " + str(accuracy))


