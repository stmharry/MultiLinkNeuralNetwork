load('projectionAD.mat');

model = svmtrain(target_train_labels, target_train_features, '-q');
[target_test_labels_predicted, acc, ~] = svmpredict(target_test_labels, target_test_features, model, '-q');
fprintf('Without amazon: %.3f\n', acc(1));

model = svmtrain([source_labels; target_train_labels], [source_features; target_train_features], '-q');
[target_test_labels_predicted, acc, ~] = svmpredict(target_test_labels, target_test_features, model, '-q');
fprintf('With amazon: %.3f\n', acc(1));

