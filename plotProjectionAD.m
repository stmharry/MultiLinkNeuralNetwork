% {
load('projectionAD.mat');

source_select = 1:length(source_labels);
target_train_select = length(source_labels) + (1:length(target_train_labels));
target_test_select = length(source_labels) + length(target_train_labels) + (1:length(target_test_labels));

features = [source_features; target_train_features; target_test_features];
labels = [source_labels; target_train_labels; target_test_labels];

features = tsne(features, []);
source_features = features(source_select, :);
target_train_features = features(target_train_select, :);
target_test_features = features(target_test_select, :);

start = zeros(31, 2);
for i = 1:31
    start(i, :) = mean(features(labels == i, :));
end
[index, center] = kmeans(features, 31, 'Start', start);
featMin = min(features);
featMax = max(features);
%}

split = 9;
small = 48;
big = 55;
source_str = char((source_labels <= split) .* (source_labels + small) + (source_labels > split) .* (source_labels + big));
target_train_str = char((target_train_labels <= split) .* (target_train_labels + small) + (target_train_labels > split) .* (target_train_labels + big));
target_test_str = char((target_test_labels <= split) .* (target_test_labels + small) + (target_test_labels > split) .* (target_test_labels + big));

figure(sum('postTest')); clf;
text(source_features(:, 1), source_features(:, 2), source_str);
text(target_test_features(:, 1), target_test_features(:, 2), target_test_str, 'Color', [0, 0, 1]);
text(target_train_features(:, 1), target_train_features(:, 2), target_train_str, 'Color', [1, 0, 0], 'FontWeight', 'bold');
text(center(:, 1), center(:, 2), num2str((1:31)'), 'Color', [0, 1, 0], 'FontWeight', 'bold');

str = ['Black: Source', char(10), 'Blue: Target Test', char(10), 'Red: Target Train', char(10), 'Green: Kmeans Center'];
text(featMin(1), featMin(2), str, 'VerticalAlignment', 'bottom', 'FontSize', 24);
set(gca, 'xLim', [featMin(1), featMax(1)], 'yLim', [featMin(2), featMax(2)], 'Visible', 'off');
set(gcf, 'Color', [1, 1, 1]);
set(findall(gcf, 'Type', 'Text'), 'FontSize', 12, 'Clipping', 'on', 'HitTest', 'off');
