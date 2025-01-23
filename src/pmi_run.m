addpath('PCA-PMI-master/lib/');

% Define the list of CSV files
csv_files = {'Muraro', 'Baron_Mouse', 'Segerstolpe', 'Baron_Human', ...
            'Zhang_T', 'Kang_ctrl', 'AMB', 'TM', 'Zheng68K'};

% File paths
data_dir = 'dataset/pre_data/scRNAseq_datasets_hvgs/';
split_dir = fullfile(data_dir, 'splits/');
output_base_dir = 'dataset/5fold_data/';

% Set threshold and maximum order
lambda = 0.05;
order0 = 1;

% Loop through each CSV file
for i = 1:length(csv_files)
    csv_file = csv_files{i};
    disp(['Processing CSV file: ', csv_file]);
    % Construct CSV file path
    data_file = fullfile(data_dir, [csv_file '_hvgs.csv']);
    
    % Read the CSV data, ignoring row and column names, transpose so that genes are in rows and cells are in columns
    data_matrix = readmatrix(data_file, 'Range', [2, 2]);
    data_matrix_tr = data_matrix';

    % Loop through the five folds of TXT files
    % for fold = 1:5
    for fold = 1:5
        % Construct TXT file path
        fold_txt = fullfile(split_dir, [csv_file '_train_f' num2str(fold) '.txt']);
        
        % Read index file
        indices = load(fold_txt);
        % Change indices from 0-based to 1-based
        indices = indices + 1;

        % Extract training data
        data_train = data_matrix_tr(:, indices);

        % Call pca_pmi function
        [G, Gval, order] = pca_pmi(data_train, lambda, order0);

        % Get the indices of the existing edges
        [row_idx, col_idx] = find(triu(G, 1)); % Only get non-zero values from the upper triangle to avoid duplicates

        % Extract the corresponding Gval values
        weights = Gval(sub2ind(size(Gval), row_idx, col_idx));

        % Change indices from 1-based to 0-based
        row_idx = row_idx - 1;
        col_idx = col_idx - 1;

        % Construct result matrix (three-column format)
        result = [row_idx, col_idx, weights];

        % Output the total number of edges
        num_edges = size(result, 1); % The number of rows corresponds to the number of edges
        disp(['Total number of edges: ', num2str(num_edges)]);

        % Construct output directory and file path
        output_dir = fullfile(output_base_dir, csv_file, 'pca_pmi');
        if ~exist(output_dir, 'dir')
            mkdir(output_dir);
        end

        output_filename = fullfile(output_dir, ['pca_pmi_f' num2str(fold) '.tsv']);

        % Save the result as a TSV file
        writematrix(result, output_filename, 'FileType', 'text', 'Delimiter', '\t');

        % Display save information
        disp(['Save successfully: ', output_filename]);
    end
end