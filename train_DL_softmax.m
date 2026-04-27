function train_DL_softmax()
    load dataset_final.mat final_dataset;
    
    X = final_dataset(:,1:4);
    Y_raw = final_dataset(:,5:16); % Y gốc chứa số lượng cổng tuyệt đối
    
    % ====================================================================
    % [THAY ĐỔI 1]: CHUẨN HÓA DỮ LIỆU ĐẦU RA Y SANG TỶ LỆ (PROPORTIONS)
    % ====================================================================
    N_values = X(:, 4); % Lấy cột N (số lượng cổng) từ X
    % Chia mỗi hàng của Y cho N tương ứng để ra tỷ lệ (tổng mỗi hàng sẽ = 1)
    Y_normalized = Y_raw ./ N_values; 
    
    % Shuffle dữ liệu
    idx = randperm(size(X,1));
    X = X(idx,:);
    Y = Y_normalized(idx,:); % Dùng Y đã chuẩn hóa để train
    
    % Train/test split (80% train)
    Ntrain = round(0.8*size(X,1));
    Xtrain = X(1:Ntrain,:);
    Ytrain = Y(1:Ntrain,:);
    Xtest  = X(Ntrain+1:end,:);
    Ytest  = Y(Ntrain+1:end,:);
    
    % ====================================================================
    % [THAY ĐỔI 2]: THÊM SOFTMAX LAYER VÀO KIẾN TRÚC MẠNG
    % ====================================================================
    layers = [
        featureInputLayer(4)
        fullyConnectedLayer(128)
        batchNormalizationLayer 
        reluLayer
        fullyConnectedLayer(256)
        reluLayer
        fullyConnectedLayer(128)
        reluLayer
        fullyConnectedLayer(12)
        softmaxLayer       % ÉP BỘ TỐI ƯU: Tổng 12 nơ-ron luôn chính xác = 1 (100%)
        regressionLayer    % Tính sai số MSE dựa trên tỷ lệ %
    ];
    
    % TÍNH TOÁN THÔNG SỐ TRAIN
    numTrainImages = size(Xtrain, 1);
    miniBatchSize = 256; 
    
    valFrequency = floor(numTrainImages/miniBatchSize); 
    options = trainingOptions('adam', ...
        'MaxEpochs', 50, ...          
        'MiniBatchSize', miniBatchSize, ...
        'InitialLearnRate', 1e-3, ...
        'LearnRateSchedule', 'piecewise', ... 
        'LearnRateDropFactor', 0.5, ...
        'LearnRateDropPeriod', 15, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', {Xtest, Ytest}, ...
        'ValidationFrequency', valFrequency, ... 
        'Plots', 'training-progress', ...
        'ExecutionEnvironment', 'auto', ... 
        'Verbose', false);
        
    % Train mô hình
    net = trainNetwork(Xtrain, Ytrain, layers, options);
    
    % Đổi tên file lưu để phân biệt với mô hình cũ
    save trained_net_softmax.mat net;
    disp("Train hoàn tất với phương pháp Softmax/Tỷ lệ!");
end